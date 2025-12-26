import clip
import numpy as np
from PIL import Image
import torch
import warnings
import os
import argparse
import warnings
from coop_model import *
from datasets import build_dataset
import tqdm
from torch.autograd import Variable
from metrics import *
from untils import ranking_loss,pairwise_accuracy

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('auto-prompt clip', add_help=False)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default='0', type=int, help='seed')
    parser.add_argument('--pretrain_clip', default='RN50', type=str, choices=['RN50', 'ViT16', 'ViT32'], help='pretrained clip backbone')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--dataset', default='coco-lt', type=str, choices=['coco-lt','voc-lt'])
    parser.add_argument('--ctx_init', default='This is a professional photograph of a specific item, which is clearly visible and centered in the frame', type=str, help='init context prompt')
    parser.add_argument('--n_ctx', default=4, type=int, help='length of context prompt when initializing')
    parser.add_argument('--class_token_position', default='end', type=str, help='position of class token')
    parser.add_argument('--training_method', default='lmpt', type=str, choices=['coop', 'cocoop', 'dualcoop', 'lmpt'], help='training method (coop cocoop)')
    parser.add_argument('--csc', action='store_true', default=False, help='class-specific contexts (if False then initialize a generic context)')
    parser.add_argument('--thre', default=0.3, type=float, help='threshold value')
    return parser

def main(args):
    print(args)

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    """ 
    model
    """
    if args.pretrain_clip == "RN50":
        pretrain_clip_path = '../pretrained/RN50.pt'
    elif args.pretrain_clip == "ViT16":
        pretrain_clip_path = '../pretrained/ViT-B-16.pt'

    print(f"Loading CLIP (backbone: {args.pretrain_clip})")
    clip_model, preprocess = clip.load(pretrain_clip_path, device='cpu', jit=False) # Must set jit=False for training

    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 
    
    clip.model.convert_weights(clip_model) # Actually this line is unnecessary since clip by default already on float16

    if args.dataset=='coco-lt':
        dataset_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
            'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
            'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
            ]
    elif args.dataset=='voc-lt' or args.dataset=='voc': 
        dataset_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor'
            ]

    print("Building custom CLIP")
    model = CustomCLIP(args, classnames=dataset_classes, clip_model=clip_model)
    model.to(args.device)
    if args.dataset=='voc-lt': 
        model.load_state_dict(torch.load('../checkpoint_newhin_voc/noaug_32_lmpt_RN50_voc-lt_asl_supcon_csc.pt'))
    elif args.dataset=='coco-lt': 
        model.load_state_dict(torch.load('../checkpoint_newhin_coco/rand_rand_32_lmpt_RN50_coco-lt_asl_supcon_csc.pt'))

    test_dataset = build_dataset(dataset=args.dataset, split='test')
    test_loader = torch.utils.data.DataLoader(
                                            test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            drop_last=False
                                        )
    tp, fp, fn, tn = 0, 0, 0, 0
    sf = nn.Softmax(dim=1)
    all_labels = []
    all_preds = []
    with torch.no_grad():
        preds = []
        targets = []
        for data in tqdm.tqdm(test_loader):
            input, target = data
            target = target
            target = target.max(dim=1)[0]
            outputs, _, _ = model(input.cuda())
            #predict_p.extend(sf(outputs).cpu().detach().numpy())
            #output = sf(model(input.cuda())).cpu()
            output = sf(outputs).cpu()
            # for mAP calculation


            targets.extend(target.cpu().numpy().tolist())
            preds.extend(output.detach().numpy())


            mAP, APs = eval_map(preds, targets)
            
            # measure accuracy and record loss
            pred = output.data.gt(args.thre).long()

            tp += (pred + target).eq(2).sum(dim=0)
            fp += (pred - target).eq(1).sum(dim=0)
            fn += (pred - target).eq(-1).sum(dim=0)
            tn += (pred + target).eq(0).sum(dim=0)

            this_tp = (pred + target).eq(2).sum()
            this_fp = (pred - target).eq(1).sum()
            this_fn = (pred - target).eq(-1).sum()
            this_tn = (pred + target).eq(0).sum()

            this_prec = this_tp.float() / (
                this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
            this_rec = this_tp.float() / (
                this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

            p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                                i] > 0 else 0.0
                for i in range(len(tp))]
            r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                                i] > 0 else 0.0
                for i in range(len(tp))]
            f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
                i in range(len(tp))]

            mean_p_c = sum(p_c) / len(p_c)
            mean_r_c = sum(r_c) / len(r_c)
            mean_f_c = sum(f_c) / len(f_c)

            p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
            r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
            f_o = 2 * p_o * r_o / (p_o + r_o)

        print("test per-class precision:{}, recall:{}, f1:{}".format(mean_p_c, mean_r_c, mean_f_c))
        print("test overall precision:{}, recall:{}, f1:{}".format(p_o, r_o, f_o))
        print("test mAP:{}".format(mAP))
        ltAnalysis(APs, args.dataset)

        # all_labels = torch.cat(all_labels, dim=0)
        # all_preds = torch.cat(all_preds, dim=0)

        all_labels.append(targets)
        all_preds.append(preds)

        Y_preds = np.concatenate(all_preds, axis=0)
        Y_true = np.concatenate(all_labels, axis=0)

        np.save('../plot/y_pred.npy',Y_preds)
        np.save('../plot/y_true.npy', Y_true)


        # rloss = ranking_loss(all_labels, all_preds)
        # pacc = pairwise_accuracy(all_labels, all_preds)
        # print('Rankloss: {}, Pairwise: {}'.format(rloss, pacc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('auto-prompt clip', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
