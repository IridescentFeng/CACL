import clip
from PIL import Image
import torch
import warnings
import os
import argparse
import warnings
from coop_model import *
from datasets import build_dataset, build_dataset_with_caption
import tqdm
import torch.optim as optim
from torch.autograd import Variable
from metrics import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from spc_average_loss import *

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('auto-prompt clip', add_help=False)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default='0', type=int, help='seed')
    parser.add_argument('--pretrain_clip', default='RN50', type=str, choices=['RN50', 'ViT16', 'ViT32'],
                        help='pretrained clip backbone')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--dataset', default='coco-lt', type=str, choices=['coco-lt', 'voc-lt'])
    parser.add_argument('--ctx_init', default='A photo of a', type=str,
                        help='init context prompt')
    parser.add_argument('--n_ctx', default=4, type=int, help='length of context prompt when initializing')
    parser.add_argument('--class_token_position', default='end', type=str, help='position of class token')
    parser.add_argument('--training_method', default='lmpt', type=str, choices=['coop', 'cocoop', 'dualcoop', 'lmpt'],
                        help='training method (coop cocoop)')
    parser.add_argument('--csc', action='store_true', default=False,
                        help='class-specific contexts (if False then initialize a generic context)')
    parser.add_argument('--thre', default=0.3, type=float, help='threshold value')
    return parser


class ResNet50GradVisualizer:
    def __init__(self, image_encoder):
        self.image_encoder = image_encoder
        self._init_grad_records()
        self._register_hooks()

    def _init_grad_records(self):
        self.grad_data = {
            'layer4_conv': [],
            'attnpool': []
        }

    def _register_hooks(self):
        # 监控最后一个卷积层
        layer4_conv = self.image_encoder.layer4[-1].conv3
        layer4_conv.register_backward_hook(self._save_grad('layer4_conv'))

        # 监控投影池化层
        attnpool = self.image_encoder.attnpool
        attnpool.register_backward_hook(self._save_grad('attnpool'))

    def _save_grad(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.grad_data[name].append(grad_output[0].abs().mean().item())

        return hook

    def visualize(self):
        plt.figure(figsize=(10, 4))

        # 绘制所有训练步骤的梯度变化
        plt.plot(self.grad_data['layer4_conv'], 'b-o', label='Layer4 Conv')
        plt.plot(self.grad_data['attnpool'], 'r-s', label='AttnPool')

        plt.title("CLIP ResNet50")
        plt.xlabel("Training Steps")
        plt.ylabel("Gradient Magnitude")
        plt.legend()
        plt.grid(True)
        plt.show()




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
    clip_model, preprocess = clip.load(pretrain_clip_path, device='cpu', jit=False)  # Must set jit=False for training

    def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16

    if args.dataset == 'coco-lt':
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
    elif args.dataset == 'voc-lt' or args.dataset == 'voc':
        dataset_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    print("Building custom CLIP")
    model = CustomCLIP(args, classnames=dataset_classes, clip_model=clip_model)
    # 初始化监控器
    #grad_monitor = PromptGradMonitor(model.prompt_learner)
    image_encoder = clip_model.visual
    visualizer = ResNet50GradVisualizer(image_encoder)

    model.to(args.device)
    '''if args.dataset == 'voc-lt':
        model.load_state_dict(torch.load('../checkpoint_voc/ctx_8_32_lmpt_RN50_voc-lt_asl_supcon_csc.pt'))
    elif args.dataset == 'coco-lt':
        model.load_state_dict(
            torch.load('../checkpoint_sigmoid/32_0.5_lmpt_RN50_coco-lt_asl_supcon_csc.pt'))'''

    #test_dataset = build_dataset(dataset=args.dataset, split='test')
    train_dataset = build_dataset_with_caption(dataset=args.dataset, split='train', clip_model=clip_model)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
    optimizer = optim.Adam(model.prompt_learner.parameters(), lr=0.01, weight_decay=1e-4)
    #model.eval()
    cls_num_list = train_dataset.get_cls_num_list()
    sf = nn.Softmax(dim=1)
    for epoch in range(50):
        model.train()
        gt_labels = []
        predict_p = []
        for data in tqdm.tqdm(train_loader):

            inputs, sample1, sample2, labels, captions, captions_ = data
            captions = Variable(captions.cuda())
            captions_ = Variable(captions_.cuda())

            batch_size = labels.shape[0]
            labels = labels.to(torch.float32)
            labels = torch.squeeze(labels, 1)
            inputs = torch.cat([inputs, sample1, sample2], dim=0)
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()

            outputs, image_features,_ = model(inputs)

            outputs, _, __ = torch.split(outputs, [batch_size, batch_size, batch_size], dim=0)
            _, f2, f3 = torch.split(image_features, [batch_size, batch_size, batch_size], dim=0)
            features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)

            gt_labels.extend(labels.cpu().numpy().tolist())
            predict_p.extend(sf(outputs).cpu().detach().numpy())

            contrastive_loss = aveSupConLoss(cls_num_list)
            #contrastive_loss = BalSCL(cls_num_list)
            # centers = centers[:80]
            loss_2 = contrastive_loss(features, labels)


            loss_2.backward()
            optimizer.step()
    visualizer.visualize()







if __name__ == '__main__':
    parser = argparse.ArgumentParser('auto-prompt clip', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
