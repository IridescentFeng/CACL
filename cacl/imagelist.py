import os
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import clip
from randaugment import rand_augment_transform,AutoAugmentOp

class ImageListCaption(object):

    def __init__(self, root, list_file, caption_file, label_file, nb_classes, split, clip_model):
        with open(list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # ['1','2'] -> [1,2]
        def list_str2int(x: list):
            return [int(i) for i in x]

        # image path & label
        self.fns = []
        self.labels = [] # one-hot e.g. [0,0,1,0,0,0,1...]
        '''for line in lines:
            self.fns.append(line.strip().split(' ')[0])
            label = list_str2int(line.strip().split(' ')[1:])
            # one-hot
            n_label = [[1 if i in label else 0 for i in range(nb_classes)]]
            y_all = np.ndarray([0, nb_classes])
            y_all = np.concatenate((y_all, np.array(n_label)))
            self.labels.append(y_all)
        self.labels = np.array(self.labels)
        self.fns = [os.path.join(root, fn) for fn in self.fns]'''
        for line in lines:
            # 提取文件名和标签
            parts = line.strip().split(' ')
            fn = parts[0]  # 文件名
            labels = list_str2int(parts[1:])  # 标签列表

            # 生成 one-hot 编码
            one_hot = np.zeros(nb_classes, dtype=np.int32)  # 初始化 one-hot 向量
            for label in labels:
                one_hot[label] = 1  # 标记对应类别

            # 添加到列表
            self.fns.append(fn)
            self.labels.append(one_hot)

        self.fns = [os.path.join(root, fn) for fn in self.fns]  # 拼接文件路径
        self.labels = np.array(self.labels)
        # captions
        captions = []
        with open(caption_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            captions.append(' '.join(line.strip().split()[1:])[:75])
        self.captions_tokenized = torch.cat([clip.tokenize(c) for c in captions])
        dtype = clip_model.dtype
        with torch.no_grad():
            self.captions_embedding = clip_model.token_embedding(self.captions_tokenized).type(dtype)

        # label name
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.categories = [{"id": int(line.strip().split(' ')[-1]), "name": line.strip().split(' ')[0]} for line in lines]

        self.split = split

        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(32 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
        '''augmentation_randncls = [
            transforms.RandomResizedCrop(28, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
            transforms.ToTensor(),
            Normalize,
        ]'''
        augmentation_randnclsstack = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(100, 2), ra_params),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        transform_train = transforms.Compose(augmentation_sim)

        augmentation_auto = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            AutoAugmentOp(name='Rotate', prob=0.7, magnitude=5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        transform_train1 = transforms.Compose(augmentation_sim)
        transform_train2 = transforms.Compose(augmentation_auto)
        transform_train3 = transforms.Compose(augmentation_randnclsstack)

        self.data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                   ])

        self.transforms1 = transform_train1
        self.transforms2 = transform_train2
        self.transforms3 = transform_train3


        self.data_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])


        #self.transforms = transform_train

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        ori_img = img.convert('RGB')
        out_img = self.data_transforms(ori_img)
        sample1 = self.transforms3(ori_img)
        sample2 = self.transforms3(ori_img)
        target = self.labels[idx]
        caption = self.captions_tokenized[idx]
        caption_ = self.captions_embedding[idx]
        return out_img, sample1, sample2, target, caption, caption_

class ImageList(object):

    def __init__(self, root, list_file, label_file, nb_classes, split):
        with open(list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # ['1','2'] -> [1,2]
        def list_str2int(x: list):
            return [int(i) for i in x]

        # image path & label
        self.fns = []
        self.labels = [] # one-hot e.g. [0,0,1,0,0,0,1...]
        for line in lines:
            self.fns.append(line.strip().split(' ')[0])
            label = list_str2int(line.strip().split(' ')[1:])
            # one-hot
            n_label = [[1 if i in label else 0 for i in range(nb_classes)]]
            y_all = np.ndarray([0, nb_classes])
            y_all = np.concatenate((y_all, np.array(n_label)))
            self.labels.append(y_all)
        self.labels = np.array(self.labels)
        self.fns = [os.path.join(root, fn) for fn in self.fns]

        # label name
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.categories = [{"id": int(line.strip().split(' ')[-1]), "name": line.strip().split(' ')[0]} for line in lines]

        self.split = split
        self.data_transforms = {
            'train': transforms.Compose([transforms.Resize((224,224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ]),
            'test': transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
            }

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        img = self.data_transforms[self.split](img)
        target = self.labels[idx]
        return img, target