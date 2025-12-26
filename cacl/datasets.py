import torch
from torch.utils.data import Dataset
from imagelist import *

class CustomDatasetCaption(Dataset):
    """Dataset.
    """

    def __init__(self, dataset, split, clip_model):
        assert dataset in ["coco-lt", "voc-lt", "coco", "voc"]
        if dataset == 'coco-lt':
            self.data_source = ImageListCaption(root='/media/kin/新加卷/coco/',
                                        list_file='/home/kin/桌面/LMPT-main/data/coco/coco_lt_%s.txt' % split,
                                        caption_file='/home/kin/桌面/LMPT-main/data/coco/coco_lt_captions.txt',
                                        label_file='/home/kin/桌面/LMPT-main/data/coco/coco_labels.txt',
                                        nb_classes=80,
                                        split=split,
                                        clip_model=clip_model)
        elif dataset == 'voc-lt':
            self.data_source = ImageListCaption(root='/media/kin/新加卷/voc/VOCtrainval_11-May-2012/',
                                        list_file='../data/voc/voc_lt_%s.txt' % split,
                                        caption_file='../data/voc/voc_lt_captions.txt',
                                        label_file='../data/voc/voc_labels.txt',
                                        nb_classes=20,
                                        split=split,
                                        clip_model=clip_model)

        self.targets = self.data_source.labels # one-hot label
        self.captions = self.data_source.captions_tokenized
        self.captions_ = self.data_source.captions_embedding
        self.categories = self.data_source.categories
        self.fns = self.data_source.fns

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        out_img, sample1, sample2, target, caption, caption_ = self.data_source.get_sample(idx)
        return out_img, sample1, sample2, target, caption, caption_

    def get_cls_num_list(self):
        labels = np.array(self.targets)  # 将标签列表转换为 NumPy 数组
        cls_num_list = labels.sum(axis=0).tolist()  # 按列求和，得到每个类别的样本数
        return cls_num_list

class CustomDataset(Dataset):
    """Dataset.
    """

    def __init__(self, dataset, split):
        assert dataset in ["coco-lt", "voc-lt", "voc", "nus-wide"]
        if dataset == 'coco-lt':
            self.data_source = ImageList(root='/media/kin/新加卷/coco/',
                                        list_file='../data/coco/coco_lt_%s.txt' % split,
                                        label_file='../data/coco/coco_labels.txt',
                                        nb_classes=80,
                                        split=split)
        elif dataset == 'voc-lt':
            self.data_source = ImageList(root='/media/kin/新加卷/voc/VOCtrainval_11-May-2012/',
                                        list_file='../data/voc/voc_lt_%s.txt' % split,
                                        label_file='../data/voc/voc_labels.txt',
                                        nb_classes=20,
                                        split=split)

        self.targets = self.data_source.labels # one-hot label
        self.categories = self.data_source.categories
        self.fns = self.data_source.fns

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        return img, target

def build_dataset_with_caption(dataset, split, clip_model=None):
    assert split in ['train', 'test', 'val']

    assert dataset in ["coco-lt", "voc-lt"]
    if split == 'train':
        dataset = CustomDatasetCaption(
        dataset=dataset, 
        split=split,
        clip_model=clip_model
        )
    elif split == 'test':
        dataset = CustomDataset(
        dataset=dataset, 
        split=split
        )

    return dataset

def build_dataset(dataset, split):
    assert split in ['train', 'test', 'val']

    assert dataset in ["coco-lt", "voc-lt"]
    if split == 'train':
        dataset = CustomDataset(
        dataset=dataset, 
        split=split
        )
    elif split == 'test':
        dataset = CustomDataset(
        dataset=dataset, 
        split=split
        )

    return dataset
