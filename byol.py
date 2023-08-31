from torchvision import datasets, transforms
from cityscapes import load_cityscapes_dataset
from moco_loader import TwoCropsTransform, GaussianBlur
from cityscapes import *

import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset


'''
This piece of code was an adaptation taken from:
https://github.com/yuanqijue/cvt
'''


def build_transform(is_train, args, aug=True):
    if args.data_set == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif args.data_set == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    elif args.data_set == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    elif args.data_set == "cityscapes":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if not aug:
        return transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor(), normalize, ])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [transforms.RandomResizedCrop(args.image_size, scale=(args.crop_min, 1.)),
                     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                                             ], p=0.8), transforms.RandomGrayscale(p=0.2),
                     transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
                     transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]

    augmentation2 = [transforms.RandomResizedCrop(args.image_size, scale=(args.crop_min, 1.)),
                     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                                             ], p=0.8), transforms.RandomGrayscale(p=0.2),
                     transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
                     transforms.RandomApply([Solarize()], p=0.2), transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), normalize]

    transform_train = TwoCropsTransform(transforms.Compose(augmentation1),
                                                    transforms.Compose(augmentation2))
    
    return transform_train


def build_dataset(is_train, args, aug=True):
    transform = build_transform(is_train, args, aug)

    if args.data_set == 'imagenet':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'cifar10':
        dataset = datasets.CIFAR10(root=args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.data_set == "cifar100":
        dataset = datasets.CIFAR100(root=args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif args.data_set == "svhn":
        dataset = datasets.SVHN(root=args.data_path, split='train' if is_train else 'test', download=True,
                                transform=transform)
        nb_classes = 10
    elif args.data_set == "lsun":
        dataset = datasets.LSUN(root=args.data_path, classes='train' if is_train else 'test', transform=transform)
        nb_classes = 20
    elif args.data_set == "place365":
        dataset = datasets.Places365(root=args.data_path, split='train-standard' if is_train else 'val', download=True,
                                     small=True, transform=transform)
        nb_classes = 365

    elif args.data_set == "cityscapes":
        dataset = load_cityscapes_dataset(root=args.data_path, split='train1' if is_train else 'val1', transform=transform)
        nb_classes = len(cityscapes.train_ids)
        #nb_classes = 21
    else:
        raise NotImplementedError("Only [imagenet, cifar10, cifar100, svhn, lsun, place365 ] are supported")

    return dataset, nb_classes






class SemanticSegmentationDataset():
    """Function designed to create the query and key embeddings for contrastive learning"""

    def __init__(self, root_dir, feature_extractor, transforms=None, train=True, image_size=512, crop_min=0.08):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train=train
        self.transforms = transforms
        self.image_size = image_size
        self.crop_min = crop_min

        if train==True:
            self.img_dir = os.path.join(self.root_dir, "images/city_gt_fine/train1")
            self.ann_dir = os.path.join(self.root_dir, "annotation/city_gt_fine/train1")
        else:
            self.img_dir = os.path.join(self.root_dir, "images/city_gt_fine/val1")
            self.ann_dir = os.path.join(self.root_dir, "annotation/city_gt_fine/val1")
        
        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)
        
        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            filtered_files = [file for file in files if file.endswith("_gtFine_labelTrainIds.png")]
            annotation_file_names.extend(filtered_files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        segmentation_map = cv2.imread(os.path.join(self.ann_dir, self.annotations[idx]))
        segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=segmentation_map)
            encoded_inputs = self.feature_extractor(augmented['image'], augmented['mask'], return_tensors="pt")

        else:
            encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_() 

        return {
            "encoded_inputs": encoded_inputs
        }
    


root_dir = '/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/Data'

feature_extractor = SegformerFeatureExtractor(align=False, reduce_labels=True)
transform = aug.Compose([aug.Flip(p=0.5)])


# note, the identical procedure should be performed to obtain the test and validation dataloaders. That requires changing the root directories and pointing to the appropriate test masks 
train_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor, transforms=transform)
valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor, transforms=None, train=False)
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=10)

batch = next(iter(train_dataloader))

classes = colour_df['label']
id2label = classes.to_dict()
label2id = {v: k for k, v in id2label.items()}