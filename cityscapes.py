from collections import namedtuple
import os
from typing import List, Optional, Callable, Tuple, Any
from PIL import Image
import pandas as pd


'''
Much of this code was taken from  the authors of:
https://github.com/tianyu0207/PEBAL/blob/main/docs/installation.md

Slight adjustments were made surrounding the update in data structure. Data is to be downloaded from:
https://www.cityscapes-dataset.com
with specific reference to: "leftImg8bit_trainvaltest.zip (11GB) [md5]" and "gtCoarse.zip (1.3GB) [md5]" and "gtFine_trainvaltest.zip (241MB) [md5]" 

Please see the data structure in the ReadMe file to understand what your root directory should look like
'''


def load_cityscapes_dataset(root: str = "/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/Data", split: str = "train",
                            mode: str = "gtFine", target_type: str = "semantic_train_id", transform: Optional[Callable] = None,
                            predictions_root: Optional[str] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Load Cityscapes dataset.

    Args:
        root (str): Root directory of the Cityscapes dataset.
        split (str): Split of the dataset (train, val, or test).
        mode (str): Mode of the dataset (gtFine or gtCoarse).
        target_type (str): Type of the target (instance, semantic_id, semantic_train_id, or color).
        transform (Optional[Callable]): Optional data transformation to be applied to the images and targets.
        predictions_root (Optional[str]): Root directory of the predictions (optional, used for evaluation).

    Returns:
        Tuple[List[str], List[str], List[str]]: A tuple containing three lists:
            - List of image file paths.
            - List of target file paths.
            - List of prediction file paths (if provided).
    """

    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
    images_dir = os.path.join(root, 'images/city_gt_fine', split)
    targets_dir = os.path.join(root, 'annotation/city_gt_fine', split)
    predictions_dir = os.path.join(predictions_root, split) if predictions_root is not None else ""
    images = []
    targets = []
    predictions = []

    for city in os.listdir(images_dir):
        img_path = os.path.join(images_dir, city)
        target_dir = os.path.join(targets_dir)
        pred_dir = os.path.join(predictions_dir)

        # Append the individual image file path directly to the images list
        images.append(img_path)

        # For target and prediction file paths, we need to construct the file names based on the image file name
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        #target_name = '{}_{}'.format(file_name.replace("_leftImg8bit", ""), _get_target_suffix(mode, 'semantic_id'))
        target_name = file_name
        prediction_name = file_name.replace("_leftImg8bit", "")

        # Append the target and prediction file paths to their respective lists
        targets.append(os.path.join(target_dir, target_name))
        predictions.append(os.path.join(pred_dir, prediction_name))

    return images, targets, predictions

def get_useful_label_info():
    
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)
    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}

    return ignore_in_eval_ids, label_ids, train_ids, train_id2id, color_palette_train_ids, num_label_ids, num_train_ids, id2label, train_id2label


def _get_target_suffix(mode: str, target_type: str) -> str:
    if target_type == 'instance':
        return '{}_instanceIds.png'.format(mode)
    elif target_type == 'semantic_id':
        return '{}_labelIds.png'.format(mode)


def get_encoded_inputs(img_dir, ann_dir):
    image = Image.open(img_dir)
    annotation = Image.open(ann_dir)

    annotation = np.array(annotation)
    annotation_2d = np.zeros((annotation.shape[0], annotation.shape[1]), dtype = np.uint8)

    for id, color in id2color.items():
        annotation_2d[(annotation == color).all(axis=1)] =id

    encoded_inputs = feature_extractor(image, Image.fromarray(annotation_2d), return_tensors='pt')

    for k,v in encoded_inputs.items():
        encoded_inputs[k].squeeze()

    return encoded_inputs


data = []

for label in labels:
    label_info = {
        'label_idx': label.id,
        'label': label.name,
        'R': label.color[0],
        'G': label.color[1],
        'B': label.color[2]
    }
    data.append(label_info)

# Create a pandas DataFrame
colour_df = pd.DataFrame(data)