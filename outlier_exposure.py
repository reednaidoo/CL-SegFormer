
'''
The script provided performs the augmentations necessary for creating the out of distribution images
The adaptations were inspired from the following repository:
https://github.com/yyliu01/RPL/blob/main/rpl.code/dataset/data_loader.py

'''



def move_images_based_on_dynamic_label(root, split, dynamic_folder, non_dynamic_folder):

    '''
    This function moves all of the dynamic and static images into their own respective folders   
    '''
    
    # Load the list of image file paths and their corresponding target labels
    images, targets, _ = load_cityscapes_dataset(root, split=split)

    for img_path, target_path in zip(images, targets):
        img = Image.open(img_path)

        # Extract the 'dynamic' label from the target image
        target = Image.open(target_path)
        dynamic_label = 5

        if (np.array(target) == dynamic_label).any():
            # Move images with 'dynamic' label to the dynamic_folder
            shutil.move(img_path, os.path.join(dynamic_folder, os.path.basename(img_path)))
            shutil.move(target_path, os.path.join("/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/Data/annotation/city_gt_fine/val_dynamic", os.path.basename(target_path)))
        else:
            # Move images without 'dynamic' label to the non_dynamic_folder
            shutil.move(img_path, os.path.join(non_dynamic_folder, os.path.basename(img_path)))
            shutil.move(target_path, os.path.join("/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/Data/annotation/city_gt_fine/val_undynamic", os.path.basename(target_path)))


# implementing the above function
root = "/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/Data"
split = 'val_copy'
dynamic_folder = "/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/Data/images/city_gt_fine/val_dynamic"
non_dynamic_folder = "/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/Data/images/city_gt_fine/val_undynamic"

move_images_based_on_dynamic_label(root, split, dynamic_folder, non_dynamic_folder)



import os
import random
import numpy as np
from PIL import Image

def extract_bboxes(mask):
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)  

def mix_object(current_labeled_image, current_labeled_mask, cut_object_image, cut_object_mask):
    print("Mixing dynamic object into undynamic image...")
    train_id_out = 5
    cut_object_mask = np.array(cut_object_mask)
    cut_object_mask[cut_object_mask == train_id_out] = 5

    mask = cut_object_mask == 5

    ood_mask = np.expand_dims(mask, axis=2)
    ood_boxes = extract_bboxes(ood_mask)
    ood_boxes = ood_boxes[0, :]
    y1, x1, y2, x2 = ood_boxes
    cut_object_mask = cut_object_mask[y1:y2, x1:x2]
    cut_object_image = np.array(cut_object_image)[y1:y2, x1:x2, :]
    idx = np.transpose(np.repeat(np.expand_dims(cut_object_mask, axis=0), 3, axis=0), (1, 2, 0))

    print("Bounding Box Coordinates:")
    print("y1:", y1)
    print("x1:", x1)
    print("y2:", y2)
    print("x2:", x2)

    print("Shape of cut_object_image:", cut_object_image.shape)

    h_start_point = random.randint(0, current_labeled_mask.size[1] - (y2 - y1))
    h_end_point = h_start_point + (y2 - y1)
    w_start_point = random.randint(0, current_labeled_mask.size[0] - (x2 - x1))
    w_end_point = w_start_point + (x2 - x1)

    current_labeled_image_array = np.array(current_labeled_image)
    current_labeled_mask_array = np.array(current_labeled_mask)

    current_labeled_image_array[h_start_point:h_end_point, w_start_point:w_end_point, :][np.where(idx == 5)] = \
        cut_object_image[np.where(idx == 5)]

    current_labeled_mask_array[h_start_point:h_end_point, w_start_point:w_end_point][np.where(cut_object_mask == 5)] = \
        cut_object_mask[np.where(cut_object_mask == 5)]
    
    # Create the new mask based on the desired labels
    new_mask = np.zeros_like(cut_object_mask)  # Initialize a new mask with zeros
    new_mask[cut_object_mask == 5] = 1  # Mark dynamic object as "out-distribution"
    new_mask[cut_object_mask != 5] = 0  # Mark other regions as "in-distribution"
    
    # Create a white outline for the dynamic object in the mask
    outline_mask = np.zeros_like(new_mask)
    outline_mask[1:, :] += new_mask[:-1, :]  # Top neighbor
    outline_mask[:-1, :] += new_mask[1:, :]  # Bottom neighbor
    outline_mask[:, 1:] += new_mask[:, :-1]  # Left neighbor
    outline_mask[:, :-1] += new_mask[:, 1:]  # Right neighbor
    outline_mask = np.logical_and(outline_mask, np.logical_not(new_mask))
    
    # Resize the outline mask and new mask to match the dimensions of the current labeled mask
    outline_mask_resized = np.array(Image.fromarray(outline_mask).resize((current_labeled_mask_array.shape[1], current_labeled_mask_array.shape[0])))
    new_mask_resized = np.array(Image.fromarray(new_mask).resize((current_labeled_mask_array.shape[1], current_labeled_mask_array.shape[0])))

    # Apply the white outline directly to the dynamic object region
    new_mask_with_outline = new_mask_resized + outline_mask_resized

    # Apply the new mask with outline to the final mask
    bw_mask = np.copy(current_labeled_mask_array)  # Make a copy of the current mask
    bw_mask[new_mask_with_outline == 1] = 255  # Mark dynamic object as white
    bw_mask[new_mask_with_outline == 2] = 255  # Mark dynamic object outline as white

    mixed_current_labeled_image = Image.fromarray(current_labeled_image_array)
    mixed_current_labeled_mask = Image.fromarray(bw_mask)

    return mixed_current_labeled_image, mixed_current_labeled_mask




def preprocess_and_save(city_img_paths, city_mask_paths, ood_img_paths, ood_mask_paths, output_dir, output_mask_dir, num_samples=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    dynamic_img_paths = os.listdir(ood_img_paths)
    undynamic_img_paths = os.listdir(city_img_paths) 

    for idx in range(num_samples):
        dynamic_img_path = os.path.join(ood_img_paths, random.choice(dynamic_img_paths)) 
        dynamic_mask_path = os.path.join(ood_mask_paths, os.path.basename(dynamic_img_path).replace("leftImg8bit", "gtFine_labelIds"))
        undynamic_img_path = os.path.join(city_img_paths, random.choice(undynamic_img_paths)) 
        undynamic_mask_path = os.path.join(city_mask_paths, os.path.basename(undynamic_img_path).replace("leftImg8bit", "gtFine_labelIds")) 
        dynamic_mask_path = os.path.join(ood_mask_paths, os.path.basename(dynamic_img_path).replace("leftImg8bit", "gtFine_labelIds"))

        dynamic_img = Image.open(dynamic_img_path)
        dynamic_mask = Image.open(dynamic_mask_path)
        undynamic_img = Image.open(undynamic_img_path)
        undynamic_mask = Image.open(undynamic_mask_path)

        print(f" {idx} Processing: \nDynamic image: '{dynamic_img_path}', \nUndynamic image '{undynamic_img_path}' \n--------------------------")

        mixed_undynamic_img, mixed_undynamic_mask = mix_object(undynamic_img.copy(), undynamic_mask.copy(), dynamic_img.copy(), dynamic_mask.copy())

        filename = f"preprocessed_{idx}_leftImg8bit.png"
        output_path = os.path.join(output_dir, filename)

        mixed_undynamic_img.save(output_path)

        # Save the new mask
        mask_filename = f"preprocessed_{idx}_gtFine_labelIds.png"
        output_mask_path = os.path.join(output_mask_dir, mask_filename)
        mixed_undynamic_mask.save(output_mask_path)

# Example usage:
city_img_paths = "/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/Data/images/city_gt_fine/val_undynamic"
city_mask_paths = "/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/Data/annotation/city_gt_fine/val_undynamic"

ood_img_paths = "/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/Data/images/city_gt_fine/val_dynamic"
ood_mask_paths = "/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/Data/annotation/city_gt_fine/val_dynamic"

output_dir = "/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/newdat"
output_mask_dir = "/Users/reednaidoo/Documents/My Mac Docs/Uni/2022 S2 - UoB/DISSERTATION/Coding/newdat_mask"

num_samples = 500  # number of images to generate for the augmentation set 

preprocess_and_save(city_img_paths, city_mask_paths, ood_img_paths, ood_mask_paths, output_dir, output_mask_dir, num_samples)





