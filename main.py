from pathlib import Path
import os
import cv2
from utils import yolo_segmentation_to_unet_mask
from tqdm import tqdm

dataset1 = Path('data/Teeth_Segmentation_X-ray')
dataset2 = Path('data/Teeth_Segmentation_X-ray2')



for dataset_path in [dataset1,dataset2]:
    tbar = tqdm(dataset_path.iterdir())

    for folder in tbar:

        if folder.is_file():
            continue


        images_folder = folder / 'images'
        labels_folder = folder / 'labels'
        masks_folder = folder / 'masks'
        masks_folder.mkdir(parents=True, exist_ok=True)

        size = len(os.listdir(images_folder))


        for i,img_path in enumerate(images_folder.iterdir(),start=1):
            label_path = labels_folder / (img_path.stem + '.txt')
            mask_path = masks_folder / (img_path.stem + '.png')

            yolo_segmentation_to_unet_mask(str(img_path),str(label_path),str(mask_path))

            tbar.set_description(f'{dataset_path.stem},{folder.name}:[{i}/{size}]')








