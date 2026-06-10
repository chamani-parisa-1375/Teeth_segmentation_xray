import cv2
from ultralytics import YOLO
from pathlib import Path

from src.utils import dice_iou_multiclass
from src.pred_unet import load_wight,pred_unet
from tqdm import tqdm
import json
from src.yolo_pred import pred_yolo

data_path = Path('data/predict')
unet_model_path =  Path("models/unet_model")
yolo_model_path = Path(rf"models/yolo_model")

list_images_path = list((data_path / 'images').glob('*.jpg'))
list_masks_path = [data_path / 'masks' / (i.stem + '.png') for i in list_images_path]


list_yolo_models = [ (fpt.stem,YOLO(fpt,task='segment'),int(fpt.stem.split(',')[-1]))
                     for fpt in tqdm(yolo_model_path.iterdir(),desc="loading yolo model")]
list_unet_models = [(fpt.stem,)+(load_wight(fpt)) for fpt in tqdm(unet_model_path.iterdir(),desc="loading Unet model")]

results = {}

tbar = tqdm(enumerate(zip(list_images_path,list_masks_path),start=1))
for i,(img_path,mask_path) in tbar:

    results[img_path.stem] = {}

    img = cv2.imread(str(img_path))
    org_mask = cv2.imread(str(mask_path))
    org_mask = cv2.cvtColor(org_mask, cv2.COLOR_BGR2GRAY)


    for model_name,model,imgsize in (list_unet_models + list_yolo_models):

        tbar.set_description(f"predicting {img_path.stem},{model_name}")

        model_folder = data_path / model_name
        model_folder.mkdir(parents=True, exist_ok=True)

        pred_path = model_folder / mask_path.name
        # if pred_path.exists():
        #     continue

        pred_mask = pred_unet(img,model,imgsize) if model_name.startswith('Unet') else pred_yolo(img,model)

        true_mask = cv2.resize(org_mask,(imgsize,imgsize),cv2.INTER_NEAREST) \
            if model_name.startswith('Unet') else org_mask.copy()

        results[img_path.stem][model_name] = dice_iou_multiclass(true_mask, pred_mask, 33)



        # cv2.imwrite(str(pred_path),pred_mask)

    if i%20 == 0:
        tbar.set_description('saving result')
        with open(data_path/'result.json','w') as f:

            json.dump(results,f,indent=4)

















