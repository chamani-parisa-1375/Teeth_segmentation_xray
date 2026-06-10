from pathlib import Path

import cv2
import numpy as np
import torch

from src.engine_unet import Callback
from src.model_unet import load_model

models_folder = Path(rf'runs\unet_segment')
call_back = Callback()

def load_wight(model_path):

    decoder_name, encoder_name, imgsize = Path(model_path).stem.split(',')


    model = load_model(decoder_name, encoder_name, classes=33)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])




    return model,int(imgsize)





def pred_unet(img, model, imgsize):

    model.eval()
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    img = cv2.resize(img,(imgsize,imgsize))
    img = torch.from_numpy(img).float() /255
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)

    with torch.no_grad():



        pred = model(img)

        pred = torch.softmax(pred, dim=1)

        pred = torch.argmax(pred, dim=1)

        pred = pred.cpu().numpy()[0].astype(np.uint8)

        return pred









