from torch.cuda import device
import numpy as np
import cv2

def pred_yolo(img,model):
    pred_mask = np.zeros(img.shape[:2], np.uint8)

    res = model.predict(img,device=0,verbose=False)[0]

    if res.masks is None:
        return pred_mask

    for k in range(len(res.masks.xy)):
        xy = res.masks.xy[k].astype(np.int32)
        cls = int(res.boxes.cls[k].item()) + 1

        cv2.fillPoly(pred_mask, [xy], cls)


    return pred_mask



