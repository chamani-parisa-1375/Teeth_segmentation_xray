
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


def creat_n_colors(n=32):
    m = n + 1

    color_list = np.zeros((m, 1, 3), np.uint8)

    color_list[:, :, 1] = 255
    color_list[:, :, 2] = 255

    color_list[:, :, 0] = np.linspace(0, 180, m).reshape((-1, 1)).astype(np.uint8)
    color_list = cv2.cvtColor(color_list, cv2.COLOR_HSV2BGR)
    color_list = color_list.reshape((-1, 3))

    color_list = [(int(c[0]), int(c[1]), int(c[2])) for c in color_list]

    return color_list[:-1]


yolo_color_list = creat_n_colors(n=32)
def create_pred_img(img,model):

    res = model.predict(img,device=0,verbose=False)[0]

    overwrite = img.copy()

    if res.masks is None:
        return img

    for box,mask in zip(res.boxes,res.masks):
        xy = mask.xy[0].astype(np.int32)
        cls = int(box.cls[0].item())
        conf = round(box.conf[0].cpu().item(),2)

        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()


        cv2.fillPoly(overwrite, [xy], yolo_color_list[cls])
        cv2.drawContours(img, [xy], 0,  yolo_color_list[cls], 2)

        cv2.putText(img,f'{cls},{conf}',(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,1,yolo_color_list[cls],2)
        cv2.putText(overwrite, f'{cls},{conf}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, yolo_color_list[cls], 2)


    img = cv2.addWeighted(img,0.5,overwrite,0.5,0,img)

    return img

















