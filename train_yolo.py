from ultralytics import YOLO
from pathlib import Path


def train_yolo(yaml, model_path,imgsize ,batch_size,epochs=300,patience=20):

    model = YOLO(model_path)
    save_name = f'{model_path.stem},{imgsize}'

    model.train(data=yaml,epochs=epochs,batch=batch_size,patience=patience,imgsz=imgsize,name=save_name + '-train')





if __name__ == '__main__':

    data_yaml = Path('data/Teeth_Segmentation_X-ray/Teeth_Segmentation_X-ray.yaml')

    model_n = Path('models/yolo11n-seg.pt')
    model_m = Path('models/yolo11m-seg.pt')


    imgsize_960 = 960
    imgsize_640 = 640


    train_yolo(data_yaml,model_n,300,batch_size=4)





















