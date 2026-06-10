

import os
import cv2
import torch
import random
import numpy as np
from PIL import Image

# =========================
# Dataset
# =========================

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 nclass,
                 transform=None):

        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.masks_dir = os.path.join(data_dir, 'masks')

        self.images = sorted(os.listdir(self.images_dir))
        self.masks = sorted(os.listdir(self.masks_dir))

        self.transform = transform
        self.num_classes = nclass

    def __len__(self):
        return len(self.images)

    def mask_to_onehot(self, mask):

        """
        mask shape : (H,W)
        output     : (C,H,W)
        """

        onehot = np.zeros(
            (self.num_classes + 1, mask.shape[0], mask.shape[1]),
            dtype=np.float32
        )

        for c in range(self.num_classes):
            onehot[c] = (mask == c ).astype(np.float32)

        return onehot

    def __getitem__(self, index):

        image_path = os.path.join(
            self.images_dir,
            self.images[index]
        )

        mask_path = os.path.join(
            self.masks_dir,
            self.masks[index]
        )

        # -------------------------
        # read image
        # -------------------------

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # image shape => (H,W)

        # -------------------------
        # read mask
        # -------------------------

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # mask values:
        # 0,1,2,3,...,31

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # -------------------------
        # normalize image
        # -------------------------

        image = image.astype(np.float32) / 255.0

        # (H,W) -> (1,H,W)
        image = np.expand_dims(image, axis=0)

        # -------------------------
        # one hot mask
        # -------------------------

        mask = self.mask_to_onehot(mask)

        # -------------------------
        # to tensor
        # -------------------------

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask


# =========================
# Resize
# =========================

class Resize:

    def __init__(self, size=(256,256)):
        self.size = size

    def __call__(self, image, mask):

        image = cv2.resize(
            image,
            self.size,
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            self.size,
            interpolation=cv2.INTER_NEAREST
        )

        return image, mask


# =========================
# Horizontal Flip
# =========================

class HorizontalFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):

        if random.random() < self.p:

            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        return image, mask


# =========================
# Random Rotation
# =========================

class RandomRotate:

    def __init__(self,
                 angle=10,
                 p=0.5):

        self.angle = angle
        self.p = p

    def __call__(self, image, mask):

        if random.random() < self.p:

            h, w = image.shape

            angle = random.uniform(
                -self.angle,
                self.angle
            )

            M = cv2.getRotationMatrix2D(
                (w // 2, h // 2),
                angle,
                1.0
            )

            image = cv2.warpAffine(
                image,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            mask = cv2.warpAffine(
                mask,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT_101
            )

        return image, mask


# =========================
# Random Brightness
# =========================

class RandomBrightness:

    def __init__(self,
                 limit=0.15,
                 p=0.5):

        self.limit = limit
        self.p = p

    def __call__(self, image, mask):

        if random.random() < self.p:

            alpha = 1.0 + random.uniform(
                -self.limit,
                self.limit
            )

            image = image.astype(np.float32)
            image = image * alpha

            image = np.clip(image, 0, 255)

            image = image.astype(np.uint8)

        return image, mask


# =========================
# Compose
# =========================

class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):

        for t in self.transforms:
            image, mask = t(image, mask)

        return image, mask


# =========================
# Dataloader
# =========================

def get_train_loader(
        data_path,
        imgsize=(256,256),
        batch_size=8,
        nclass=32,
        num_workers=4):

    train_transform = Compose([

        Resize(imgsize),

        HorizontalFlip(p=0.5),

        RandomRotate(
            angle=8,
            p=0.5
        ),

        RandomBrightness(
            limit=0.10,
            p=0.3
        ),

    ])

    train_dataset = ImageDataset(
        data_dir=data_path,
        nclass=nclass,
        transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(

        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True

    )

    return train_loader





def get_test_loader(
        data_path,
        imgsize=(256,256),
        batch_size=8,
        nclass=32,
        num_workers=4):

    test_transform = Compose([

        Resize(imgsize),


    ])

    test_dataset = ImageDataset(
        data_dir=data_path,
        nclass=nclass,
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(

        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True

    )

    return test_loader


    
