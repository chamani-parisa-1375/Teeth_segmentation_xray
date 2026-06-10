
import os
import shutil

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# =====================================
# Visualization
# =====================================
class Callback:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def save_ckp(self, state, validation_loss, save_result):

        os.makedirs(save_result, exist_ok=True)
        os.makedirs(save_result, exist_ok=True)
        path = rf'{save_result}/checkpoint.pt'
        torch.save(state, path)

        if validation_loss < self.min_validation_loss:
            best = rf'{save_result}/best.pt'
            shutil.copyfile(path, best)

    def load_ckp(self, checkpoint_fpath, model, optimizer):
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']



# =====================================
# Visualization
# =====================================

def save_fig(
        input_img,
        target_img,
        output,
        save_result,
        mode,
        epoch=''):

    os.makedirs(save_result, exist_ok=True)

    input_img = input_img[0,0].detach().cpu().numpy()

    # ---------------------------------
    # target
    # one-hot => argmax
    # ---------------------------------

    if target_img.ndim == 4:

        target_img = torch.argmax(
            target_img,
            dim=1
        )

    target_img = target_img[0].detach().cpu().numpy()

    # ---------------------------------
    # prediction
    # logits => argmax
    # ---------------------------------

    pred_mask = torch.argmax(
        output,
        dim=1
    )

    pred_mask = pred_mask[0].detach().cpu().numpy()

    # ---------------------------------
    # plot
    # ---------------------------------

    plt.figure(figsize=(18,6))

    plt.subplot(1,3,1)
    plt.imshow(input_img, cmap='gray')
    plt.title('Input')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(target_img)
    plt.title('Target')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(pred_mask)
    plt.title('Prediction')
    plt.axis('off')

    plt.tight_layout()

    plt.savefig(
        f'{save_result}/{mode}_ep{epoch}.png'
    )

    plt.close()


# =====================================
# Dice Score
# =====================================



def dice_score(pred, target, smooth=1e-6):
    """
    pred   : (B, C, H, W)  logits
    target : (B, C, H, W)  one-hot
    """

    # تبدیل logits به کلاس
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)      # (B,H,W)
    target = torch.argmax(target, dim=1)  # (B,H,W)

    num_classes = pred.max().item() + 1
    dice_per_class = []

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice)

    dice_per_class = torch.stack(dice_per_class)

    return dice_per_class.mean()



# =====================================
# Train
# =====================================

def train_step(
        epoch,
        num_epochs,
        model,
        train_dataloader,
        optimizer,
        loss_fn,
        device,
        save_result,
        scaler=None,
        save_step=10):

    train_save_dir = os.path.join(
        save_result,
        'train'
    )

    os.makedirs(train_save_dir, exist_ok=True)

    model.train()

    train_losses = []
    train_dices = []

    train_desc = f'Train [{epoch}/{num_epochs}]'

    with tqdm(
        train_dataloader,
        desc=train_desc,
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
    ) as pbar:

        for i, (input_img, target_img) in enumerate(pbar):

            input_img = input_img.to(device)
            target_img = target_img.to(device)

            # ---------------------------------
            # onehot -> class index
            # CE Loss نیاز دارد
            # (B,C,H,W) -> (B,H,W)
            # ---------------------------------

            target_ce = torch.argmax(
                target_img,
                dim=1
            )

            optimizer.zero_grad()

            # =================================
            # Mixed Precision
            # =================================

            with torch.amp.autocast('cuda'):

                output = model(input_img)

                loss = loss_fn(
                    output,
                    target_ce
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            # ---------------------------------
            # metrics
            # ---------------------------------

            dice = dice_score(
                output.detach(),
                target_img
            )

            train_losses.append(loss.item())
            train_dices.append(dice.item())

            pbar.set_description(

                f'{train_desc} '
                f'loss:{np.mean(train_losses):.4f} '
                f'dice:{np.mean(train_dices):.4f}'

            )

            # ---------------------------------
            # save image
            # ---------------------------------

            if (
                i == len(train_dataloader)-1
                and epoch % save_step == 0
            ):

                save_fig(
                    input_img[:1],
                    target_img[:1],
                    output[:1],
                    train_save_dir,
                    'train',
                    epoch
                )

    return (
        np.mean(train_losses),
        np.mean(train_dices)
    )


# =====================================
# Validation
# =====================================

def val_step(
        epoch,
        num_epochs,
        model,
        val_dataloader,
        loss_fn,
        device,
        save_result,

        save_step=10):

    val_save_dir = os.path.join(
        save_result,
        'val'
    )

    os.makedirs(val_save_dir, exist_ok=True)

    model.eval()

    val_losses = []
    val_dices = []

    with torch.inference_mode():

        val_desc = f'Val [{epoch}/{num_epochs}]'

        with tqdm(
            val_dataloader,
            desc=val_desc,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        ) as pbar:

            for i, (input_img, target_img) in enumerate(pbar):

                input_img = input_img.to(device)
                target_img = target_img.to(device)

                target_ce = torch.argmax(
                    target_img,
                    dim=1
                )

                output = model(input_img)

                loss = loss_fn(
                    output,
                    target_ce
                )

                dice = dice_score(
                    output,
                    target_img
                )

                val_losses.append(loss.item())
                val_dices.append(dice.item())

                pbar.set_description(

                    f'{val_desc} '
                    f'loss:{np.mean(val_losses):.4f} '
                    f'dice:{np.mean(val_dices):.4f}'

                )

                # -----------------------------
                # save sample
                # -----------------------------

                if (
                    i == len(val_dataloader)-1
                    and epoch % save_step == 0
                ):

                    save_fig(
                        input_img[:1],
                        target_img[:1],
                        output[:1],
                        val_save_dir,
                        'val',
                        epoch
                    )

    mean_loss = np.mean(val_losses)
    mean_dice = np.mean(val_dices)


    return (
        mean_loss,
        mean_dice,

    )


def test_step(
        model,
        test_dataloader,
        loss_fn,
        device,
        save_result):

    test_save_dir = os.path.join(
        save_result,
        'test'
    )

    os.makedirs(test_save_dir, exist_ok=True)

    model.eval()

    test_losses = []
    test_dices = []

    with torch.inference_mode():

        test_desc = 'Test'

        with tqdm(
            test_dataloader,
            desc=test_desc,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        ) as pbar:

            for i, (input_img, target_img) in enumerate(pbar):

                # ---------------------------------
                # device
                # ---------------------------------

                input_img = input_img.to(device)

                target_img = target_img.to(device)

                # ---------------------------------
                # one-hot -> class index
                # CE Loss نیاز دارد
                # ---------------------------------

                target_ce = torch.argmax(
                    target_img,
                    dim=1
                )

                # ---------------------------------
                # prediction
                # ---------------------------------

                output = model(input_img)

                # ---------------------------------
                # loss
                # ---------------------------------

                loss = loss_fn(
                    output,
                    target_ce
                )

                # ---------------------------------
                # dice
                # ---------------------------------

                dice = dice_score(
                    output,
                    target_img
                )

                # ---------------------------------
                # save metrics
                # ---------------------------------

                test_losses.append(loss.item())

                test_dices.append(dice.item())

                # ---------------------------------
                # tqdm
                # ---------------------------------

                pbar.set_description(

                    f'{test_desc} '
                    f'loss:{np.mean(test_losses):.4f} '
                    f'dice:{np.mean(test_dices):.4f}'

                )

                # ---------------------------------
                # save sample result
                # ---------------------------------

                if i == len(test_dataloader)-1:

                    save_fig(
                        input_img[:1],
                        target_img[:1],
                        output[:1],
                        test_save_dir,
                        'test'
                    )

    mean_loss = np.mean(test_losses)

    mean_dice = np.mean(test_dices)

    print(
        f'\nTest Loss:{mean_loss:.4f} '
        f'Test Dice:{mean_dice:.4f}'
    )

    return (
        mean_loss,
        mean_dice
    )



