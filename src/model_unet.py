from segmentation_models_pytorch import Unet,UnetPlusPlus,MAnet
import torch
import gc
from contextlib import suppress

dic_model = {'Unet':Unet,'UnetPlusPlus':UnetPlusPlus,'MAnet':MAnet}
dic_encoder = {'efficientnet-b3':'models/checkpoints/efficientnet-b3-5fb5a3c3.pth',
               "efficientnet-b5":'models/checkpoints/efficientnet-b5-b6417697.pth',
               "resnet34":"models/checkpoints/resnet34-333f7ec4.pth"}

def load_model(decoder_name,encoder_name,in_channels=1,classes=32):


    model = dic_model[decoder_name](
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes
    )


    return model







def get_gpu_memory_info(device=0):

    total_memory = torch.cuda.get_device_properties(device).total_memory

    reserved_memory = torch.cuda.memory_reserved(device)

    allocated_memory = torch.cuda.memory_allocated(device)

    free_memory = total_memory - max(reserved_memory, allocated_memory)

    return {
        "total_gb": total_memory / 1024**3,
        "free_gb": free_memory / 1024**3,
        "reserved_gb": reserved_memory / 1024**3,
        "allocated_gb": allocated_memory / 1024**3
    }


def clear_gpu():

    gc.collect()

    torch.cuda.empty_cache()

    with suppress(Exception):
        torch.cuda.ipc_collect()


def estimate_best_batch_size(
        model,
        image_size=(256, 256),
        in_channels=3,
        num_classes=1,
        device='cuda',
        target_gpu_usage=0.5,
        start_batch_size=1,
        max_batch_size=2048,
        use_amp=True,
        optimizer_class=torch.optim.Adam,
        criterion=None,
):

    """
    پیدا کردن بهترین batch size بر اساس درصد استفاده از GPU

    Parameters
    ----------
    model : torch.nn.Module
    image_size : tuple
        (H,W)
    in_channels : int
    num_classes : int
    target_gpu_usage : float
        مثلا 0.5 یعنی 50 درصد حافظه GPU
    use_amp : bool
    """

    assert torch.cuda.is_available(), "CUDA not available"

    device = torch.device(device)

    model = model.to(device)

    model.train()

    if criterion is None:

        if num_classes == 1:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

    optimizer = optimizer_class(model.parameters())

    torch.cuda.reset_peak_memory_stats(device)

    gpu_info = get_gpu_memory_info(device)

    total_memory = (
        torch.cuda.get_device_properties(device).total_memory
    )

    target_memory = total_memory * target_gpu_usage

    print("=" * 60)

    print(f"GPU Total Memory : {gpu_info['total_gb']:.2f} GB")

    print(f"Target Usage     : {target_gpu_usage*100:.0f}%")

    print(
        f"Target Memory    : "
        f"{target_memory / 1024**3:.2f} GB"
    )

    print("=" * 60)

    batch_size = start_batch_size

    best_batch_size = batch_size

    while batch_size <= max_batch_size:

        try:

            clear_gpu()

            # ساخت input fake
            x = torch.randn(
                batch_size,
                in_channels,
                image_size[0],
                image_size[1],
                device=device
            )

            if num_classes == 1:

                y = torch.randn(
                    batch_size,
                    1,
                    image_size[0],
                    image_size[1],
                    device=device
                )

            else:

                y = torch.randint(
                    0,
                    num_classes,
                    (
                        batch_size,
                        image_size[0],
                        image_size[1]
                    ),
                    device=device
                )

            optimizer.zero_grad(set_to_none=True)

            torch.cuda.reset_peak_memory_stats(device)

            if use_amp:

                scaler = torch.cuda.amp.GradScaler()

                with torch.amp.autocast('cuda'):

                    output = model(x)

                    loss = criterion(output, y)

                scaler.scale(loss).backward()

                scaler.step(optimizer)

                scaler.update()

            else:

                output = model(x)

                loss = criterion(output, y)

                loss.backward()

                optimizer.step()

            peak_memory = torch.cuda.max_memory_allocated(device)

            used_gb = peak_memory / 1024**3

            used_percent = peak_memory / total_memory

            print(
                f"Batch Size: {batch_size:<5} | "
                f"GPU Usage: {used_gb:.2f} GB "
                f"({used_percent*100:.1f}%)"
            )

            if peak_memory <= target_memory:

                best_batch_size = batch_size

                batch_size *= 2

            else:

                break

        except RuntimeError as e:

            if "out of memory" in str(e).lower():

                print(
                    f"OOM at batch size = {batch_size}"
                )

                clear_gpu()

                break

            else:
                raise e

    print("\n" + "=" * 60)

    print(
        f"Recommended Batch Size = {best_batch_size}"
    )

    print("=" * 60)

    clear_gpu()

    return best_batch_size