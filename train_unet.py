import torch
from src.model_unet import load_model, estimate_best_batch_size
from src.engine_unet import *
from src.data_setup import get_train_loader,get_test_loader
from src.engine_unet import Callback
import pandas as pd
import os
import matplotlib.pyplot as plt



def create_save_dir(runs, result_path, resume, model_path):

    if resume:
        split = model_path.split('/')
        save_result = split[0] + '/' + split[1] + '/' + split[2]

    else:
        save_result = os.path.join(runs, 'unet_segment', result_path)

        temp = save_result
        i = 1

        while True:

            if not os.path.exists(temp):
                os.makedirs(temp, exist_ok=True)
                save_result = temp
                break

            i += 1
            temp = save_result + str(i)

        print(f'result_save_in: {save_result}')

    return save_result

def data_loader(dataset_path,batch_size,imgsize,nclasses):

    train_path = dataset_path + rf'/train'
    val_path = dataset_path + rf'/valid'
    test_path = dataset_path + rf'/test'
    test_flag = False
    train_dataloader = get_train_loader(train_path,
                                        batch_size=batch_size,
                                        nclass=nclasses,
                                        imgsize=imgsize)

    val_path = val_path if os.path.exists(val_path) else test_path
    val_dataloader = get_test_loader(val_path,
                                     batch_size=batch_size,
                                     nclass=nclasses,
                                     imgsize=imgsize)

    test_dataloader = None
    if os.path.exists(test_path) and not (val_path == test_path):
        test_dataloader = get_test_loader(test_path,
                                          batch_size=batch_size,
                                          nclass=nclasses,
                                          imgsize=imgsize)





    return train_dataloader,val_dataloader,test_dataloader




#### dataloader

def main(decoder_name='Unet',encoder_name='resnet34',num_epochs = 2,batch_size=-1,imgsize = (256, 256),
         patience = 10,lr=0.001,beta1=0.9 ,save_step=2,resume=False,model_path=''):
    runs = 'runs'
    dataset_path = rf'D:\MY_GIT\Teeth_segmentation_xray\data\Teeth_Segmentation_X-ray'
    result_path = f'{decoder_name},{encoder_name},{imgsize[0]}'
    save_result = create_save_dir(
        runs=runs,
        result_path=result_path,
        resume=resume,
        model_path=model_path
    )
    nclasses = 32



    call_back = Callback(patience=patience)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler('cuda')
    # device = 'cpu'v

    start_epoch = 1
    epoch_list = []
    train_losses_plot,train_accuracy_plot = [],[]
    valid_losses_plot,valid_accuracy_plot = [],[]

    unet_model = load_model(decoder_name,encoder_name,1,nclasses+ 1).to(device)
    if batch_size == -1:
        batch_size = estimate_best_batch_size(
            model=unet_model,
            image_size=imgsize,
            in_channels=1,
            num_classes=nclasses+ 1,
            target_gpu_usage=0.5,
            use_amp=True
        )

    train_dataloader, val_dataloader, test_dataloader = data_loader(dataset_path, batch_size, imgsize,nclasses)

    # unet_model = UNet(n_channel=1,n_class=1).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=lr)

    if resume:

        unet_model,optimizer,start_epoch = call_back.load_ckp(rf"runs\chest-segmentation\checkpoint.pt",
                                                              model=unet_model, optimizer=optimizer)
        start_epoch +=1
        print(f'resume training from epoch={start_epoch}')

        if os.path.exists(rf'{save_result}/result.csv'):

            df = pd.read_csv(rf'{save_result}/result.csv')
            epoch_list = list(df['epochs'])
            train_losses_plot = list(df['train_loss'])
            train_accuracy_plot = list(df['train_acc'])
            valid_losses_plot = list(df['valid_loss'])
            valid_accuracy_plot = list(df['valid_acc'])



    for epoch in range(start_epoch, num_epochs + 1):

        train_loss,train_acc = train_step(epoch=epoch,
                                          num_epochs=num_epochs,
                                          model=unet_model,
                                          train_dataloader=train_dataloader,
                                          optimizer=optimizer,
                                          loss_fn=criterion,
                                          scaler=scaler,
                                          device=device,
                                          save_result=save_result,
                                          save_step=save_step)

        train_losses_plot.append(train_loss)
        train_accuracy_plot.append(train_acc)

        best_dice = 0

        val_loss,val_acc = val_step(epoch=epoch,
                                    num_epochs=num_epochs,
                                    model=unet_model,
                                    val_dataloader=val_dataloader,
                                    loss_fn=criterion,
                                    device=device,
                                    save_result=save_result,
                                    save_step=save_step)

        valid_losses_plot.append(val_loss)
        valid_accuracy_plot.append(val_acc)

        state = {'epoch': epoch,
        'state_dict': unet_model.state_dict(),
        'optimizer': optimizer.state_dict()}

        call_back.save_ckp(state,val_loss,save_result)
        epoch_list.append(epoch)
        df = pd.DataFrame({"epochs":epoch_list, 'train_loss':train_losses_plot, 'train_acc':train_accuracy_plot, 'valid_loss':valid_losses_plot, 'valid_acc':valid_accuracy_plot})
        df.to_csv(rf'{save_result}/result.csv',index=False)


        if call_back.early_stop(val_loss):

            print(f'early stopping in [{epoch}\{num_epochs}]')
            break

    x = np.arange(len(valid_losses_plot))+1
    plt.plot(x,train_losses_plot)
    plt.plot(x,valid_losses_plot)
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend(('train_loss','val_loss'))
    plt.savefig(rf'{save_result}\loss.png')
    plt.close()

    plt.plot(x,train_accuracy_plot)
    plt.plot(x,valid_accuracy_plot)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(('train_accuracy','val_accuracy'))
    plt.savefig(rf'{save_result}\accuracy.png')
    plt.close()


    if not test_dataloader is None:
        test_loss, test_dice = test_step(

            model=unet_model,

            test_dataloader=test_dataloader,

            loss_fn=criterion,

            device=device,

            save_result=save_result
        )

        with open(
                rf'{save_result}\result_test.txt',
                'w'
        ) as f:
            f.write(
                f'test_loss:{test_loss}\n'
            )

            f.write(
                f'test_dice:{test_dice}\n'
            )


if __name__ == '__main__':


    # main('UnetPlusPlus','resnet34',imgsize=(512,512))
    main('UnetPlusPlus','efficientnet-b3',imgsize=(256,256))
    # main('UnetPlusPlus', 'efficientnet-b5', imgsize=(256, 256))

