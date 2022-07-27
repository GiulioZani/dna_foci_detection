from torch.utils import data
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tifffile import TiffWriter
from tifffile import imread, imsave
import random
import torch as t

from dataset import Dataset
from config import Config
from utils.utils import dice_loss

# from unet3d import Unet3d
from unet2d import UNet
from utils.log import Log
from utils.predict_by_parts import predict_by_parts
import h5py
from tqdm import tqdm
from utils.utils import minmax
import ipdb


def main():
    device = torch.device("cuda:0")
    config = Config()

    try:
        os.mkdir(Config.tmp_save_dir)
    except:
        pass

    with h5py.File(config.hdf5_filename, "r") as h5data:
        filenames = np.array(
            [file[:-4] for file in list(h5data.keys()) if "data" in file]
        )
        permuted_indices = t.randperm(len(filenames))
    valid_size = int(len(filenames) * 0.2)
    valid_filenames = filenames[permuted_indices[:valid_size]]
    train_filenames = filenames[permuted_indices[2 * valid_size :]]
    os.system("rm -rf ./imgs/*")
    loader = Dataset(
        hdf5_filename=config.hdf5_filename,
        filenames=train_filenames,
        split="train",
        crop_size=config.crop_size,
    )
    trainloader = data.DataLoader(
        loader,
        batch_size=config.train_batch_size,
        shuffle=True,
        drop_last=True,
    )

    loader = Dataset(
        hdf5_filename=config.hdf5_filename,
        filenames=valid_filenames,
        split="valid",
        crop_size=config.crop_size,
    )
    valid_loader = data.DataLoader(
        loader,
        batch_size=config.test_batch_size,
        # num_workers=config.test_num_workers,
        shuffle=False,
        drop_last=False,
    )
    loader = Dataset(
        hdf5_filename=config.hdf5_filename,
        filenames=train_filenames,
        split="train",
        crop_size=config.crop_size,
    )
    trainloader = data.DataLoader(
        loader,
        batch_size=config.train_batch_size,
        # num_workers=config.train_num_workers,
        shuffle=True,
        drop_last=True,
    )
    model = UNet(
        n_channels=3,
        n_classes=1
        # filters=config.filters, in_size=config.input_size, out_size=config.output_size
    )
    # model.test_filenames = test_filenames
    # model.valid_filenames = valid_filenames
    # model.train_filenames = train_filenames

    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.init_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-8,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, config.lr_changes_list, gamma=config.gamma, last_epoch=-1
    )

    model.log = Log(names=["loss"])

    for epoch_num in range(Config.max_epochs):
        print(f"Epoch: {epoch_num}")
        model.train()
        N = len(trainloader)
        for it, (batch, lbls, _) in enumerate(tqdm(trainloader)):

            batch = batch.to(device)
            lbls = lbls.to(device)

            res = model(batch).squeeze(1)
            # ipdb.set_trace()
            # res = torch.sigmoid(res)
            # loss = dice_loss(res,lbls)

            # loss = torch.mean((res - lbls) ** 2)
            loss = F.mse_loss(res, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            # res = res.detach().cpu().numpy()
            # lbls = lbls.detach().cpu().numpy()

            model.log.append_train([loss])

        with torch.no_grad():
            model.eval()
            N = len(valid_loader)
            plt.title(f"Epoch: {epoch_num}")
            plt.clf()
            plt.close()
            rows = 6
            fig, axes = plt.subplots(6, 3)
            for it, (batch, lbls, _) in enumerate(tqdm(valid_loader)):

                batch = batch.to(device)
                lbls = lbls.to(device)

                res = model(batch).squeeze(1)

                # res = torch.sigmoid(res)
                # loss = dice_loss(res,lbls)

                loss = F.mse_loss(res, lbls)

                loss = loss.cpu().numpy()
                # res = res.detach().cpu().numpy()
                # lbls = lbls.detach().cpu().numpy()

                model.log.append_valid([loss])

                if it < 6:
                    axes[it][0].imshow(minmax(batch[0].cpu().permute(1, 2, 0)))
                    axes[it][1].imshow(lbls[0].cpu())
                    axes[it][2].imshow(res[0].cpu())

            plt.savefig(f"./imgs/epoch_{epoch_num}.png", dpi=300)

        model.log.save_and_reset()

        lr = optimizer.param_groups[0]["lr"]
        info = f"{epoch_num} {lr:.5f} train {model.log.train_log['loss'][-1]:.5f} valid {model.log.valid_log['loss'][-1]:.5f}"

        print(info)

        model_name = (
            config.tmp_save_dir + os.sep + config.model_name + info + ".pt"
        )

        model.log.save_log_model_name('detection_model.pt')

        torch.save(model.state_dict(), 'detection_model.pt')
        print(f"Saved model to detection_model.pt")
        # model.log.plot(model_name.replace(".pt", "loss.png"))

        scheduler.step()

    with torch.no_grad():
        model.eval()
        N = len(valid_loader)
        for it, (batch, lbls, _) in enumerate(tqdm(valid_loader)):

            batch = batch.to(device)
            lbls = lbls.to(device)

            res = model(batch).squeeze(1)

            # res = torch.sigmoid(res)
            # loss = dice_loss(res,lbls)

            loss = F.mse_loss(res, lbls)

            loss = loss.cpu().numpy()
            # res = res.detach().cpu().numpy()
            # lbls = lbls.detach().cpu().numpy()

            model.log.append_valid([loss])


if __name__ == "__main__":
    main()
    # model = torch.load('../../data_zenodo/tmp_segmentation_model\\segmentation_model_1_164_0.00000_train_0.43957_valid_0.99332.pt')

    # model = model.to(device)

    # loader = Dataset(hdf5_filename=config.hdf5_filename,split='test',crop_size=config.crop_size)
    # testLoader= data.DataLoader(loader, batch_size=1, num_workers=0, shuffle=False,drop_last=False)
    # with torch.no_grad():
    #     model.eval()
    #     for it, (batch,lbls,filenames) in enumerate(testLoader):

    #         batch=batch.to(device)
    #         lbls=lbls.to(device)

    #         res = predict_by_parts(model,batch[0,:,:,:], crop_size=config.crop_size)

    #         # res = torch.sigmoid(res)

    #         batch = batch.detach().cpu().numpy()
    #         res = res.detach().cpu().numpy()
    #         lbls = lbls.detach().cpu().numpy()

    #         plt.imshow(np.max(batch[0,0,:,:,:],axis=2))
    #         plt.show()
    #         plt.imshow(np.max(lbls[0,0,:,:,:],axis=2))
    #         plt.show()
    #         plt.imshow(np.max(res[0,:,:,:],axis=2))
    #         plt.show()
    #         print()

    #         filename_saveimg = config.tmp_save_dir + os.sep + 'result_' + filenames[0]  + '.tiff'

    #         # res = (res > 0.5).astype(np.float32)

    #         res = np.transpose(res,(3,0,1,2))

    #         with TiffWriter(filename_saveimg,bigtiff=True) as tif:

    #             for k in range(res.shape[0]):

    #                 tif.write(res[k,:,:,:,] ,compress = 2)

    #         # res_loaded = imread(filename_saveimg,key = range(48))

    #         # print(np.sum(np.abs(res[:,0,:,:,] - res_loaded )))
