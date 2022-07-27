import torch as t
import numpy as np
import h5py
from unet2d import UNet
import os
import ipdb
from tifffile import imread
from utils.utils import norm_percentile_nocrop, minmax
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    cuda = False
    device = t.device('cuda' if t.cuda.is_available() and cuda else 'cpu')
    model = UNet(3, 1).to(device)
    model.load_state_dict(t.load('./detection_model.pt'))
    model.eval()
    root = "/mnt/deep_foci/domenico_dataset/"
    files = [os.path.join(root, f) for f in os.listdir(root)]
    imgsize = 96
    normalization_percentile = 0.0001
    results = []
    with t.no_grad():
        for f in tqdm(files[:5]):
            input = norm_percentile_nocrop(t.from_numpy(imread(f)[2:-2, 2:-2]).permute(2, 0, 1).to(device), normalization_percentile)
            output = t.zeros(input.shape)
            for i in range(3):
                for j in range(3):
                    index = ((i*imgsize,(i+1)*imgsize), (j*imgsize, (j+1)*imgsize))
                    input_cropped = input[:, index[0][0]:index[0][1], index[1][0]:index[1][1]].unsqueeze(0)
                    output_cropped = model(input_cropped).squeeze(0)
                    output_cropped[output_cropped > 0.5] = 1
                    output_cropped[output_cropped <= 0.5] = 0
                    output[:, index[0][0]:index[0][1], index[1][0]:index[1][1]] = output_cropped
            results.append((input.permute(1, 2, 0).cpu().numpy(), output.permute(1, 2, 0).cpu().numpy()))

    for (input, out) in results:
        plt.close()
        plt.clf()
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(minmax(input))
        axes[1].imshow(minmax(out))
        plt.show()


if __name__ == '__main__':
    main()
