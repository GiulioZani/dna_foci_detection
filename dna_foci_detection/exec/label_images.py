import torch as t
import os
from tqdm import tqdm
import imageio
import numpy as np
from scipy import ndimage as ndi
import ipdb
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import json


def max_distance(coords, debug=False):
    max_x = t.max(coords[0])
    max_y = t.max(coords[1])
    min_x = t.min(coords[0])
    min_y = t.min(coords[1])
    return t.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)


def extract_label(label_img, color=(1, 0, 1)):
    label_img = watershed_label(label_img)
    # print(f"{t.unique(label_img)=}")
    label_img = label_img.squeeze(0)
    values = np.unique(label_img)
    labels = []
    for value in values:
        if value == 0:
            continue
        value_labels = []
        labels.append(value_labels)
        for i in range(label_img.shape[0]):
            for j in range(label_img.shape[1]):
                if label_img[i, j] == value:
                    value_labels.append([i, j, 1])
    foci = []
    for label in labels:
        label = t.tensor(label).T.float()
        label_means = t.mean(label, dim=1)
        r = max_distance(label) / 2
        if r > 0:
            label_means[-1] = r
            foci.append([label_means[0], label_means[1], label_means[2]])
    labels = t.tensor(foci)
    return labels


def watershed_label(image):
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    image = image.cpu().numpy().astype(int)
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    return t.from_numpy(labels)


def run(model):
    """
    Labels new images using the trained model.
    """
    images_path = "test_images"  # "/mnt/deep_foci/new_images"
    out_path = "test_label"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # Set the model to eval mode
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = model.to(device)
    images = [os.path.join(images_path, f) for f in os.listdir(images_path)]
    model.eval()
    # Create a dataloader for the test dataset
    with t.no_grad():
        for image_path in tqdm(images):

            image = (
                t.from_numpy(imageio.imread(image_path))
                .to(device)
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255
            )
            # Move the inputs and labels to the GPU
            # Forward pass
            predicted_label_img = (
                model(image).squeeze(0).squeeze(0) > 0.5
            ).float()
            pred_label = extract_label(predicted_label_img)
            new_name = os.path.basename(image_path).replace(".png", ".json")
            print(pred_label)
            with open(os.path.join(out_path, new_name), "w") as f:
                json.dump(pred_label.tolist(), f)
            print(f"Created file {new_name}")
            # Get the index of the max log-probability
