import os
from glob import glob

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import model
import train


def maybe_download_and_extract(path='./data'):
    if not os.path.exists(path):
        raise NotImplemented

def read_data(paths, dtype=None, norm=False):
    images = []
    for p in paths:
        img = Image.open(p)
        img = F.center_crop(img, (500, 500))
        img = np.asarray(img)
        if dtype:
            img = img.astype(dtype)
        if norm:
            img = img / 255
        img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
        images.append(img)
    return images

def main():
    jpeg_path = './data/jpg/*.jpg'
    seg_path = './data/segmim/*.jpg'
    jpegs = sorted(glob(jpeg_path))
    segs = sorted(glob(seg_path))

    size_limit = 40
    if size_limit is not None:
        jpegs = jpegs[:size_limit]
        segs = segs[:size_limit]

    X = read_data(jpegs, norm=True)
    y = read_data(segs)
    net = model.UVAENet(vae_config={'initial_channels': 4, 'input_size': 62 * 62})
    train.train(net, X, y, epochs=1)


if __name__ == '__main__':
    main()
