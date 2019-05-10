import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
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
        img = Ft.center_crop(img, (384, 384))
        img = np.asarray(img)
        if dtype:
            img = img.astype(dtype)
        if norm:
            img = img / 255
            img = torch.Tensor(img).permute(2, 0, 1)
        else:
            img = np.apply_along_axis(
                lambda x: (x != np.array([0, 0, 254])),
                arr=img, axis=-1).astype(int)[:, :, 0]
            img = torch.Tensor(img)
            img = F.one_hot(img.long(), num_classes=2).permute(2, 0, 1)
        img = img.unsqueeze(0)
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
    net = model.UVAENet(X[0].shape,
                        encoder_config={'input_channels': 3},
                        vae_config={'initial_channels': 4},
                        semantic_config={'output_channels': 2})
    train.train(net, X, y, epochs=1, batch_size=4)


if __name__ == '__main__':
    main()
