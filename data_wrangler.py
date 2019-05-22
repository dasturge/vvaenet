import os
import zipfile
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
from PIL import Image

here = os.path.dirname(os.path.realpath(__file__))


def maybe_download_and_extract(path='./data_of'):
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


def get_oxford_data():
    jpeg_path = './data_of/jpg/*.jpg'
    seg_path = './data_of/segmim/*.jpg'
    jpegs = sorted(glob(jpeg_path))
    segs = sorted(glob(seg_path))

    size_limit = 40
    if size_limit is not None:
        jpegs = jpegs[:size_limit]
        segs = segs[:size_limit]

    X = read_data(jpegs, norm=True)
    y = read_data(segs)

    return X, y


def read_prostate_cancer_dataset(box_zipfile):
    out_dir = os.path.join(here, 'data')
    data_dir = os.path.join(out_dir, 'Prostate cancer dataset')
    if not os.path.exists(data_dir):
        with open(box_zipfile, 'rb') as fd:
            zipf = zipfile.ZipFile(fd)
            zipf.extractall(out_dir)
    X = []
    y = []
    for sid in filter(lambda x: not x.startswith('.'),
                      os.listdir(os.path.join(data_dir, 'ADC'))):
        adc_files = pcdata_getter(data_dir, 'ADC', sid)
        prostate_files = pcdata_getter(data_dir, 'prostate_roi', sid)
        t2_files = pcdata_getter(data_dir, 'T2', sid)
        tumor_files = pcdata_getter(data_dir, 'tumor_roi', sid)
        for idx in range(len(adc_files)):
            try:
                iefm = PCImageExample(sid, idx)
                iefm.set_adc_image_file(adc_files[idx])
                iefm.set_t2_image_file(t2_files[idx])
                iefm.set_prostate_roi_file(prostate_files[idx])
                iefm.set_tumor_roi_file(tumor_files[idx])
                # gotta go through each directory and grab relevant files
                iefm.bake()
                xi, yi = iefm.get_data()
                X.append(xi)
                y.append(yi)
            except:
                raise
    #X = TensorDataset(*X)
    #y = TensorDataset(*y)

    return X, y


def one_hot(targets):
    C = targets.max() + 1
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(0) # convert to 1xHxW
    one_hot = torch.cuda.FloatTensor(C, targets_extend.size(1), targets_extend.size(2)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot


def pcdata_getter(base_dir, imagetype, subject_id):
    path = os.path.join(base_dir, imagetype, subject_id)
    _, _, files = next(os.walk(path))
    files = filter(lambda x: not x.startswith('.'), files)
    fpaths = [os.path.join(path, f) for f in files]
    return fpaths


class PCImageExample():

    gid_list = set()

    def __init__(self, subject_id, slice_idx):
        """
        handler for constructing PyTorch Tensors from dataset
        :param subject_id: identifier for separate scans
        :param slice_idx: identifier for slices of scans
        """
        self.subject_id = subject_id
        self.slice_idx = slice_idx
        self.gid = '%s_%s' % (subject_id, slice_idx)
        assert self.gid not in self.gid_list, \
            'subject id and slice index not unique for %s' % \
            self.__class__.__name__
        self.gid_list.add(self.gid)
        self.adc_image_file = None
        self.t2_image_file = None
        self.prostate_roi_file = None
        self.tumor_roi_file = None

    def set_adc_image_file(self, adc_image_file):
        self.adc_image_file = adc_image_file

    def set_t2_image_file(self, t2_image_file):
        self.t2_image_file = t2_image_file

    def set_prostate_roi_file(self, prostate_roi_file):
        self.prostate_roi_file = prostate_roi_file

    def set_tumor_roi_file(self, tumor_roi_file):
        self.tumor_roi_file = tumor_roi_file

    def bake(self, **options):
        adc_image = np.asarray(Image.open(self.adc_image_file))
        t2_image = np.asarray(Image.open(self.t2_image_file))[:, :, 0]
        example_image = np.stack((t2_image, adc_image))
        self.example_image = torch.Tensor(example_image)

        prostate_roi = np.asarray(Image.open(self.prostate_roi_file), dtype=np.uint8)
        tumor_roi = np.where(np.asarray(Image.open(self.tumor_roi_file)), 1, 0)
        if len(tumor_roi.shape) == 3:
            tumor_roi = tumor_roi[:, :, 0]
        roi = torch.Tensor(prostate_roi + tumor_roi)
        example_roi = one_hot(roi)
        #example_roi = np.stack((bg_roi, tumor_roi, prostate_roi))
        self.example_roi = torch.Tensor(example_roi)

    def get_data(self):
        return self.example_image, self.example_roi


if __name__ == '__main__':
    read_prostate_cancer_dataset(
        os.path.expanduser('~/Downloads/Prostate_cancer_dataset.zip'))
