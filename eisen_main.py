import json
import os
from copy import deepcopy

from eisen.models.segmentation import UNet3D
from eisen.datasets import MSDDataset
from eisen.ops.losses import DiceLoss
from eisen.io import LoadNiftiFromFilename
from eisen.transforms import (
    ResampleNiftiVolumes,
    NiftiToNumpy,
    CropCenteredSubVolumes,
    AddChannelDimension,
    MapValues,
    ThresholdValues,
)
from eisen.ops.losses import DiceLoss
from eisen.ops.metrics import DiceMetric
from eisen.transforms.imaging import RenameFields
from eisen.utils import EisenModuleWrapper, EisenDatasetSplitter
from eisen.utils.workflows import Training, Testing
from eisen.utils.logging import LoggingHook, TensorboardSummaryHook

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import Interpolate, UVAENet
from losses import KLDivergence


class DeepCopy:
    def __call__(self, data):
        return deepcopy(data)


class CopyFields:
    """
    Transform allowing to copy fields in the data dictionary, performs a deepcopy operation

    .. code-block:: python

        from eisen.transforms import CopyFields
        tform = CopyFields(['old_name1', 'old_name2'], ['new_name1', 'new_name2'])
        tform = tform(data)

    """

    def __init__(self, fields, new_fields):
        """
        :param fields: list of names of the fields of data dictionary to copy
        :type fields: list of str
        :param new_fields: new field names for the data dictionary
        :type new_fields: list of str

        .. code-block:: python

            from eisen.transforms import CopyFields

            tform = CopyFields(
                fields=['old_name1', 'old_name2'],
                new_fields=['new_name1', 'new_name2']
            )

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "new_fields", "type": "list:string", "value": ""}
        ]
        </json>
        """
        self.fields = fields
        self.new_fields = new_fields

        assert len(self.new_fields) == len(self.fields)

    def __call__(self, data):
        for field, new_field in zip(self.fields, self.new_fields):
            data[new_field] = deepcopy(data[field])

        return data


class OneHotify:
    def __init__(self, fields, num_classes=None, dtype=np.float) -> None:
        self.fields = fields
        self.num_classes = num_classes
        self.dtype = dtype

    def __call__(self, data):
        for field in self.fields:
            x = np.asarray(data[field], dtype=np.int)
            n = np.max(x) + 1 if self.num_classes is None else self.num_classes
            data[field] = np.eye(n, dtype=self.dtype)[x]
        return data


class Transpose:
    def __init__(self, fields, order) -> None:
        self.fields = fields
        self.order = order

    def __call__(self, data):
        for field in self.fields:
            data[field] = data[field].transpose(self.order)
        return data


class RemoveChannel:
    def __init__(self, fields) -> None:
        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            data[field] = data[field][1:, ...]
        return data


def main():

    # Defining some constants
    PATH_DATA = "./Task01_BrainTumour"
    PATH_ARTIFACTS = "./results"

    NAME_MSD_JSON = "dataset.json"

    with open(os.path.join(PATH_DATA, NAME_MSD_JSON)) as fd:
        msd_metadata = json.load(fd)
    input_channels = len(msd_metadata["modality"])
    output_channels = len(msd_metadata["labels"])

    NUM_EPOCHS = 100
    BATCH_SIZE = 16

    # create a transform to manipulate and load data
    # image manipulation transforms
    deepcopy_tform = DeepCopy()
    read_tform = LoadNiftiFromFilename(["image", "label"], PATH_DATA)
    image_resample_tform = ResampleNiftiVolumes(["image"], [2.0, 2.0, 2.0], "linear")
    label_resample_tform = ResampleNiftiVolumes(["label"], [2.0, 2.0, 2.0], "nearest")

    image_to_numpy_tform = NiftiToNumpy(["image"], multichannel=True)
    label_to_numpy_tform = NiftiToNumpy(["label"])

    crop = CropCenteredSubVolumes(fields=["image", "label"], size=[64, 64, 64])

    add_channel = AddChannelDimension(["label"])

    map_intensities = MapValues(["image"], min_value=0.0, max_value=1.0)

    rename_fields = RenameFields(["label"], ["one_hot"])
    one_hotify = OneHotify(["one_hot"], num_classes=output_channels)
    transpose = Transpose(["one_hot"], [3, 0, 1, 2])
    remove_channel = RemoveChannel(["one_hot"])

    # threshold_labels = ThresholdValues(["label"], threshold=0.5)

    tform = Compose(
        [
            deepcopy_tform,
            read_tform,
            image_resample_tform,
            label_resample_tform,
            image_to_numpy_tform,
            label_to_numpy_tform,
            crop,
            map_intensities,
            rename_fields,
            one_hotify,
            transpose,
            remove_channel,
        ]
    )

    # create a dataset from the training set of the MSD dataset
    dataset = MSDDataset(PATH_DATA, NAME_MSD_JSON, "training", transform=None)

    # define a splitter to do a 80%-20% split of the data
    splitter = EisenDatasetSplitter(
        0.8, 0.2, 0.0, transform_train=tform, transform_valid=tform
    )

    # define a training and test sets
    train_dataset, test_dataset, _ = splitter(dataset)

    # create data loader for training, this functionality is pure pytorch
    data_loader_train = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    # create one for testing (shuffle off)
    data_loader_test = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    base_module = UVAENet(
        train_dataset[0]["image"].shape,
        encoder_config={"input_channels": input_channels, "imdim": 3,},
        vae_config={"initial_channels": 4, "imdim": 3,},
        semantic_config={
            "output_channels": output_channels - 1,
            "imdim": 3,
            "final_activation": "sigmoid",
        },
    )

    # base_module = UNet3D(
    #    input_channels=input_channels,
    #    output_channels=output_channels,
    #    outputs_activation="softmax",
    # )
    if torch.cuda.device_count() > 1:
        base_module = nn.DataParallel(base_module)

    model = EisenModuleWrapper(
        module=base_module,
        input_names=["image"],
        output_names=["segmentation", "reconstruction", "mean", "log_variance",],
    )

    dice_loss = EisenModuleWrapper(
        module=DiceLoss(),
        input_names=["one_hot", "segmentation"],
        output_names=["dice_loss"],
    )
    reconstruction_loss = EisenModuleWrapper(
        module=nn.MSELoss(),
        input_names=["image", "reconstruction"],
        output_names=["reconstruction_loss"],
    )
    kl_loss = EisenModuleWrapper(
        module=KLDivergence(),
        input_names=["mean", "log_variance"],
        output_names=["kl_loss"],
    )

    metric = EisenModuleWrapper(
        module=DiceMetric(),
        input_names=["one_hot", "segmentation"],
        output_names=["dice_metric"],
    )

    optimizer = Adam(model.parameters(), 1e-3)

    training = Training(
        model=model,
        losses=[dice_loss, reconstruction_loss, kl_loss],
        data_loader=data_loader_train,
        optimizer=optimizer,
        metrics=[metric],
        gpu=True,
    )
    # make another workflow for testing
    testing = Testing(
        model=model, data_loader=data_loader_test, metrics=[metric], gpu=True
    )
    train_hook = LoggingHook(training.id, "Training", PATH_ARTIFACTS)
    test_hook = LoggingHook(testing.id, "Testing", PATH_ARTIFACTS)
    train_board = TensorboardSummaryHook(training.id, "Training", PATH_ARTIFACTS)
    test_board = TensorboardSummaryHook(testing.id, "Testing", PATH_ARTIFACTS)

    # run optimization for NUM_EPOCHS
    for i in range(NUM_EPOCHS):
        training.run()
        testing.run()


if __name__ == "__main__":
    main()
