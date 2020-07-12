import json
import os
from copy import deepcopy

from eisen.models.segmentation import UNet3D
from eisen.datasets import MSDDataset
from eisen.ops.losses import DiceLoss
from eisen.io import LoadNiftiFromFilename
from eisen.ops.losses import dice
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
from eisen.utils import EisenModuleWrapper, EisenDatasetSplitter
from eisen.utils.workflows import Training, Testing
from eisen.utils.logging import LoggingHook, TensorboardSummaryHook

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import Interpolate, UVAENet
from losses import KLDivergence


class DeepCopy:
    def __call__(self, data):
        return deepcopy(data)


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

    threshold_labels = ThresholdValues(["label"], threshold=0.5)

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
            threshold_labels,
            add_channel,
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
        semantic_config={"output_channels": output_channels, "imdim": 3,},
    )

    base_module = UNet3D(
        input_channels=input_channels,
        output_channels=output_channels,
        outputs_activation="softmax",
    )
    if torch.cuda.device_count() > 1:
        base_module = nn.DataParallel(base_module)

    model = EisenModuleWrapper(
        module=base_module,
        input_names=["image"],
        output_names=["segmentation"],  #  "reconstruction", "mean", "log_variance"],
    )

    dice_loss = EisenModuleWrapper(
        module=DiceLoss(),
        input_names=["label", "segmentation"],
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
        input_names=["label", "segmentation"],
        output_names=["dice_metric"],
    )

    optimizer = Adam(model.parameters(), 1e-3)

    training = Training(
        model=model,
        losses=[dice_loss],  # , reconstruction_loss, kl_loss],
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
