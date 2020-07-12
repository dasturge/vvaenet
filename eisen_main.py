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
from eisen.utils import EisenModuleWrapper
from eisen.utils.workflows import Training, Testing
from eisen.utils.logging.logs import LoggingHook
from torch import mul

import torch.nn as nn
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import UVAENet
from losses import KLDivergence


def main():

    # Defining some constants
    PATH_DATA = "./Task01_BrainTumour"
    PATH_ARTIFACTS = "./results"

    NAME_MSD_JSON = "dataset.json"

    NUM_EPOCHS = 100
    BATCH_SIZE = 1

    # create a transform to manipulate and load data
    # image manipulation transforms
    read_tform = LoadNiftiFromFilename(["image", "label"], PATH_DATA)
    resample_tform = ResampleNiftiVolumes(["image", "label"], [1.0, 1.0, 1.0], "linear")

    image_to_numpy_tform = NiftiToNumpy(["image"], multichannel=True)
    label_to_numpy_tform = NiftiToNumpy(["label"])

    crop = CropCenteredSubVolumes(fields=["image", "label"], size=[64, 64, 64])

    add_channel = AddChannelDimension(["label"])

    map_intensities = MapValues(["image"], min_value=0.0, max_value=1.0)

    threshold_labels = ThresholdValues(["label"], threshold=0.5)

    tform = Compose(
        [
            read_tform,
            resample_tform,
            image_to_numpy_tform,
            label_to_numpy_tform,
            crop,
            map_intensities,
            threshold_labels,
            add_channel,
        ]
    )

    train_dataset = MSDDataset(PATH_DATA, NAME_MSD_JSON, "training", transform=tform)
    test_dataset = MSDDataset(PATH_DATA, NAME_MSD_JSON, "test", transform=tform)

    # create data loader for training, this functionality is pure pytorch
    data_loader_train = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    # create one for testing (shuffle off)
    data_loader_test = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = EisenModuleWrapper(
        module=UVAENet(
            train_dataset[0]["image"].shape,
            encoder_config={"input_channels": 4, "imdim": 3,},
            vae_config={"initial_channels": 4, "imdim": 3,},
            semantic_config={"output_channels": 2, "imdim": 3,},
        ),
        input_names=["image"],
        output_names=["segmentation", "reconstruction", "mean", "log_variance"],
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
    train_hook = LoggingHook(training.id, "Training", "./")
    test_hook = LoggingHook(testing.id, "Testing", "./")
    # run optimization for NUM_EPOCHS
    for i in range(NUM_EPOCHS):
        training.run()
        testing.run()


if __name__ == "__main__":
    main()