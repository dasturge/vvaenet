import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
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
from eisen.utils import EisenModuleWrapper, EisenDatasetSplitter
from eisen.utils.artifacts import SaveTorchModelHook
from eisen.utils.workflows import Training
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import Compose

from eisen_mods.hooks import PatienceHook, LoggingHook
from eisen_mods.summaries import TensorboardSummaryHook
from eisen_mods.workflows import Testing
from model import UVAENet
from losses import KLDivergence, SingleDiceLoss, SingleDiceMetric
from util import *


def run_decathalon_step(task_path, use_multiclass=True):
    """
    executes training for one of the tasks of the medical segmentation decathalon

    :param task_path: Path to the MSD dataset folder
    :type task_path: str
    :param use_multiclass: 
      If False, just does background/foreground classification for the task, defaults to True
    :type use_multiclass: bool, optional
    """

    # Defining some constants
    artifacts_dir = "./results/%s" % os.path.basename(task_path)
    os.makedirs(artifacts_dir, exist_ok=True)

    metadata_json = "dataset.json"

    with open(os.path.join(task_path, metadata_json)) as fd:
        msd_metadata = json.load(fd)
    input_channels = len(msd_metadata["modality"])
    output_channels = len(msd_metadata["labels"])
    multichannel = input_channels > 1
    multiclass = output_channels > 2 and use_multiclass

    read_tform = LoadNiftiFromFilename(["image", "label"], task_path)
    image_to_numpy_tform = NiftiToNumpy(["image"], multichannel=multichannel)

    single_image = deepcopy(msd_metadata["training"][0])
    example = image_to_numpy_tform(read_tform(single_image))
    original_size = example["image"].shape[-3:]
    original_image_resolution = example["image_affines"][np.diag_indices(3)]

    NUM_EPOCHS = 100
    BATCH_SIZE = 4

    MEMORY_LIMIT = 4 * 3932160 / 2
    IMAGE_SIZE = original_size
    IMAGE_RESOLUTION = original_image_resolution
    factor = 1

    while input_channels * np.product(IMAGE_SIZE) > MEMORY_LIMIT:
        factor += 1
        if factor == 2:
            IMAGE_SIZE = [s // factor for s in original_size]
            IMAGE_SIZE = [
                s + 16 - s % 16 if s % 16 else s for s in IMAGE_SIZE
            ]  # makes convolutions cleaner
            IMAGE_RESOLUTION = [r * factor for r in original_image_resolution]
        else:
            IMAGE_SIZE = [s - 16 if s > 96 else s for s in original_size]
            IMAGE_RESOLUTION = [
                r * round(s2 / s1)
                for s1, s2, r in zip(
                    original_size, IMAGE_SIZE, original_image_resolution
                )
            ]

    # image manipulation transforms
    deepcopy_tform = DeepCopy()
    read_tform = LoadNiftiFromFilename(["image", "label"], task_path)
    image_resample_tform = ResampleNiftiVolumes(["image"], IMAGE_RESOLUTION, "linear")
    label_resample_tform = ResampleNiftiVolumes(["label"], IMAGE_RESOLUTION, "nearest")
    image_to_numpy_tform = NiftiToNumpy(["image"], multichannel=multichannel)
    label_to_numpy_tform = NiftiToNumpy(["label"])

    crop = CropCenteredSubVolumes(fields=["image", "label"], size=IMAGE_SIZE)

    map_intensities = MapValues(["image"], min_value=0.0, max_value=1.0)

    one_hotify = OneHotify(["label"], num_classes=output_channels)
    transpose = Transpose(["label"], [3, 0, 1, 2])
    add_channel_dimension = AddChannelDimension(
        ["label"] + (["image"] if output_channels > 2 else [])
    )
    threshold_labels = ThresholdValues(["label"], threshold=0.5)
    tform_list = [
        deepcopy_tform,
        read_tform,
        image_resample_tform,
        label_resample_tform,
        image_to_numpy_tform,
        label_to_numpy_tform,
        crop,
        map_intensities,
    ]
    if multiclass:
        tform_list += [one_hotify, transpose]
    else:
        tform_list += [add_channel_dimension, threshold_labels]
    tform = Compose(tform_list)

    # create a dataset from the training set of the MSD dataset
    dataset = MSDDataset(task_path, metadata_json, "training", transform=None)
    splitter = EisenDatasetSplitter(
        0.8, 0.2, 0.0, transform_train=tform, transform_valid=tform
    )
    train_dataset, test_dataset, _ = splitter(dataset)
    data_loader_train = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        worker_init_fn=np.random.seed,
    )
    data_loader_test = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # construct new segmentation model with random weights
    base_module = UVAENet(
        train_dataset[0]["image"].shape,
        encoder_config={"input_channels": input_channels, "imdim": 3,},
        vae_config={"initial_channels": 4, "imdim": 3,},
        semantic_config={
            "output_channels": output_channels if multiclass else 1,
            "imdim": 3,
            "final_activation": "softmax" if multiclass else "sigmoid",
        },
    )
    if torch.cuda.device_count() > 1 and BATCH_SIZE > 1:
        base_module = nn.DataParallel(base_module)
    model = EisenModuleWrapper(
        module=base_module,
        input_names=["image"],
        output_names=["segmentation", "reconstruction", "mean", "log_variance",],
    )

    # define model 3 model losses: dice, reconstruction, and kl divergence.
    # for multiclass segmentation, use the average of the class dice losses.
    if multiclass:
        dice_losses = [
            EisenLossWrapper(
                module=SingleDiceLoss(int(i)),
                input_names=["segmentation", "label"],
                output_names=["%s_dice_loss" % name],
                weight=1.0 / (len(msd_metadata["labels"]) - 1),
            )
            for i, name in msd_metadata["labels"].items()
            if int(i)
        ]
        metrics = [
            EisenModuleWrapper(
                module=SingleDiceMetric(int(i)),
                input_names=["segmentation", "label"],
                output_names=["%s_dice_metric" % name],
            )
            for i, name in msd_metadata["labels"].items()
            if int(i)
        ]
    else:
        dice_losses = [
            EisenLossWrapper(
                module=DiceLoss(),
                input_names=["segmentation", "label"],
                output_names=["dice_loss"],
                weight=1.0,
            )
        ]
        metrics = [
            EisenModuleWrapper(
                module=DiceMetric(),
                input_names=["segmentation", "label"],
                output_names=["dice_metric"],
            )
        ]
    reconstruction_loss = EisenLossWrapper(
        module=nn.MSELoss(),
        input_names=["image", "reconstruction"],
        output_names=["reconstruction_loss"],
        weight=0.1,
    )
    kl_loss = EisenLossWrapper(
        module=KLDivergence(N=np.product(IMAGE_SIZE)),
        input_names=["mean", "log_variance"],
        output_names=["kl_loss"],
        weight=0.1,
    )
    optimizer = Adam(model.parameters(), 1e-4, weight_decay=1e-5)

    # define eisen workflows for training and testing
    training_workflow = Training(
        model=model,
        losses=dice_losses + [reconstruction_loss, kl_loss],
        data_loader=data_loader_train,
        optimizer=optimizer,
        metrics=metrics,
        gpu=True,
    )
    testing_workflow = Testing(
        model=model, data_loader=data_loader_test, metrics=metrics, gpu=True
    )
    # set of hooks logs all artifacts
    hooks = [
        LoggingHook(training_workflow.id, "Training", artifacts_dir),
        LoggingHook(testing_workflow.id, "Testing", artifacts_dir),
        TensorboardSummaryHook(training_workflow.id, "Training", artifacts_dir),
        TensorboardSummaryHook(testing_workflow.id, "Testing", artifacts_dir),
        SaveTorchModelHook(
            testing_workflow.id, "Testing", artifacts_dir, select_best_loss=False
        ),
    ]
    patience = PatienceHook(testing_workflow.id, 10, select_best_loss=False)

    output_metadata = {
        "task": task_path,
        "original_image_size": original_size,
        "original_image_resolution": original_image_resolution,
        "final_image_resolution": IMAGE_SIZE,
        "final_image_size": IMAGE_RESOLUTION,
    }

    # run optimization for NUM_EPOCHS
    for i in range(NUM_EPOCHS):
        if patience.is_stop():
            print("early stopping triggered on epoch %s" % (i - 1))
            break
        training_workflow.run()
        testing_workflow.run()


if __name__ == "__main__":
    task_folders = sorted(glob("Task*_*"))
    for i, task_folder in enumerate(task_folders):
        try:
            run_decathalon_step(task_folder)
        except:
            if i:
                print("failed on %s" % task_folder)
                import traceback

                traceback.print_exc()
            else:
                raise
