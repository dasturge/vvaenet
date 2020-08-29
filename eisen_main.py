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
from torch.nn.modules.activation import Tanhshrink
from torch.nn.parallel.data_parallel import DataParallel
from torch.optim import adam
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import Compose

from eisen_mods.hooks import PatienceHook, LoggingHook, SaveArtifactsHook
from eisen_mods.summaries import TensorboardSummaryHook
from eisen_mods.workflows import Testing
from model import UVAENet, ParallelUVAENet
from losses import KLDivergence, SingleDiceLoss, SingleDiceMetric
from util import *


def run_decathalon_step(
    task_path,
    use_multiclass=True,
    relu_type="lrelu",
    variational=True,
    model_parallel=True,
    run_shape=None,
    run_resolution=None,
):
    """
    executes training for one of the tasks of the medical segmentation decathalon

    :param task_path: Path to the MSD dataset folder
    :type task_path: str
    :param use_multiclass: 
      If False, just does background/foreground classification for the task, defaults to True
    :type use_multiclass: bool, optional
    """
    unet_type = "uvaenet" if variational else "unet"
    if relu_type == "lrelu":
        activation = nn.LeakyReLU(0.2)
    elif relu_type == "relu":
        activation = nn.ReLU()
    elif relu_type == "rlrelu":
        activation = nn.RReLU(lower=0.125, upper=1 / 3)

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
    BATCH_SIZE = 1 if model_parallel else torch.cuda.device_count()

    # image manipulation transforms
    deepcopy_tform = DeepCopy()
    read_tform = LoadNiftiFromFilename(["image", "label"], task_path)
    image_resample_tform = ResampleNiftiVolumes(["image"], run_resolution, "linear")
    label_resample_tform = ResampleNiftiVolumes(["label"], run_resolution, "nearest")
    image_to_numpy_tform = NiftiToNumpy(["image"], multichannel=multichannel)
    label_to_numpy_tform = NiftiToNumpy(["label"])

    crop = CropCenteredSubVolumes(fields=["image", "label"], size=run_shape)

    map_intensities = MapValues(["image"], min_value=0.0, max_value=1.0)

    one_hotify = OneHotify(["label"], num_classes=output_channels)
    transpose = Transpose(["label"], [3, 0, 1, 2])
    add_channel_dimension = AddChannelDimension(["label"])
    add_channel_dimension_input = AddChannelDimension(["image"])
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
    if not multichannel:
        tform_list += [add_channel_dimension_input]
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
    if model_parallel:
        Net = ParallelUVAENet
    else:
        Net = UVAENet

    base_module = Net(
        train_dataset[0]["image"].shape,
        encoder_config={"input_channels": input_channels, "imdim": 3,},
        vae_config={"initial_channels": 4, "imdim": 3,},
        semantic_config={
            "output_channels": output_channels if multiclass else 1,
            "imdim": 3,
            "final_activation": "softmax" if multiclass else "sigmoid",
        },
    )
    if not model_parallel and torch.cuda.device_count() > 1:
        base_module = DataParallel(base_module)

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
        module=KLDivergence(N=np.product(run_shape)),
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
        SaveArtifactsHook(
            training_workflow.id,
            "Training",
            artifacts_dir,
            "_%s_%s" % (unet_type, relu_type),
        ),
        SaveArtifactsHook(
            training_workflow.id,
            "Testing",
            artifacts_dir,
            "_%s_%s" % (unet_type, relu_type),
        ),
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
        "original_image_size": list(original_size),
        "original_image_resolution": list(original_image_resolution),
        "final_image_resolution": run_resolution,
        "final_image_size": run_shape,
    }
    print(output_metadata)
    with open(os.path.join("./results", task_path.rstrip("/") + ".json"), "w") as fd:
        json.dump(output_metadata, fd)

    # run optimization for NUM_EPOCHS
    for i in range(NUM_EPOCHS):
        if patience.is_stop():
            print("early stopping triggered on epoch %s" % (i - 1))
            break
        training_workflow.run()
        testing_workflow.run()


if __name__ == "__main__":
    import pandas as pd

    # get task metadata and prepare run params
    task_folders = sorted(glob("Task*_*"))
    metadata_list = []
    for folder in glob("Task*"):
        json_file = os.path.join(folder, "dataset.json")
        with open(json_file) as fd:
            metadata = json.load(fd)
        metadata["folder"] = folder
        metadata_list.append(metadata)
    df = pd.DataFrame(metadata_list).sort_values(by="folder")

    df["multiclass"] = df["labels"].apply(len) > 2
    df["multichannel"] = df["modality"].apply(len) > 1

    def get_one_image(folder):
        train_folder = os.path.join(folder, "imagesTr")
        image = os.listdir(train_folder)[0]
        image = os.path.join(train_folder, image)
        image = os.path.abspath(image)
        return {"image": image}

    xfm = Compose(
        [get_one_image, LoadNiftiFromFilename(["image"], "/"), lambda x: x["image"],]
    )
    df["nifti"] = df.folder.apply(xfm)
    df["shape"] = df["nifti"].apply(lambda x: x.header.get_data_shape()[:3])
    df["resolution"] = df["nifti"].apply(lambda x: x.header.get_zooms()[:3])
    df["run_shape"] = df["folder"].map(
        {
            "Task01_BrainTumour": (160, 192, 128),
            "Task02_Heart": (320, 320, 120),
            "Task04_Hippocampus": (35, 51, 34),
            "Task05_Prostate": (320, 320, 24),
            "Task06_Lung": [x // 2 - (x // 2 % 16) for x in (512, 512, 531)],
            "Task07_Pancreas": [x - 48 for x in (512, 512)] + [81],
            "Task08_HepaticVessel": (512, 512, 35),
            "Task09_Spleen": (512, 512, 44),
            "Task10_Colon": [x // 2 - (x // 2 % 16) for x in (512, 512, 100)],
        }
    )
    df["run_resolution"] = df["folder"].map(
        {
            "Task01_BrainTumour": (1.0, 1.0, 1.0),
            "Task02_Heart": (1.25, 1.25, 1.37),
            "Task04_Hippocampus": (1.0, 1.0, 1.0),
            "Task05_Prostate": (0.625, 0.625, 3.6),
            "Task06_Lung": [2.0 * x for x in (0.820312, 0.820312, 0.625)],
            "Task07_Pancreas": (0.898438, 0.898438, 2.5),
            "Task08_HepaticVessel": (0.878906, 0.878906, 7.5),
            "Task09_Spleen": (0.9375, 0.9375, 5.0),
            "Task10_Colon": [2.0 * x for x in (0.583984, 0.583984, 4.0)],
        }
    )

    for i, task_metadata in df.iterrows():
        for relu_type in ["lrelu", "rlrelu", "relu"]:
            try:
                run_decathalon_step(
                    task_metadata["folder"],
                    run_shape=task_metadata["run_shape"],
                    run_resolution=task_metadata["run_resolution"],
                )
            except:
                if task_metadata["folder"] != "Task01_BrainTumour":
                    print("failed on %s" % task_metadata["folder"])
                    import traceback

                    traceback.print_exc()
                else:
                    raise


################# COMPLETION STEPS #######################

# Step 1: complete model construction for various cases

# Step 2: distributed model execution

