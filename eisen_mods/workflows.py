import logging
import numpy as np
import torch
import uuid

from eisen import EISEN_END_BATCH_EVENT, EISEN_END_EPOCH_EVENT
from eisen.utils.workflows import GenericWorkflow, EpochDataAggregator
from eisen.utils import merge_two_dicts
from pydispatch import dispatcher


class Testing(GenericWorkflow):
    """
    This object implements a testing workflow, which allows to test a model on a certain dataset by running only the
    forward pass of the model. The user is allowed to specify model, data loader and metrics to use for evaluation.
    This workflow supports GPUs and data parallelism across multiple processors.
    """

    def __init__(self, model, data_loader, metrics, gpu=False):
        """
        :param model: The model to be used for testing. This model instance will be used only for forward passes.
        :type model: torch.nn.Module
        :param data_loader: A DataLoader instance which handles the data loading and batching
        :type data_loader: torch.utils.data.DataLoader
        :param metrics: A list of metrics objects to be evaluated during test
        :type metrics: list
        :param gpu: A flag indicating whether GPUs should be used during test
        :type gpu: bool

        <json>
        [
            {"name": "gpu", "type": "bool", "value": "false"}
        ]
        </json>
        """

        super(Testing, self).__init__(model, gpu)

        self.data_loader = data_loader
        self.metrics = metrics

        self.epoch_aggregator = EpochDataAggregator(self.id)

        self.epoch = 0

    def get_output_dictionary(self, batch):
        """
        Calls the class on the batch and converts output tuple to an output dictionary.

        :param batch: a dictionary containing a batch of data (as per Eisen specifications)
        :type batch: dict

        :return: output dictionary

        """

        outputs, losses, metrics = super(Testing, self).__call__(batch)

        output_dictionary = {
            "inputs": batch,
            "losses": losses,
            "outputs": outputs,
            "metrics": metrics,
            "model": self.model,
            "epoch": self.epoch,
        }

        return output_dictionary

    def run(self, epoch=None):
        logging.info("INFO: Running Testing")

        self.model.eval()

        if epoch is not None:
            self.epoch = epoch

        with self.epoch_aggregator as ea:
            with torch.no_grad():
                for i, batch in enumerate(self.data_loader):
                    if self.gpu:
                        for key in batch.keys():
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].cuda()

                    logging.debug("DEBUG: Testing epoch batch {}".format(i))

                    output_dictionary = self.get_output_dictionary(batch)

                    dispatcher.send(
                        message=output_dictionary,
                        signal=EISEN_END_BATCH_EVENT,
                        sender=self.id,
                    )

                    ea(output_dictionary)

        dispatcher.send(
            message=ea.epoch_data, signal=EISEN_END_EPOCH_EVENT, sender=self.id
        )
        self.epoch += 1


def convert_output_dict_to_cpu(output_dict):
    for typ in ["losses", "metrics"]:
        for i in range(len(output_dict[typ])):
            for key in list(output_dict[typ][i].keys()):
                if isinstance(output_dict[typ][i][key], torch.Tensor):
                    output_dict[typ][i][key] = (
                        output_dict[typ][i][key].cpu().data.numpy()
                    )
                elif isinstance(output_dict[typ][key], np.ndarray):
                    pass
                else:
                    output_dict[typ][i].pop(key, None)

    for typ in ["inputs", "outputs"]:
        for key in list(output_dict[typ].keys()):
            if isinstance(output_dict[typ][key], torch.Tensor):
                output_dict[typ][key] = output_dict[typ][key].cpu().data.numpy()
            elif isinstance(output_dict[typ][key], np.ndarray):
                pass
            else:
                output_dict[typ].pop(key, None)

    return output_dict


"""
class EpochDataAggregator:
    def __init__(self, workflow_id):
        self.best_avg_loss = 10 ** 10
        self.best_avg_metric = -(10 ** 10)
        self.workflow_id = workflow_id

    def __enter__(self):
        self.epoch_data = {}

        return self

    def __call__(self, output_dictionary):
        output_dictionary = convert_output_dict_to_cpu(output_dictionary)

        self.epoch_data["epoch"] = output_dictionary["epoch"]
        self.epoch_data["model"] = output_dictionary["model"]

        for typ in ["losses", "metrics"]:
            if typ not in self.epoch_data.keys():
                self.epoch_data[typ] = [{} for _ in range(len(output_dictionary[typ]))]

            for i in range(len(output_dictionary[typ])):
                for key in output_dictionary[typ][i].keys():
                    try:
                        data = output_dictionary[typ][i][key]
                        if isinstance(data, np.ndarray):
                            if key not in self.epoch_data[typ][i].keys():
                                self.epoch_data[typ][i][key] = [data]
                            else:
                                self.epoch_data[typ][i][key].append(data)
                    except KeyError:
                        pass

        for typ in ["inputs", "outputs"]:
            if typ not in self.epoch_data.keys():
                self.epoch_data[typ] = {}

            for key in output_dictionary[typ].keys():
                try:
                    data = output_dictionary[typ][key]

                    if isinstance(data, np.ndarray):
                        if key not in self.epoch_data[typ].keys():
                            self.epoch_data[typ][key] = data

                        else:
                            # if data is NOT high dimensional (Eg. it is a vector) we save all of it (throughout the epoch)
                            # the behaviour we want to have is that classification data (for example) can be saved for the
                            # whole epoch instead of only one batch
                            if output_dictionary[typ][key].ndim == 0:
                                output_dictionary[typ][key] = output_dictionary[typ][
                                    key
                                ][np.newaxis]

                            elif output_dictionary[typ][key].ndim == 1:
                                self.epoch_data[typ][key] = np.concatenate(
                                    [
                                        self.epoch_data[typ][key],
                                        output_dictionary[typ][key],
                                    ],
                                    axis=0,
                                )
                            # save logits/multilabel predictions
                            elif output_dictionary[typ][key].ndim == 2:
                                self.epoch_data[typ][key] = np.concatenate(
                                    [
                                        self.epoch_data[typ][key],
                                        output_dictionary[typ][key],
                                    ],
                                    axis=0,
                                )
                            else:
                                # we do not save high dimensional data throughout the epoch, we just save the last batch
                                # the behaviour in this case is to save images and volumes only for the last batch of the epoch
                                self.epoch_data[typ][key] = output_dictionary[typ][key]
                    elif isinstance(data, list):
                        pass
                except KeyError:
                    pass

    def __exit__(self, *args, **kwargs):
        if any([isinstance(x, Exception) for x in args]):
            return
        for typ in ["losses", "metrics"]:
            for i in range(len(self.epoch_data[typ])):
                for key in self.epoch_data[typ][i].keys():
                    self.epoch_data[typ][i][key] = np.asarray(
                        self.epoch_data[typ][i][key]
                    )

        all_losses = []
        for dct in self.epoch_data["losses"]:
            for key in dct.keys():
                all_losses.append(np.mean(dct[key]))

        if len(all_losses) > 0:
            avg_all_losses = np.sum(all_losses)

            if avg_all_losses <= self.best_avg_loss:
                self.best_avg_loss = avg_all_losses
                dispatcher.send(
                    message=self.epoch_data,
                    signal=EISEN_BEST_MODEL_LOSS,
                    sender=self.workflow_id,
                )

        all_metrics = []
        for dct in self.epoch_data["metrics"]:
            for key in dct.keys():
                all_metrics.append(dct[key][-1])

        if len(all_metrics) > 0:
            avg_all_metrics = np.mean(all_metrics)

            if avg_all_metrics >= self.best_avg_metric:
                self.best_avg_metric = avg_all_metrics
                dispatcher.send(
                    message=self.epoch_data,
                    signal=EISEN_BEST_MODEL_METRIC,
                    sender=self.workflow_id,
                )
                """
