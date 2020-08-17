import numpy as np
from pydispatch import dispatcher
from prettytable import PrettyTable

from eisen import EISEN_END_EPOCH_EVENT, EISEN_BEST_MODEL_LOSS, EISEN_BEST_MODEL_METRIC


class PatienceHook:
    """
    Keeps track of the number of epochs after the best loss or metric is reached in the workflow,
    allowing for early stopping based on patience.  In the example below, early stopping is attached
    to the validation workflow, such that when validation loss does not improve for 6 epochs, the
    training loop will terminate.
    .. code-block:: python
        from eisen.utils.hooks import PatienceHook
        train_workflow = Training(model, train_data_loader, [loss])
        val_workflow = Validation(model, val_data_loader, [loss])
        early_stopping = PatienceHook(val_workflow.id, patience=6, select_best_loss=True)
        i = 0
        while not early_stopping.is_stop():
            train_workflow.run()
            val_workflow.run()
            i += 1
        print("stopped after %s epochs" % i)
    """

    def __init__(self, workflow_id, patience, select_best_loss=True):
        """
        :param workflow_id: the ID of the workflow that should be tracked by this hook
        :type workflow_id: UUID
        :param patience: number of epochs to run without reaching a new best loss or metric
        :type patience: int
        :param select_best_loss: whether the criterion for saving the model should be best loss or best metric
        :type select_best_loss: bool
        """
        if select_best_loss:
            dispatcher.connect(
                self._reset_on_best, signal=EISEN_BEST_MODEL_LOSS, sender=workflow_id
            )
        else:
            dispatcher.connect(
                self._reset_on_best, signal=EISEN_BEST_MODEL_METRIC, sender=workflow_id
            )
        dispatcher.connect(
            self.increment_patience, signal=EISEN_END_EPOCH_EVENT, sender=workflow_id
        )

        self.patience = patience

        self.patience_counter = 0

    def reset(self):
        """resets the patience counter manually, for example for using a learning rate scheduler"""
        self.patience_counter = 0

    def _reset_on_best(self, message):
        self.patience_counter = -1  # end epoch triggers after best loss/metric

    def increment_patience(self, message):
        self.patience_counter += 1

    def is_stop(self):
        """returns true after loss/metric has not improved for N epochs.
        :return: workflow should halt
        :rtype: bool
        """
        return self.patience_counter >= self.patience


class LoggingHook:
    """
    Logging object aiming at printing on the console the progress of model training/validation/testing.
    This logger uses an event based system. The training, validation and test workflows emit events such as
    EISEN_END_BATCH_EVENT and EISEN_END_EPOCH_EVENT which are picked up by this object and handled.

    Once the user instantiates such object, the workflow corresponding to the ID passes as argument will be
    tracked and the results of the workflow in terms of losses and metrics will be printed on the console

    .. code-block:: python

            from eisen.utils.logging import LoggingHook

            workflow = # Eg. An instance of Training workflow

            logger = LoggingHook(workflow.id, 'Training', '/artifacts/dir')

    """

    def __init__(self, workflow_id, phase, artifacts_dir):
        """
        :param workflow_id: string containing the workflow id of the workflow being monitored (workflow_instance.id)
        :type workflow_id: UUID
        :param phase: string containing the name of the phase (training, testing, ...) of the workflow monitored
        :type phase: str
        :param artifacts_dir: The path of the directory where the artifacts of the workflow are stored
        :type artifacts_dir: str

        .. code-block:: python

            from eisen.utils.logging import LoggingHook

            workflow = # Eg. An instance of Training workflow

            logger = LoggingHook(
                workflow_id=workflow.id,
                phase='Training',
                artifacts_dir='/artifacts/dir'
            )

        <json>
        []
        </json>
        """

        dispatcher.connect(
            self.end_epoch, signal=EISEN_END_EPOCH_EVENT, sender=workflow_id
        )

        self.table = PrettyTable()
        self.phase = phase
        self.workflow_id = workflow_id
        self.artifacts_dir = artifacts_dir

    def end_epoch(self, message):
        all_losses = []
        all_losses_names = []
        for dct in message["losses"]:
            for key in dct.keys():
                all_losses_names.append(key)
                all_losses.append(np.mean(dct[key]))

        all_metrics = []
        all_metrics_names = []
        for dct in message["metrics"]:
            for key in dct.keys():
                all_metrics_names.append(key)
                all_metrics.append(dct[key][-1])

        self.table.field_names = (
            ["Phase"]
            + [str(k) + " (L)" for k in all_losses_names]
            + [str(k) + " (M)" for k in all_metrics_names]
        )

        self.table.add_row(
            ["{} - Epoch {}".format(self.phase, message["epoch"])]
            + [str(loss) for loss in all_losses]
            + [str(metric) for metric in all_metrics]
        )

        print(self.table)
