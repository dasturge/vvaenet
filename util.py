from copy import deepcopy

import numpy as np
from torch.nn import Module


class EisenLossWrapper(Module):
    """
    This object implements a wrapper allowing standard PyTorch Modules (Eg. those implemented in torchvision)
    to be used within Eisen.

    Modules in Eisen accept positional and named arguments in the forward() method. They return values or a tuple of
    values.

    Eisen workflows make use of dictionaries. That is, data batches are represented as dictionaries and directly fed
    into modules using the **kwargs mechanism provided by Python.

    This wrapper causes standard Modules to behave as prescribed by Eisen. Wrapped modules accept as input a dictionary
    of keyword arguments with arbitrary (user defined) keys. They return as output a dictionary of keyword values
    with arbitrary (user defined) keys.

    .. code-block:: python

        # We import the Module we want to wrap. In this case we import from torchvision

        from torchvision.models import resnet18

        # We can then instantiate an object of class EisenModuleWrapper and instantiate the Module we want to
        # wrap as well as the fields of the data dictionary that will interpreted as input, and the fields
        # that we desire the output to be stored at. Additional arguments for the Module itself can
        # be passed as named arguments.

        module = resnet18(pretrained=False)

        adapted_module = EisenModuleWrapper(module, ['image'], ['prediction'])

    """

    def __init__(self, module, input_names, output_names, weight=1.0):
        """
        :param module: This is a Module instance
        :type module: torch.nn.Module
        :param input_names: list of names for positional arguments of module. Must match field names in data batches
        :type input_names: list of str
        :param output_names: list of names for the outputs of the module
        :type output_names: list of str
        """
        super(EisenLossWrapper, self).__init__()

        self.input_names = input_names
        self.output_names = output_names

        self.module = module

        self.weight = weight

    def forward(self, *args, **kwargs):
        input_list = list(args)
        n_args = len(input_list)

        for key in kwargs.keys():
            if key in self.input_names[n_args:]:
                input_list.append(kwargs[key])

        outputs = self.module(*input_list)

        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)

        ret_dict = {}

        for output, output_name in zip(outputs, self.output_names):
            ret_dict[output_name] = self.weight * output

        return ret_dict


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