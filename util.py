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
        super(EisenModuleWrapper, self).__init__()

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