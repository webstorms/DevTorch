import torch
import torch.nn as nn


class DevModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._weight_initializers = {}

    @property
    def hyperparams(self):
        hyperparams = {
            "name": self.__class__.__name__,
            "weights": self._weight_initializers,
        }

        return hyperparams

    @property
    def device(self):
        return next(self.parameters()).device.type

    def _get_variable_name(self, src_param):
        for name, param in self.named_parameters():
            if torch.equal(param, src_param):
                return name

        raise ValueError(f"Tensor needs to be part of model.")

    def init_weight(self, weight, init_type, **kwargs):
        """
        Initialize the weights of a tensor using specified initialization methods.

        This function selects an initialization function based on `init_type` and applies
        it to the `weight` tensor. It keeps track of initializations to prevent re-initializing
        a weight that has already been initialized.

        Parameters:
        weight (torch.Tensor): The weight tensor to be initialized.
        init_type (str): The type of initializer to use. Supported initializers include:
            - "constant": Initializes the tensor with a constant value. Requires 'val' as a keyword argument.
            - "uniform": Initializes the tensor with values uniformly drawn from [lower, upper].
                          Requires 'a' and 'b' as keyword arguments for bounds.
            - "glorot_uniform": Uses Xavier uniform initializer. No additional parameters needed.
            - "normal": Initializes the tensor with values drawn from a normal distribution.
                        Requires 'mean' and 'std' as keyword arguments for distribution parameters.
            - "glorot_normal": Uses Xavier normal initializer. No additional parameters needed.
            - "identity": Initializes the tensor as an identity matrix, scaled by 'val' (default is 1).
                          Only applicable for 2D square matrices.
        **kwargs: Arbitrary keyword arguments specific to the type of initializer. Common arguments include:
            - 'val' for constant initialization
            - 'a' and 'b' for uniform bounds
            - 'mean' and 'std' for normal distribution parameters

        Raises:
        ValueError: If `weight_name` has already been initialized or if the specified `init_type` is not supported.

        Example:
        >>> init_weight(my_tensor, 'uniform', a=0, b=1)
        Initializes `my_tensor` with uniform distribution parameters between 0 and 1.
        """
        weight_name = self._get_variable_name(weight)

        initializer_functions = {
            "constant": lambda: nn.init.constant_(weight, **kwargs),
            "uniform": lambda: nn.init.uniform_(weight, **kwargs),
            "glorot_uniform": lambda: nn.init.xavier_uniform_(weight),
            "normal": lambda: nn.init.normal_(weight, **kwargs),
            "glorot_normal": lambda: nn.init.xavier_normal_(weight),
            "identity": lambda: (
                nn.init.eye_(weight),
                weight.data.mul_(kwargs.get("val", 1)),
            ),
        }

        if weight_name in self._weight_initializers:
            raise ValueError(f"{weight_name} weights already initialized.")

        if init_type in initializer_functions:
            initializer = initializer_functions[init_type]
            initializer()  # Initialize the tensor
            self._weight_initializers[weight_name] = {
                "init_type": init_type,
                "dtype": str(weight.dtype),
                "params": kwargs,
            }
        else:
            raise ValueError(f"{init_type} initializer not implemented.")
