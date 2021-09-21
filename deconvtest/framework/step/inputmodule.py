from ..module.module import Module


class InputModule(Module):
    """
    Image module (ground truth or PSF)
    """

    def __init__(self, method: str = None, parameters: dict = None, parent_name: str = 'deconvtest.modules'):
        super(InputModule, self).__init__(method=method,
                                          parameters=parameters,
                                          parent_name=parent_name)
        self.n_inputs = 0
        self.n_outputs = 1
        self.type_input = None
        self.type_output = 'image'
