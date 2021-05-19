from ...framework.module.module import Module


class Convolution(Module):
    """
    Transform module
    """

    def __init__(self, parameters: dict = None,
                 parent_name: str = 'deconvtest.modules.convolution'):
        super(Convolution, self).__init__(method='convolve',
                                          parameters=parameters,
                                          parent_name=parent_name)
        self.n_inputs = 2
        self.n_outputs = 1
