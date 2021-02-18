from ..module.module import Module


class DeconvolutionModule(Module):
    """
    Deconvolution module
    """

    def __init__(self, method, parameters: dict = None):
        super(DeconvolutionModule, self).__init__(method=method,
                                                  parameters=parameters,
                                                  parent_name='deconvtest2.modules.deconvolution')
