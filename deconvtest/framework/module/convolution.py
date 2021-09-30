from deconvtest.framework.module.combine import Combine


class Convolution(Combine):
    """
    Transform module
    """

    def __init__(self, method: str = 'convolve',
                 parameters: dict = None,
                 parent_name: str = 'deconvtest.methods.convolution'):
        super(Convolution, self).__init__(method=method,
                                          parameters=parameters,
                                          parent_name=parent_name)
