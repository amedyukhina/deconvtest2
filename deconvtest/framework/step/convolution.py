from ...framework.step.combine import Combine


class Convolution(Combine):
    """
    Transform module
    """

    def __init__(self, method: str = 'convolve',
                 parameters: dict = None,
                 parent_name: str = 'deconvtest.modules.convolution'):
        super(Convolution, self).__init__(method=method,
                                          parameters=parameters,
                                          parent_name=parent_name)
