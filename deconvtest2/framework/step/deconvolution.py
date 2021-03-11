from deconvtest2.framework.module.module import Module


class Deconvolution(Module):
    """
    Deconvolution module
    """

    def __init__(self, method, parameters: dict = None):
        super(Deconvolution, self).__init__(method=method,
                                            parameters=parameters,
                                            parent_name='deconvtest2.modules.deconvolution')
