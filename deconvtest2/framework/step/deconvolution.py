from ...framework.step.transform import Transform


class Deconvolution(Transform):
    """
    Deconvolution module
    """

    def __init__(self, method: str = None, parameters: dict = None):
        super(Deconvolution, self).__init__(method=method,
                                            parameters=parameters,
                                            parent_name='deconvtest2.modules.deconvolution')
