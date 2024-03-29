from deconvtest.framework.module.transform import Transform


## TODO: create an intermediate class "External transform"

class Deconvolution(Transform):
    """
    Deconvolution module
    """

    def __init__(self, method: str = None, parameters: dict = None):
        super(Deconvolution, self).__init__(method=method,
                                            parameters=parameters,
                                            parent_name='deconvtest.methods.deconvolution')
