from ..module.module import Module


class Transform(Module):
    """
    Transform module
    """

    def __init__(self, method, parameters: dict = None):
        super(Transform, self).__init__(method=method,
                                        parameters=parameters,
                                        parent_name='deconvtest2.modules.transforms')
