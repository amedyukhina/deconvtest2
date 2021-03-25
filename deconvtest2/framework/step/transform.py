from ...framework.module.module import Module


class Transform(Module):
    """
    Transform module
    """

    def __init__(self, method: str = None, parameters: dict = None,
                 parent_name: str = 'deconvtest2.modules.transforms'):
        super(Transform, self).__init__(method=method,
                                        parameters=parameters,
                                        parent_name=parent_name)
        self.n_inputs = 1
        self.n_outputs = 1
