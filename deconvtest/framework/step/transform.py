from ...framework.module.module import Module


class Transform(Module):
    """
    Transform module
    """

    def __init__(self, method: str = None, parameters: dict = None,
                 parent_name: str = 'deconvtest.modules.transforms'):
        super(Transform, self).__init__(method=method,
                                        parameters=parameters,
                                        parent_name=parent_name)
        self.n_inputs = 1
        self.n_outputs = 1
        self.type_input = 'image'
        self.type_output = 'image'
