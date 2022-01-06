from deconvtest.framework.module.transform import Transform


class Training(Transform):
    """
    Training module
    """

    def __init__(self, method: str = None, parameters: dict = None,
                 parent_name: str = 'deconvtest.methods.training'):
        super(Training, self).__init__(method=method,
                                       parameters=parameters,
                                       parent_name=parent_name)
        self.n_inputs = 1
        self.n_outputs = 1
        self.type_input = 'data'
        self.type_output = 'model'
