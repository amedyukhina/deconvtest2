from ...framework.module.module import Module


class Evaluation(Module):
    """
    Evaluation module
    """

    def __init__(self, method: str = None, parameters: dict = None):
        super(Evaluation, self).__init__(method=method,
                                         parameters=parameters,
                                         parent_name='deconvtest.modules.evaluation')
        self.n_inputs = 2
        self.n_outputs = 0
