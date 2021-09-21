from ...framework.step.align import Align


class Evaluation(Align):
    """
    Evaluation module
    """

    def __init__(self, method: str = None, parameters: dict = None):
        super(Evaluation, self).__init__(method=method,
                                         parameters=parameters,
                                         parent_name='deconvtest.modules.evaluation')
        self.type_output = 'number'

