from ..module.module import Module


class Accuracy(Module):
    """
    Accuracy module
    """

    def __init__(self, method, parameters: dict = None):
        super(Accuracy, self).__init__(method=method,
                                       parameters=parameters,
                                       parent_name='deconvtest2.modules.accuracy')
