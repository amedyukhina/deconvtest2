from deconvtest.framework.module.combine import Combine


class Restoration(Combine):
    """
    Transform module
    """

    def __init__(self, method: str = None,
                 parameters: dict = None,
                 parent_name: str = 'deconvtest.methods.restoration'):
        super(Restoration, self).__init__(method=method,
                                          parameters=parameters,
                                          parent_name=parent_name)
        self.type_input = ['image', 'model']
