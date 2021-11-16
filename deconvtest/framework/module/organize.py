from deconvtest.framework.module.align import Align


class Organize(Align):
    """
    Transform module
    """

    def __init__(self, method: str = None,
                 parameters: dict = None,
                 parent_name: str = 'deconvtest.methods.organize'):
        super(Organize, self).__init__(method=method,
                                       parameters=parameters,
                                       parent_name=parent_name)
        self.n_inputs = 2
        self.n_outputs = 2
        self.type_input = ['file', 'file']
        self.type_output = ['folder', 'folder']
        self.align = True
