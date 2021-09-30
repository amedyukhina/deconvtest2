from deconvtest.framework.module_base.module import Module


class Align(Module):
    """
    Transform module
    """

    def __init__(self, method: str = None,
                 parameters: dict = None,
                 parent_name: str = 'deconvtest.methods'):
        super(Align, self).__init__(method=method,
                                    parameters=parameters,
                                    parent_name=parent_name)
        self.n_inputs = 2
        self.n_outputs = 1
        self.type_input = ['image', 'image']
        self.type_output = 'image'
        self.align = True
