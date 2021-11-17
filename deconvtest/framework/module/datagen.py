from deconvtest.framework.module.transform import Transform


class DataGen(Transform):
    """
    DataGen module
    """

    def __init__(self, method: str = None, parameters: dict = None,
                 parent_name: str = 'deconvtest.methods.datagen'):
        super(DataGen, self).__init__(method=method,
                                        parameters=parameters,
                                        parent_name=parent_name)
        self.n_inputs = 1
        self.n_outputs = 1
        self.type_input = 'folder'
        self.type_output = 'data'
        self.wait_complete = True
        self.add_id = False
