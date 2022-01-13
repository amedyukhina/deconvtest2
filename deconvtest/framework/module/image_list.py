from deconvtest.framework.module.inputmodule import InputModule


class ImageList(InputModule):
    """
    Ground truth module
    """

    def __init__(self, method: str = None, parameters: dict = None):
        super(ImageList, self).__init__(method=method,
                                        parameters=parameters,
                                        parent_name='deconvtest.methods.image_list')
        self.run_early = True
