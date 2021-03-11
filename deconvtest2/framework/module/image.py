from ..module.module import Module


class Image(Module):
    """
    Image module (ground truth or PSF)
    """

    def __init__(self, method, parameters: dict = None, parent_name: str = 'deconvtest2.modules'):
        super(Image, self).__init__(method=method,
                                    parameters=parameters,
                                    parent_name=parent_name)
