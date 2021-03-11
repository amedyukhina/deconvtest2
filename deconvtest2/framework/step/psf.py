from deconvtest2.framework.module.image import Image


class PSF(Image):
    """
    PSF module
    """

    def __init__(self, method, parameters: dict = None):
        super(PSF, self).__init__(method=method,
                                  parameters=parameters,
                                  parent_name='deconvtest2.modules.psf')
