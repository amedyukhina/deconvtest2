from ...framework.module.inputmodule import InputModule


class PSF(InputModule):
    """
    PSF module
    """

    def __init__(self, method: str = None, parameters: dict = None):
        super(PSF, self).__init__(method=method,
                                  parameters=parameters,
                                  parent_name='deconvtest.modules.psf')
