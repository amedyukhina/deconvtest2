from ...framework.module.image import Image


class GroundTruth(Image):
    """
    Ground truth module
    """

    def __init__(self, method, parameters: dict = None):
        super(GroundTruth, self).__init__(method=method,
                                          parameters=parameters,
                                          parent_name='deconvtest2.modules.ground_truth')
