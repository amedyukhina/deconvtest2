from ...framework.module.inputmodule import InputModule


class GroundTruth(InputModule):
    """
    Ground truth module
    """

    def __init__(self, method: str = None, parameters: dict = None):
        super(GroundTruth, self).__init__(method=method,
                                          parameters=parameters,
                                          parent_name='deconvtest.modules.ground_truth')
