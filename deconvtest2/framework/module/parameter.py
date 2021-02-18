

class Parameter:
    """
    Abstract parameter class
    """
    def __init__(self, name, default_value=None, value=None, parameter_type=None, doc=''):
        """

        Parameters
        ----------
        name : str
            Name of the parameter
        default_value : type specified by `parameter_type`, optional
            Default value of the parameter.
            Default is None.
        value : type specified by `parameter_type`, optional
            Parameter value.
            Default is None.
        parameter_type : type or typing.Union, optional
            Parameter type.
            Default is None.
        doc : str, optional
            Documentation string.
            Default is empty string.
        """
        self.name = name
        self.default_value = default_value
        self.value = value
        self.type = parameter_type
        self.__doc__ = doc + '\n----\nvariables:\n- name\n- default_value\n- value\n- type'

