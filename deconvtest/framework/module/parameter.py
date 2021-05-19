class Parameter:
    """
    Abstract parameter class
    """

    def __init__(self, name, default_value=None, value=None, optional=True, parameter_type=None, doc=''):
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
        optional : bool, optional
            If True, the parameter is optional.
            Default is True
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
        self.optional = optional
        self.__doc__ = doc + '\n----\nvariables:\n- name\n- default_value\n- value\n- type'

    def __repr__(self):
        return "Parameter: name={}, value={}, type={}, optional={}, default_value={}".format(self.name,
                                                                                             self.value,
                                                                                             self.type,
                                                                                             self.optional,
                                                                                             self.default_value)
