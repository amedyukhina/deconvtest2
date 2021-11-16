class Parameter:
    """
    Abstract module_base class
    """

    def __init__(self, name, default_value=None, value=None, optional=True, parameter_type=None, doc=''):
        """

        Parameters
        ----------
        name : str
            Name of the module_base
        default_value : type specified by `parameter_type`, optional
            Default value of the module_base.
            Default is None.
        value : type specified by `parameter_type`, optional
            Parameter value.
            Default is None.
        optional : bool, optional
            If True, the module_base is optional.
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
        return rf"Parameter: name={self.name}, value={self.value}, " \
               rf"type={self.type}, optional={self.optional}, default_value={self.default_value}"
