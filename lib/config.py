import yaml


class Config:
    def __init__(self, a_dict: dict):
        def cfg_value(obj):
            # is it a dict?
            if isinstance(obj, dict):
                # recursion
                return Config(obj)
            else:
                # everything else - including lists or tuples - stay as-is
                return obj
        for key, val in a_dict.items():
            self.__setattr__(key, cfg_value(val))

    def __str__(self) -> str:
        return self.__dict__.__str__()

    @staticmethod
    def load_file(filename: str):
        with open(filename, 'r') as cfg:
            as_dict = yaml.safe_load(cfg)
        return Config(as_dict)
