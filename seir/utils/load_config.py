import yaml

__all__ = ["load_yaml"]


def load_yaml(path):
    with open(path) as fin:
        config = yaml.load(fin)
    return config
