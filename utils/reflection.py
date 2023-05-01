import importlib


def get_class(path: str):
    module_name, name = path.rsplit('.', 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, name)


def get_function(path: str):
    module_name, name = path.rsplit('.', 1)
    print(module_name, name)
    mod = importlib.import_module(module_name)
    return getattr(mod, name)


def get_config(path: str):
    module_name, name = path.rsplit('.', 1)
    mod = importlib.import_module(module_name)
    print(f"modul_name : {name}, \n mod : {mod}")

    return getattr(mod, name)
