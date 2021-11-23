__all__ = ["read_dir", "split_data", "get_config", "update_config", "save_config", "arange", "get_connected_components",
    "EasyDict"]

import os
import os.path as path
import yaml
import numpy as np
import argparse


class EasyDict(object):
    def __init__(self, opt): self.opt = opt if opt else {}

    def __getattribute__(self, name):
        if name == 'opt' or name.startswith("_") or name not in self.opt:
            return object.__getattribute__(self, name)
        else: return self.opt[name]

    def __setattr__(self, name, value):
        if name == 'opt': object.__setattr__(self, name, value)
        else: self.opt[name] = value

    def __getitem__(self, name):
        return self.opt[name]
    
    def __setitem__(self, name, value):
        self.opt[name] = value

    def __contains__(self, item):
        return item in self.opt

    def __repr__(self):
        return self.opt.__repr__()

    def keys(self):
        return self.opt.keys()

    def values(self):
        return self.opt.values()

    def items(self):
        return self.opt.items()


def get_config(description, model_name, phase,
    config_dir="config/", run_dir="run/", cmd_opts=[]):
    ''' get the config for [model_name].
    '''

    # Parse command line for the run_name
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("run_name")
    if cmd_opts: arg = parser.parse_args(cmd_opts)
    else: arg = parser.parse_args()
    # arg.run_name = 'v3'
    config_file = path.join(config_dir, f"{model_name}.yaml")

    # Load default and run configs
    default = load_config(config_file)

    # Update default config by run config
    run_file = path.join(run_dir, f"{model_name}.yaml")
    if path.isfile(run_file):
        run = load_config(run_file)
        if arg.run_name in run and phase in run[arg.run_name]:
            update_config(default, run[arg.run_name][phase])

    run_dir = path.join(run_dir, model_name, arg.run_name)
    if not path.isdir(run_dir): os.makedirs(run_dir)

    # Save and return config
    save_config(default, path.join(run_dir, f"{phase}_opts.yaml"))
    default.run_dir = run_dir
    return default


def load_config(config_file, config_names=[]):
    ''' load config from file
    '''

    with open(config_file) as f:
        config = resolve_expression(yaml.load(f, Loader=yaml.FullLoader))
    
    if type(config_names) == str: return EasyDict(config[config_names])

    while len(config_names) != 0:
        config_name = config_names.pop(0)
        if config_name not in config:
            raise ValueError("Invalid config name: {}".format(config_name))
        config = config[config_name]

    return EasyDict(config)


def update_config(config, args):
    ''' rewrite default config with user input
    '''
    if args is None: return
    if hasattr(args, "__dict__"): args = args.__dict__

    for key, val in config.items():
        if key in args: config[key] = args[key] 
        elif type(val) == dict: update_config(val, args)


def save_config(config, config_file, print_opts=True):
    config_str = yaml.dump(config.opt, default_flow_style=False)
    with open(config_file, 'w') as f: f.write(config_str)
    print('================= Options =================')
    print(config_str[:-1])
    print('===========================================')


def resolve_expression(config):
    if type(config) is dict:
        new_config = {}
        for k, v in config.items():
            if type(v) is str and v.startswith("!!python"):
                v = eval(v[8:])
            elif type(v) is dict:
                v = resolve_expression(v)
            new_config[k] = v
        config = new_config
    return config


def read_dir(dir_path, predicate=None, name_only=False, recursive=False):
    IMG_EXTENTS = {".png", ".jpeg", ".jpg"}
    if type(predicate) is str:
        if predicate in {'dir', 'file', "img"}:
            predicate = {
                'dir': lambda x: path.isdir(path.join(dir_path, x)),
                'file':lambda x: path.isfile(path.join(dir_path, x)),
                "img": lambda x: path.splitext(x)[-1] in IMG_EXTENTS
            }[predicate]
        else:
            ext = predicate
            predicate = lambda x: ext in path.splitext(x)[-1]

    def read_dir_(output, dir_path, predicate, name_only, recursive):
        if not path.isdir(dir_path): return
        for f in os.listdir(dir_path):
            d = path.join(dir_path, f)
            if predicate is None or predicate(f):
                output.append(f if name_only else d)
            if recursive and path.isdir(d):
                read_dir_(output, d, predicate, name_only, recursive)

    output = []
    read_dir_(output, dir_path, predicate, name_only, recursive)
    return sorted(output)


def split_data(data, split):
    sp1 = int(split * len(data))
    sp2 = sp1 + int(np.round((1 - split) / 2 * len(data)))
    assert 0 < sp1 < len(data) and 0 < sp2 < len(data), \
        "Invalid split of data sets"
    splits = {"train": data[:sp1], "val": data[sp1:sp2], "test": data[sp2:]}
    return splits


def arange(start, stop, step):
    """ Matlab-like arange
    """
    r = np.arange(start, stop, step).tolist()
    if r[-1] + step == stop:
        r.append(stop)
    return np.array(r)


def get_connected_components(points):
    def get_neighbors(point):
        p0, p1 = point
        neighbors = [
            (p0 - 1, p1 - 1), (p0 - 1, p1), (p0 - 1, p1 + 1),
            (p0 + 1, p1 - 1), (p0 + 1, p1), (p0 + 1, p1 + 1),
            (p0, p1 - 1), (p0, p1 + 1)]
        return neighbors

    components = []
    while points:
        component = []
        unchecked = [points.pop()]
        while unchecked:
            point = unchecked.pop(0)
            neighbors = get_neighbors(point)
            for n in neighbors:
                if n in points:
                    points.remove(n)
                    unchecked.append(n)
            component.append(point)
        components.append(component)
    return components