from configparser import ConfigParser
cfg = ConfigParser()
cfg.read('config.cfg')
"""
Read config file and return item accordingly.
"""
def get_str(section, item):
    return cfg.get(section, item)

def get_bool(section, item):
    return cfg.getboolean(section, item)

def get_int(section, item):
    return cfg.getint(section, item)

def get_float(section, item):
    return cfg.getfloat(section, item)