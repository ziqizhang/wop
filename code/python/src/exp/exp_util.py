import os
import pandas as pd
import numpy


def describe_task(properties, overwrite_params, setting_file):
    desc = 'setting_file=' + os.path.splitext(os.path.basename(setting_file))[0]
    desc += '|embedding='
    desc += os.path.splitext(os.path.basename(
        load_setting('embedding_file', properties, overwrite_params)))[0]
    if 'training_text_data' in properties.keys():
        desc += '|training_text_data='
        desc += os.path.splitext(os.path.basename(
            load_setting('training_text_data', properties, overwrite_params)))[0]
    return desc


# the program can overwrite parameters defined in setting files. for example, if you want to overwrite
# the embedding file, you can include this as an overwrite param in the command line, but specifying
# [embedding_file= ...] where 'embedding_file' must match the parameter name. Note that this will apply
# to ALL settings
def parse_overwrite_params(argv):
    params = {}
    for a in argv:
        if "=" in a:
            values = a.split("=")
            params[values[0]] = values[1]
    return params


def load_setting(param_name, properties: {}, overwrite_params: {} = None):
    if overwrite_params is not None and param_name in overwrite_params.keys():
        return overwrite_params[param_name]
    else:
        return properties[param_name]

def load_and_merge_train_test_data(train_data_file, test_data_file, delimiter=";"):
    train = pd.read_csv(train_data_file, header=0, delimiter=delimiter, quoting=0, encoding="utf-8",
                        ).fillna('').as_matrix()
    train.astype(str)

    test = pd.read_csv(test_data_file, header=0, delimiter=delimiter, quoting=0, encoding="utf-8",
                       ).fillna('').as_matrix()
    test.astype(str)

    return numpy.concatenate((train, test), axis=0), len(train), len(test)

def load_properties(filepath, sep='=', comment_char='#'):
    """
    Read the file passed as parameter as a properties file.
    """
    props = {}
    with open(filepath, "rt") as f:
        for line in f:
            l = line.strip()
            if l and not l.startswith(comment_char):
                key_value = l.split(sep)
                key = key_value[0].strip()
                value = sep.join(key_value[1:]).strip().strip('"')
                props[key] = value
    return props