import argparse
import json
import yaml
import copy


def parse_json_args(args, json_path, encoding='utf-8'):
    def trans_dict_key(k):
        return k.replace('-', '_')
    def trans_dict_value(v):
        if isinstance(v, dict):
            new_d = dict()
            for n_k in v.keys():
                new_d[trans_dict_key(n_k)] = trans_dict_value(v[n_k])
            return new_d
        else:
            return v
    with open(json_path, 'r', encoding=encoding) as f:
        json_dict = json.load(f)
    trans_json_dict = dict()
    for k in json_dict:
        trans_json_dict[trans_dict_key(k)] = trans_dict_value(json_dict[k])
    ret_dict = copy.deepcopy(args.__dict__)
    for k in trans_json_dict.keys():
        ret_dict[k] = trans_json_dict[k]
    return argparse.Namespace(**ret_dict)


def parse_yaml_args(args, yaml_path, encoding='utf-8'):
    def trans_dict_key(k):
        return k.replace('-', '_')
    def trans_dict_value(v):
        if isinstance(v, dict):
            new_d = dict()
            for n_k in v.keys():
                new_d[trans_dict_key(n_k)] = trans_dict_value(v[n_k])
            return new_d
        else:
            return v
    with open(yaml_path, 'r', encoding=encoding) as f:
        yaml_dict = yaml.full_load(f)
    trans_yaml_dict = dict()
    for k in yaml_dict:
        trans_yaml_dict[trans_dict_key(k)] = trans_dict_value(yaml_dict[k])
    ret_dict = copy.deepcopy(args.__dict__)
    for k in trans_yaml_dict.keys():
        ret_dict[k] = trans_yaml_dict[k]
    return argparse.Namespace(**ret_dict)
