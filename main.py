import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from feature_engineering.data_engineering import data_engineer_benchmark, span_data_2d, span_data_3d
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
# from methods.gtan_GCNet_psa.gtan_main import load_gtan_data, gtan_main
from methods.gtan_GCNet_psa.gtan_main_hid_dim import load_gtan_data, gtan_main_hid_dim
from methods.gtan_GCNet_psa.gtan_main_n_layers import gtan_main_n_layers
from methods.gtan_GCNet_psa.gtan_main_batch_size import gtan_main_batch_size
from methods.gtan_psa_multi.gtan_main_psa_multi import gtan_main


logger = logging.getLogger(__name__)
# sys.path.append("..")


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", default='gtan')  # specify which method to use
    # parser.add_argument("--method", default=str)  # specify which method to use
    method = vars(parser.parse_args())['method']  # dict

    # if method in ['']:
    #     yaml_file = "config/base_cfg.yaml"
    if method in ['mcnn']:
        yaml_file = "config/mcnn_cfg.yaml"
    elif method in ['stan']:
        yaml_file = "config/stan_cfg.yaml"
    elif method in ['stan_2d']:
        yaml_file = "config/stan_2d_cfg.yaml"
    elif method in ['stagn']:
        yaml_file = "config/stagn_cfg.yaml"
    elif method in ['gtan']:
        yaml_file = "config/gtan_cfg.yaml"
    elif method in ['rgtan']:
        yaml_file = "config/rgtan_cfg.yaml"
    else:
        raise NotImplementedError("Unsupported method.")

    # config = Config().get_config()
    with open(yaml_file, 'r', encoding='utf-8') as file:
        args = yaml.safe_load(file)
    args['method'] = method
    # with open(config_file_path, 'r', encoding='utf-8') as file:
    #         args = yaml.safe_load(file)
    return args


def base_load_data(args: dict):
    # load S-FFSD dataset for base models
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args['test_size']
    method = args['method']
    # for ICONIP16 & AAAI20
    if args['method'] == 'stan':
        if os.path.exists("data/tel_3d.npy"):
            return
        features, labels = span_data_3d(feat_df)
    else:
        if os.path.exists("data/tel_2d.npy"):
            return
        features, labels = span_data_2d(feat_df)
    num_trans = len(feat_df)
    trf, tef, trl, tel = train_test_split(
        features, labels, train_size=train_size, stratify=labels, shuffle=True)
    trf_file, tef_file, trl_file, tel_file = args['trainfeature'], args[
        'testfeature'], args['trainlabel'], args['testlabel']
    np.save(trf_file, trf)
    np.save(tef_file, tef)
    np.save(trl_file, trl)
    np.save(tel_file, tel)
    return


def main(args):

    if args['method'] == 'gtan':
        feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(
            args['dataset'], args['test_size'])  # feat_data = ={DataFrame:(77881,126)是节点特征，labels = {Tensor:(77881,)}是标签，train_idx = {list: 62304}是训练集索引，test_idx = {list: 15577}是测试集索引，g是图，cat_features = {list: 3} ['Target', 'Location', 'Type' ]
        gtan_main(
            feat_data, g, train_idx, test_idx, labels, args, cat_features)

    else:
        raise NotImplementedError("Unsupported method. ")


if __name__ == "__main__":
    main(parse_args())
