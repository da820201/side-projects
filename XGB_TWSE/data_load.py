import glob
import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd

from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model

configs = {
    'fit_generator': False,
    'allow_growth': True,
    'batch_size': 50,
}

py_version = '.'.join([str(i) for i in list(sys.version_info[:3])])
tf_version = int(''.join(str(tf.__version__).split('.')[:2]))

_log = __import__('tensorflow.compat.v1', fromlist=['log'], level=0) if tf_version > 15 else __import__('tensorflow',
                                                                                                        fromlist=[
                                                                                                            'log'],
                                                                                                        level=0)
log = _log.log

fn = glob.glob("./Drive/Spliter/*")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
label_name = []


def detect_data_path(fn: glob.glob or None) -> dict:
    result = {}
    folder_path = [i.replace('\\', '/') if " " in str(i).split('\\')[-1] else None for i in fn]
    if all([True if i is None else False for i in folder_path]):
        folder_path = [i.replace('\\', '/') for i in fn]
    else:
        [folder_path.remove(None) if i is None else i for i in folder_path.copy()]
    file_name = [str(i).split('/')[-1].split('\\')[-1] if '\\' in str(i).split('/')[-1] else str(i).split('/')[-1] for i
                 in folder_path]

    for it, main_key in enumerate(file_name):
        data_file_path = [i.replace('\\', '/') for i in glob.glob(f"{folder_path[it]}/*")]
        data_file_name = [str(i).split('/')[-1] for i in data_file_path]
        temp_dict = {data_file_name[i]: m for i, m in enumerate(data_file_path)}
        temp_dict.update({'stock_id': str(folder_path[it]).split('/')[-1].split(' ')[0]})
        result.update({main_key: temp_dict})

    return result


def read_data(stock: str, use_set_remove: bool = True,
              data_main_floder: str or glob.glob or None = None,x_name=None,y_name=None,x_y_concat=True) -> pd.DataFrame:
    data_dict = detect_data_path(data_main_floder)
    df = pd.DataFrame(data_dict)

    if x_name is None: x_name = 'Tech_I.csv'
    if y_name is None: y_name = 'y_train_change_class.csv'

    x = {df[i]['stock_id']: i for i in df} if stock.isdigit() else None
    stock = x[stock] if x is not None and stock in x else stock

    x_df, y_df = [[pd.read_csv(df[i][x_name], header=0) for i in list(df.keys())],
                  [pd.read_csv(df[i][y_name], header=0) for i in
                   list(df.keys())]] if stock in [
        "all", "ALL"] else [pd.read_csv(df[stock][x_name], header=0),
                            pd.read_csv(df[stock][y_name], header=0)]

    if x_y_concat is False:
        return x_df, y_df

    x_df, y_df = [pd.concat(x_df, axis=0), pd.concat(y_df, axis=0)] if stock in ["all", "ALL"] else (x_df, y_df)


    if use_set_remove:
        x_df_c, y_df_c = list(x_df.columns), list(y_df.columns)
        same_col = set(x_df_c)
        same_col.intersection_update(y_df_c)
    else:
        same_col = []
    if len(same_col) is not 0: y_df.drop(str(same_col.pop()), axis=1)
    result = pd.concat([x_df, y_df], axis=1, join='inner')
    return result


def train_test(data: pd.DataFrame, batch_size, timestep, feature_dim, FD, drop_list: list or None = None,
               label_name: list or None = None, use_gaussian: bool = True, label_num: list = [], to_cat: bool = True,
               preds: bool = False, only_X = False):
    def de_shape(feature: np.array, batch_size=None, timestep=None, feature_dim=None, FD=None):
        shape_data = [-1 if m is None and i is 0 else m for i, m in enumerate([batch_size, timestep, feature_dim, FD])]
        [shape_data.remove(None) if i is None else i for i in shape_data.copy()]
        feature = feature.reshape(shape_data)
        return feature

    def drop_dim(data: pd.DataFrame, drop_list):
        data = data.drop(drop_list, axis=1) if isinstance(drop_list, list) else data.drop(["date"], axis=1)
        return data

    def find_labels():
        res = -1 if label_name is None else [list(data.keys()).index(i) for i in label_name]
        res.sort() if isinstance(res, list) else None
        return res[0] if isinstance(res, list) else res

    def gaussian_labels(D: pd.DataFrame, label_num: list, stage: list or None = None, deg: float = 1.0):
        stage = [0.63, 1] if stage is None else stage
        result = []

        for i, m in enumerate(label_num):
            label = list(to_categorical(pd.DataFrame(D)[i], num_classes=m)) if to_cat is True else list(
                pd.DataFrame(D)[i])

            if use_gaussian:
                res = []
                for ic in label:
                    item = list(ic)
                    _index = item.index(1)
                    item[_index] = deg
                    for it in range(1, stage[1] + 1):
                        if _index - it >= 0:
                            item[_index - it] = stage[0] ** (it)
                        if _index + it < m:
                            item[_index + it] = stage[0] ** (it)
                    res.append(item)
                result.append(list(de_shape(np.array(res), batch_size, m)))
            else:
                result.append(list(label))

        return result

    def train_test(data, row: float or None = None):
        row = round(0.85 * data.shape[0]) if row is None else row
        if preds:
            return np.array(data[:int(row)]), np.array(data[-1:])
        else:
            return np.array(data[:int(row)]), np.array(data[int(row):])

    def normal_categorical(D: pd.DataFrame, label_num: list or None = None) -> list:
        label_len = np.array(D).shape[-1]
        if label_num is None: label_num = []
        result = []
        for i in range(len(label_num), label_len):
            if to_cat:
                max_label_num = int(max(pd.DataFrame(D)[i]) + 1)
                result.append(list(to_categorical(pd.DataFrame(D)[i], num_classes=max_label_num)))
            else:
                result.append(list(pd.DataFrame(D)[i]))
        return result

    data = drop_dim(data, drop_list)
    labels_cl = find_labels()

    train, test = train_test(data)

    x_train = train[:, :labels_cl]
    label_train = train[:, labels_cl:]

    x_test = test[:, :labels_cl]
    label_test = test[:, labels_cl:]

    feature_train = de_shape(x_train, batch_size, timestep, feature_dim, FD)
    feature_test = de_shape(x_test, batch_size, timestep, feature_dim, FD)

    if only_X:
        return feature_train, feature_test

    if use_gaussian:
        # label_num= [label_num, label_num, label_num=39]
        label_train_ = gaussian_labels(label_train, label_num=label_num)
        label_test_ = gaussian_labels(label_test, label_num=label_num)
        if to_cat:
            label_test_.append(list(to_categorical(pd.DataFrame(label_test)[3], num_classes=3)))
            label_train_.append(list(to_categorical(pd.DataFrame(label_train)[3], num_classes=3)))
        else:
            label_test_.append(list(pd.DataFrame(label_test)[3]))
            label_train_.append(list(pd.DataFrame(label_train)[3]))
    else:
        label_train_ = normal_categorical(label_train, label_num)
        label_test_ = normal_categorical(label_test, label_num)

    return feature_train, [np.array(i) for i in label_train_], feature_test, [np.array(i) for i in label_test_]


def data_l(stock_id, target, csv_floder_path=None, preds: bool = False, label_name: list = [], to_cat=True, only_X=False):
    result = read_data(stock_id, data_main_floder=csv_floder_path)
    drop_list = ["date"]
    re_dict = {
        'high': '0',
        'low': '1',
        'close': '2',
        'trend': '3'
    }
    # re_dict = {m: i for i, m in enumerate(label_name)}
    if only_X:
        feature_train, feature_test = train_test(result, -1, None, 930, None,
                                                                            drop_list,
                                                                            label_name=label_name,
                                                                            use_gaussian=False,
                                                                            label_num=[],
                                                                            to_cat=to_cat, preds=preds,only_X=only_X)
        return feature_train, feature_test

    feature_train, train_labels, feature_test, test_labels = train_test(result, -1, None, 930, None,
                                                                        drop_list,
                                                                        label_name=label_name,
                                                                        use_gaussian=False,
                                                                        label_num=[],
                                                                        to_cat=to_cat, preds=preds)

    ret = {str(i): [m, test_labels[i]] for i, m in enumerate(train_labels)}
    result = ret[target]
    return feature_train, result[0], feature_test, result[1]


def odd_data_l(stock_id, target):
    dic = fn
    result = read_data(stock_id, data_main_floder=dic)
    drop_list = ["date"]
    label_name = ['high_change_class', 'low_change_class', 'close_change_class', 'close_change_class3']
    feature_train, [high_label_train, low_label_train, close_label_train, trend_label_train], feature_test, [
        high_label_test, low_label_test, close_label_test, trend_label_test] = train_test(result, -1, None, 900, None,
                                                                                          drop_list,
                                                                                          label_name,
                                                                                          use_gaussian=False,
                                                                                          label_num=[],
                                                                                          to_cat=True)
    ret = {
        'ALL': [[high_label_train, low_label_train, close_label_train, trend_label_train],
                [high_label_test, low_label_test, close_label_test, trend_label_test]],
        'high': [high_label_train, high_label_test],
        'low': [low_label_train, low_label_test],
        'close': [close_label_train, close_label_test],
        'trend': [trend_label_train, trend_label_test],
    }

    result = ret[target]

    return feature_train, result[0], feature_test, result[1]


def auto_multilabels_to_one(em_len, features: list, labels: list, output_dim: int, define_loss='mean_squared_error'):
    split_time = round(int(np.array(features[0]).shape[-1]) / em_len)

    Feature_train_ = [features[0][:, i * split_time:i * split_time + split_time] for i in range(em_len)]
    Feature_test_ = [features[1][:, i * split_time:i * split_time + split_time] for i in range(em_len)]

    Label_train = {'out_{}'.format(i): np.array(m).reshape([-1, split_time, 1]) for i, m in enumerate(Feature_train_)}
    Feature_train = {'intput_{}'.format(i): np.array(m).reshape([-1, split_time, 1]) for i, m in
                     enumerate(Feature_train_)}

    Label_test = {'out_{}'.format(i): np.array(m).reshape([-1, split_time, 1]) for i, m in enumerate(Feature_test_)}
    Feature_test = {'intput_{}'.format(i): np.array(m).reshape([-1, split_time, 1]) for i, m in
                    enumerate(Feature_test_)}

    Loss = {'out_{}'.format(i): define_loss for i in range(em_len)}
    Loss_weights = {'out_{}'.format(i): 1. for i in range(em_len)}
    # --------------------------------------------------------------------------------------------------------------#
    ouput_name = 'final_output'
    Label_train.update({ouput_name: np.array(labels[0].reshape([-1, output_dim]))})
    Label_test.update({ouput_name: np.array(labels[1].reshape([-1, output_dim]))})
    Loss.update({ouput_name: 'categorical_crossentropy'})
    Loss_weights.update({ouput_name: 1.})

    # --------------------------------------------------------------------------------------------------------------#
    return Feature_train, Label_train, Feature_test, Label_test, Loss, Loss_weights


Xgb_args = {'booster': "gbtree",
            'silent': True,
            'nthread': -1,
            'scale_pos_weight': 0,
            'n_estimators': 5,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.85,
            'colsample_bytree': 0.7,
            'learning_rate': 0.01,
            'objective': 'multi:softmax',
            'reg_alpha': 1,
            'reg_lambda': 1,
            'gamma': 0,
            }

dic = glob.glob('./Drive/Spliter/training_raw/*')
