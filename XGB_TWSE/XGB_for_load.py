import xgboost as xgb
from data_load import *
import time
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
import pickle
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import csv
from db_ORM import DB_modify as DB
import matplotlib.pyplot as plt
from sklearn.metrics import auc

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/release/bin'
l_name = ['high_change_class', 'low_change_class', 'close_change_class', 'close_change_class3']
model_save_path = './XGB_model/0_3'
fn = glob.glob("./Drive/Spliter/class_training/*")
shuff = True


# def data_l(stock_id, target,csv_floder_path,preds:bool=False):
#     dic = csv_floder_path
#     result = read_data(stock_id, data_main_floder=dic, shuff=shuff)
#     # drop_list = ['close_diff_{}'.format(i)if i > 0 else 'close_diff' for i in range(10)]
#     # drop_list.extend(['low_diff_{}'.format(i)if i > 0 else 'low_diff' for i in range(10)])
#     # drop_list.extend(['high_diff_{}'.format(i)if i > 0 else 'high_diff' for i in range(10)])
#     drop_list = ["date"]
#     label_name = l_name
#     feature_train, train_labels, feature_test, test_labels = train_test(result, -1, None, 900, None,
#                                                                         drop_list,
#                                                                         label_name,
#                                                                         use_gaussian=False,
#                                                                         label_num=[],
#                                                                         to_cat=False,
#                                                                         preds=preds)
#
#     ret = {str(i): [m, test_labels[i]] for i, m in enumerate(train_labels)}
#     result = ret[target]
#     return feature_train, result[0], feature_test, result[1]


def ceate_feature_map(features, path=None):
    path = path if path is not None else 'xgb.fmap'
    outfile = open("{}.fmap".format(path), 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def draw_tree(xlf, fmp=None):
    xgb.plot_tree(xlf, fmp if fmp is not None else None)


def fig_fixing(xlf, fmp=None, save: bool = True):
    fig, ax = pyplot.subplots()
    fig.set_size_inches(100, 100)
    if fmp is not None:
        xgb.plot_tree(xlf, ax=ax, fmap="{}.fmap".format(fmp))
    else:
        xgb.plot_tree(xlf, ax=ax)
    # print(fmp)
    if save: fig.savefig("{}.png".format(fmp))


def save_model(xlf, xmp, type_='joblib'):
    def jobl():
        xlf.save_model('{}.model'.format(xmp))
        joblib.dump(xlf, '{}.model'.format(xmp))

    def pickl():
        pickle.dump(xlf, open('{}.model'.format(xmp), "wb"))

    def xgb_sa():
        xlf.save_model('{}.model'.format(xmp))
        xlf.dump_model('{}.model'.format(xmp))

    if type_ is 'joblib': jobl()
    if type_ is 'pickle': pickl()
    if type_ is 'xgb': xgb_sa()


def load_model(xlf, xmp, type_='joblib'):
    def jobl():
        path = "{}.model".format(xmp) if 'model' not in xmp else xmp
        print(xmp)
        xlf = joblib.load('{}'.format(path))
        return xlf

    def pickl():
        path = "{}.model".format(xmp) if 'model' not in xmp else xmp
        xlf = pickle.load(open('{}'.format(path), "rb"))
        return xlf

    if type_ is 'joblib': xlf = jobl()
    if type_ is 'pickle': xlf = pickl()

    return xlf


def get_loop_target(target):
    stock_target = [i.split('\\')[-1].split(' ')[0] if ' ' in i.split('\\')[-1] else None for i in target]
    [stock_target.remove(None) if i is None else i for i in stock_target.copy()]
    return stock_target


def check_path(path):
    if not os.path.exists(path): os.mkdir(path)


def write_csv(data, f_p, ):
    f = open(f_p, 'w')
    sp = data.shape
    if len(sp) > 1:
        for i in data:
            for it in i:
                f.write(str(it))
                f.write(',')
            f.write('\n')
    else:
        for it in data:
            f.write(str(it))
            f.write('\n')
    f.close()


def XGB_loader(data_fn=None, model_fn=None, only_X=False, pred=False, LR_train=False, write_mode=True):
    commit = DB().insert_daily_trend
    # if model_fn is None: model_fn = glob.glob('./XGB_model/0_3/*')
    # if data_fn is None: data_fn = glob.glob('./Drive/Spliter/class_training/*')
    if model_fn is None: model_fn = glob.glob('./0_3/*')
    if data_fn is None: data_fn = glob.glob('../TWSE_crawler/x_input/*')
    target_dict = {
        'heigh': '0',
        'low': '1',
        'close': '2',
        'trend': '3'
    }

    def model_loader(model_fn):
        model_fn = [i.replace('\\', '/') for i in model_fn]
        result = {}
        for i in model_fn:
            temp_result = [it.replace('\\', '/') if 'model' in it.replace('\\', '/').split('/')[-1] else None for it in
                           glob.glob("{}/*".format(i))]
            [temp_result.remove(None) if i is None else i for i in temp_result.copy()]
            temp_result = {i.split('target_')[-1].split('_')[0]: i for i in temp_result}
            if temp_result: result.update({i.split('/')[-1]: temp_result})

        return result

    def defult_xgb():
        xlf = xgb.XGBClassifier(booster="gbtree",
                                silent=True,
                                nthread=-1,
                                scale_pos_weight=0,
                                n_estimators=5000,
                                max_depth=200,
                                min_child_weight=1,
                                subsample=0.85,
                                colsample_bytree=0.7,
                                learning_rate=0.001,
                                objective='multi:softmax',
                                reg_alpha=1,
                                reg_lambda=1,
                                gamma=0)
        return xlf

    def With_LR(all_model, data_fn, only_X, pred, write_mode):
        # stock_ids = list(all_model.keys())
        stock_ids = ['2303', '2330', '2344', '2329', '2369', '3530']
        result = {}
        for i in stock_ids:
            temp_result = {}
            for it in list(all_model[i].keys()):
                xlf = defult_xgb()

                feature_train, label_train, feature_test, label_test = data_l(target=target_dict[it],
                                                                              stock_id=str(i),
                                                                              csv_floder_path=data_fn,
                                                                              preds=pred, label_name=l_name,
                                                                              to_cat=False, only_X=False)
                model = load_model(xlf, xmp=all_model[i][it])
                X_train_leaves = model.apply(feature_train)
                X_test_leaves = model.apply(feature_test)

                train_rows = X_train_leaves.shape[0]
                X_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)

                xp = str(all_model[i][it]).split('/')[:-1]
                xp = '/'.join(xp)
                if write_mode:
                    Y_data = np.concatenate((label_train, label_test), axis=0)
                    x_f_p = "{}/stock_{}_target_{}_x_file.csv".format(xp, i, it)
                    y_f_p = "{}/stock_{}_target_{}_y_file.csv".format(xp, i, it)
                    write_csv(X_leaves, x_f_p)
                    write_csv(Y_data, y_f_p)
                    print(X_leaves.shape)
                    print(Y_data.shape)
                else:
                    # choice LR
                    lr = LogisticRegression()
                    xgbenc = OneHotEncoder()
                    X_leaves = X_leaves.astype(np.int32)
                    # one-hot encoding
                    X_trans = xgbenc.fit_transform(X_leaves)
                    X_train_ext = hstack([X_trans[:train_rows, :], feature_train])
                    # train
                    lr.fit(X_train_ext, label_train)

                    if only_X:
                        feature_train, feature_test = data_l(target=target_dict[it],
                                                             stock_id=str(i),
                                                             csv_floder_path=data_fn,
                                                             preds=pred, label_name=l_name,
                                                             to_cat=False, only_X=only_X)
                    X_leaves = X_leaves.astype(np.int32)
                    X_trans = xgbenc.fit_transform(X_leaves)
                    X_test_ext = hstack([X_trans[train_rows:, :], feature_test])
                    y_pred_xgblr2 = lr.predict_proba(X_test_ext)
                    preds = lr.predict(X_test_ext)
                    if pred:
                        if it == 'heigh': it = 'high'
                        temp_result.update({it: preds[-1]})
                    else:

                        data = pd.DataFrame({'label': label_test, 'pred': preds})
                        print(pd.crosstab([data.label], [data.pred]))
                        xgb_lr_auc2 = roc_auc_score(pd.get_dummies(label_test), y_pred_xgblr2)
                        print('LR AUC: %.5f' % xgb_lr_auc2)
                        fpr, tpr, thresholds = roc_curve(label_test, preds)
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            result.update({i: temp_result})
        return result

    def model_preds(all_model, data_fn, only_X, pred):
        stock_ids = list(all_model.keys())
        result = {}
        for i in stock_ids:
            temp_result = {}
            for it in list(all_model[i].keys()):
                xlf = defult_xgb()
                if only_X:
                    feature_train, feature_test = data_l(target=target_dict[it],
                                                         stock_id=str(i),
                                                         csv_floder_path=data_fn,
                                                         preds=pred, label_name=l_name,
                                                         to_cat=False, only_X=only_X)
                    model = load_model(xlf, xmp=all_model[i][it])
                    preds = model.predict(feature_test)
                    print(preds)
                    if it == 'heigh': it = 'high'
                    temp_result.update({it: preds[-1]})
                else:
                    feature_train, trend_label_train, feature_test, label_test = data_l(target=target_dict[it],
                                                                                        stock_id=str(i),
                                                                                        csv_floder_path=data_fn,
                                                                                        preds=pred, label_name=l_name,
                                                                                        to_cat=False, only_X=only_X)
                    model = load_model(xlf, xmp=all_model[i][it])
                    preds = model.predict(feature_test)
                    if it == 'heigh': it = 'high'
                    temp_result.update({it: preds[-1]})
                    data = pd.DataFrame({'label': label_test, 'pred': preds})
                    print('{}:{}'.format(i, it))
                    print(pd.crosstab([data.label], [data.pred]))
                    input('next?')
            result.update({i: temp_result})
        return result

    all_model = model_loader(model_fn)
    if LR_train:
        result = With_LR(all_model, data_fn, only_X, pred, write_mode)
    else:
        result = model_preds(all_model, data_fn, only_X, pred)
    commit(result)
    return result


if __name__ == "__main__":
    result = XGB_loader(data_fn=None, model_fn=None, only_X=False, pred=True, LR_train=True, write_mode=False)
    print(result)
