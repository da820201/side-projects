import xgboost as xgb
from data_load import *
import time
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
import pickle
import joblib
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/release/bin'
l_name = ['high_change_class', 'low_change_class', 'close_change_class', 'close_change_class3']
model_save_path = './XGB_model/0_3'
fn = glob.glob("./Drive/Spliter/class_training/*")
shuff = True

# pd.set_option('display.max_rows', 3002)


def f1_error(preds, train):
    label = train.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    pred = [int(i >= 0.5) for i in preds]
    tp = sum([int(i == 1 and j == 1) for i, j in zip(pred, label)])
    precision = float(tp) / sum(pred)
    recall = float(tp) / sum(label)
    return 'f1-score', 2 * (precision * recall / (precision + recall))


def data_l(stock_id, target,csv_floder_path,preds:bool=False,label_name=[]):
    result = read_data(stock_id, data_main_floder=csv_floder_path)
    drop_list = ["date"]
    label_name = label_name
    feature_train, train_labels, feature_test, test_labels = train_test(result, -1, None, 930, None,
                                                                        drop_list,
                                                                        label_name,
                                                                        use_gaussian=False,
                                                                        label_num=[],
                                                                        to_cat=False,
                                                                        preds=preds)
    ret = {str(i): [m, test_labels[i]] for i, m in enumerate(train_labels)}
    result = ret[target]
    return feature_train, result[0], feature_test, result[1]


def train_mode(target: str, stock_id, csv_floder_path):
    label_name = l_name
    feature_train, label_train, feature_test, label_test = data_l(stock_id, target, csv_floder_path, label_name=label_name)
    xlf = xgb.XGBClassifier(booster="gbtree",
                            silent=True,
                            nthread=-1,
                            scale_pos_weight=0,
                            n_estimators=500,
                            max_depth=100,
                            min_child_weight=1,
                            subsample=0.85,
                            colsample_bytree=0.7,
                            learning_rate=0.01,
                            objective='multi:softmax',
                            reg_alpha=1,
                            reg_lambda=1,
                            gamma=0,
                            gpu_id=0,)
    xlf.fit(X=feature_train, y=label_train, eval_metric='mlogloss', verbose=True,
            eval_set=[(feature_test, label_test)], )
    print(feature_train.shape, label_train.shape, feature_test.shape, label_test.shape)
    return xlf, feature_train, feature_test, label_test


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


def train_loop(target_id, target_save_path, crosstab_result,data_path):
    target_dict = {
        '0': 'heigh',
        '1': 'low',
        '2': 'close',
        '3': 'trend'
    }
    temp_crosstab_result = {}
    for i in list(target_dict.keys()):
        xlf, feature_train, feature_test, label_test = train_mode(target=i, stock_id=target_id,csv_floder_path=data_path)
        evals_result = xlf.evals_result()['validation_0']['mlogloss'][-1]
        xmp = '{}/stock_id_{}_target_{}_loss_{}_time_{}'.format(target_save_path, target_id, target_dict[i],
                                                                evals_result, time.time())
        save_model(xlf, xmp)
        x = pd.DataFrame(feature_train)
        ceate_feature_map(x.columns, path=xmp)
        fig_fixing(xlf, xmp)
        preds = xlf.predict(feature_test)
        data = pd.DataFrame({'label': label_test, 'pred': preds})
        temp_crosstab_result.update({target_dict[i]: pd.crosstab([data.label], [data.pred])})
    crosstab_result.update({target_id: temp_crosstab_result})
    return crosstab_result


path = 'C:/Users/ASUS/PycharmProjects/untitled1/QT/XGB_model/0_1/'
stock_target, crosstab_result = get_loop_target(fn), {}
stock_save_path = {i: '{}/{}'.format(model_save_path, i) for i in stock_target}
[check_path(i) for i in stock_save_path.values()]
for i in stock_target:
    crosstab_result = train_loop(i, stock_save_path[i], crosstab_result, data_path=fn)
