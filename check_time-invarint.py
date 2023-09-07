import os
from sklearn.svm import SVC
import numpy as np
from sklearn.utils import shuffle
import nibabel as nib
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import math
import lmdb
import warnings
from nilearn.connectome import ConnectivityMeasure
from sklearn.model_selection import StratifiedShuffleSplit
import argparse
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--time",'-t', type=int,default = 15)
parser.add_argument("--mask",'-m', type=int,default = 10)
parser.add_argument("--per",'-p', type=float,default = 0.2)
args = parser.parse_args()

#shuffle_seed = 0

template = 'sch'
root = "/data5/yang/large/timeseries_adhd"
df = pd.read_csv("/data5/yang/large/clean_adhd.csv")

use_idx = df[df['dx']!='pending'].index
df = df.iloc[use_idx].reset_index(drop=True)

names = list(df['new_name'])
all_data = np.array(names)

#lbls = np.array(list([1 if f == 1 else 0 for f in df['dx'] ])) # abide
lbls = np.array(list([0 if f == 0 else 1 for f in df['dx'].astype(int) ])) # adhd
print(lbls)
sites = np.array(df['site'])

train_length = int(len(df) * 0.7)
val_length = int(len(df) * 0.15)
test_length = int(len(df) * 0.15)

correlation_measure = ConnectivityMeasure(kind='correlation')

def get_data(lst):
    ret = []
    for elem in lst:
        img = np.load(os.path.join(root, f"{template}_{elem}.npy"))
        # for random select
        # if args.time >= img.shape[1]:
        #     select_img = img
        # else:
        #     rand_index = np.random.randint(args.time, img.shape[1])
        #     select_img = img[:,rand_index- args.time:rand_index]

        #for percantage select
        # time_len = img.shape[1]
        # select_len = int(args.per * time_len)
        # #select_len = select_len if select_len > 85 else 85
        # rand_index = np.random.randint(select_len, img.shape[1])
        # #rand_index = select_len
        # select_img = img[:,rand_index- select_len:rand_index]

        # for random mask
        time_len = img.shape[1]
        mask_index = np.array(random.sample(list(np.arange(0,time_len)),args.mask))
        bool_mask = np.zeros((time_len))
        bool_mask[mask_index]=1
        bool_mask = bool_mask.astype(bool)
        select_img = img[:,~bool_mask]

        correlation_matrix = correlation_measure.fit_transform([select_img.T])[0]
        ret.append(correlation_matrix)
    return np.array(ret)

split = StratifiedShuffleSplit(n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
for train_index, test_valid_index in split.split(all_data, sites):
    data_train, labels_train = all_data[train_index], lbls[train_index]
    data_rest, labels_rest = all_data[test_valid_index], lbls[test_valid_index]
    site_rest = sites[test_valid_index]

res_val = []
res_test = []
for shuffle_seed in range(10):
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length, random_state=shuffle_seed)
    for valid_index, test_index in split2.split(data_rest, site_rest):
        data_test, labels_test = data_rest[test_index], labels_rest[test_index]
        data_val, labels_val = data_rest[valid_index], labels_rest[valid_index]

    fit_train = get_data(data_train).reshape(len(data_train), -1)
    fit_val = get_data(data_val).reshape(len(data_val), -1)
    fit_test = get_data(data_test).reshape(len(data_test), -1)

    clf = SVC()
    #print(fit_train.shape, labels_train.shape);exit()
    clf.fit(fit_train, labels_train)

    pred_val = clf.predict(fit_val)
    pred_test = clf.predict(fit_test)

    acc_val = accuracy_score(pred_val, labels_val)
    cm_val = confusion_matrix(pred_val,labels_val)
    val_sen = round(cm_val[1, 1] / float(cm_val[1, 1]+cm_val[1, 0]),4)
    val_spe = round(cm_val[0, 0] / float(cm_val[0, 0]+cm_val[0, 1]),4)
    val_auc = roc_auc_score(pred_val, labels_val)

    acc_test = accuracy_score(pred_test, labels_test)
    cm_test = confusion_matrix(pred_test,labels_test)
    test_sen = round(cm_test[1, 1] / float(cm_test[1, 1]+cm_test[1, 0]),4)
    test_spe = round(cm_test[0, 0] / float(cm_test[0, 0]+cm_test[0, 1]),4)
    tets_auc = roc_auc_score(pred_test, labels_test)

    res_val.append({"acc":acc_val,"spe":val_spe,"sen":val_sen,"auc":val_auc})
    res_test.append({"acc":acc_test,"spe":test_spe,"sen":test_sen,"auc":tets_auc})

mean_acc_val = np.mean([f['acc'] for f in res_val if f['acc'] == f['acc']]) * 100
mean_sen_val = np.mean([f['sen'] for f in res_val if f['sen'] == f['sen']]) * 100
mean_spe_val = np.mean([f['spe'] for f in res_val if f['spe'] == f['spe']]) * 100
mean_auc_val = np.mean([f['auc'] for f in res_val if f['auc'] == f['auc']]) * 100

mean_acc_test = np.mean([f['acc'] for f in res_test if f['acc'] == f['acc']]) * 100
mean_sen_test = np.mean([f['sen'] for f in res_test if f['sen'] == f['sen']]) * 100
mean_spe_test = np.mean([f['spe'] for f in res_test if f['spe'] == f['spe']]) * 100
mean_auc_test = np.mean([f['auc'] for f in res_test if f['auc'] == f['auc']]) * 100

print(f"validation acc: {mean_acc_val:.3f} test acc: {mean_acc_test:.3f}")
print(f"validation sen: {mean_sen_val:.3f} test sen: {mean_sen_test:.3f}")
print(f"validation spe: {mean_spe_val:.3f} test spe: {mean_spe_test:.3f}")
print(f"validation auc: {mean_auc_val:.3f} test auc: {mean_auc_test:.3f}")

with open("check_adhd_mask.txt", 'a') as f:
    f.write(f"======{args.mask}======\n")
    f.write(f"validation acc: {mean_acc_val:.3f} test acc: {mean_acc_test:.3f}\n")
    f.write(f"validation sen: {mean_sen_val:.3f} test sen: {mean_sen_test:.3f}\n")
    f.write(f"validation spe: {mean_spe_val:.3f} test spe: {mean_spe_test:.3f}\n")
    f.write(f"validation auc: {mean_auc_val:.3f} test auc: {mean_auc_test:.3f}\n")