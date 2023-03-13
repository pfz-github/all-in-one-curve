from builtins import print
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
# import operator
from tensorflow.keras import backend as K
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import interp1d 

def lr_func_step_pretrain(cur_iter):
    max_iter = 2000 
    base_lr = 3e-3 
    lr_out = base_lr
    iter_frac = float(cur_iter) / max_iter
    if iter_frac < 0.05:
        lr_out = base_lr
    elif iter_frac < 0.1:
        lr_out = base_lr*0.3
    elif iter_frac < 0.2:
        lr_out = base_lr*0.1
    elif iter_frac < 0.35:
        lr_out = base_lr*0.03
    elif iter_frac < 0.5:
        lr_out = base_lr*0.01
    elif iter_frac < 0.65:
        lr_out = base_lr*0.003
    elif iter_frac < 0.8:
        lr_out = base_lr*0.001
    else:
        lr_out = base_lr*0.0003
    return lr_out

def lr_func_step(cur_iter):
    max_iter = 500 #1000 
    base_lr = 5e-3 
    lr_out = base_lr
    iter_frac = float(cur_iter) / max_iter
    if iter_frac < 0.05:
        lr_out = base_lr
    elif iter_frac < 0.1:
        lr_out = base_lr*0.5
    elif iter_frac < 0.25:
        lr_out = base_lr*0.1
    elif iter_frac < 0.5:
        lr_out = base_lr*0.05
    elif iter_frac < 0.75:
        lr_out = base_lr*0.01
    else:
        lr_out = base_lr*0.001
    return lr_out
    
def read_dataset(root_dir,archive_name,dataset_name):
    datasets_dict = {}
    file_name = root_dir+archive_name+'/'+dataset_name+'/'
    x_train = np.load(file_name + 'x_train.npy',allow_pickle=True)
    y_train = np.load(file_name + 'y_train.npy',allow_pickle=True)
    x_test = np.load(file_name + 'x_valid.npy',allow_pickle=True)
    y_test = np.load(file_name + 'y_valid.npy',allow_pickle=True)
    datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
        y_test.copy())
    return datasets_dict

def load_and_preprocess(root_path,archive_name,dataset_name):
    datasets_dict = read_dataset(root_path,archive_name,dataset_name)
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))

    # make the min to zero of labels
    y_train,y_test = transform_labels(y_train,y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = OneHotEncoder()
    enc.fit(np.concatenate((y_train,y_test),axis =0).reshape(-1,1))
    y_train = enc.transform(y_train.reshape(-1,1)).toarray()
    y_test = enc.transform(y_test.reshape(-1,1)).toarray()

    if len(x_train.shape) == 2: # if univariate
        x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))
        
    return x_train, y_train, x_test, y_test, y_true, nb_classes


def transform_labels(y_train,y_test,y_val=None):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    if not y_val is None :
        # index for when resplitting the concatenation
        idx_y_val = len(y_train)
        idx_y_test = idx_y_val + len(y_val)
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_val_test = np.concatenate((y_train,y_val,y_test),axis =0)
        # fit the encoder
        encoder.fit(y_train_val_test)
        # transform to min zero and continuous labels
        new_y_train_val_test = encoder.transform(y_train_val_test)
        # resplit the train and test
        new_y_train = new_y_train_val_test[0:idx_y_val]
        new_y_val = new_y_train_val_test[idx_y_val:idx_y_test]
        new_y_test = new_y_train_val_test[idx_y_test:]
        return new_y_train, new_y_val,new_y_test
    else:
        # no validation split
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_test = np.concatenate((y_train,y_test),axis =0)
        # fit the encoder
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels
        new_y_train_test = encoder.transform(y_train_test)
        # resplit the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        return new_y_train, new_y_test



def save_logs(output_directory, hist, y_pred, y_true,duration,lr=True,y_true_val=None,y_pred_val=None):

    np.save(output_directory+'y_pred.npy',y_pred)
    np.save(output_directory+'y_true.npy',y_true)

    hist_df = pd.DataFrame(hist.history)
    print(hist_df.head())
    hist_df.to_csv(output_directory+'history.csv', index=True)

    return None 

