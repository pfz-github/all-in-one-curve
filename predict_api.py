import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sys.path.append("/rdfs/fast/home/peifang/cardi/tsc_1d")
from losses import *

def print_top1_metrics(y_test, y_pred_top1, num_cls, model_term=''):
    metrics = []
    print(model_term + ': top1 recall for class 0 to '+str(num_cls-1))
    for i in range(num_cls):
        metrics.append(recall_per_class(y_test, y_pred_top1, i)); print(metrics[-1])
    print(model_term + ': top1 precision for class 0 to '+str(num_cls-1))
    for i in range(num_cls):
        metrics.append(precision_per_class(y_test, y_pred_top1, i)); print(metrics[-1])
    print(model_term + ': top1 F1 score for class 0 to '+str(num_cls-1))
    for i in range(num_cls):
        metrics.append(top1_f1_score_per_class(y_test, y_pred_top1, i)); print(metrics[-1])
    print(model_term + ': top1 confusion matrix')
    cm_top1 = confusion_matrix(y_test, y_pred_top1); print(cm_top1)
    return metrics, cm_top1

# confusion matrix in (N,N)
def metrics_from_cm(cm):
    num_class = cm.shape[0]
    assert(cm.shape[0]==cm.shape[1])
    metrics = np.zeros((num_class,3))  #3:(recall, precision, F1)
    acc = 0
    for nn in range(num_class):
        metrics[nn,0] = cm[nn,nn]/np.sum(cm[nn,:])  # recall
        metrics[nn,1] = cm[nn,nn]/np.sum(cm[:,nn])  # precision
        metrics[nn,2] = (2*metrics[nn,0]*metrics[nn,1])/(metrics[nn,0]+metrics[nn,1]+1e-6)  # F1
        acc += cm[nn,nn]
    acc /= np.sum(np.sum(cm,axis=0))
    return metrics,acc

def get_curve_model(model_path, loss_type):

    if (loss_type == 'ce'):
        custom_objects = {}
    elif (loss_type == 'ce_focal'):
        custom_objects = {'ce_focal_loss':ce_focal_loss}
    elif (loss_type == 'focal'):
        custom_objects = {'focal_loss':focal_loss}
    elif (loss_type == 'amsoft'):
        custom_objects = {'amsoftmax_loss':amsoftmax_loss}
    elif (loss_type == 'ce_amsoft'):
        custom_objects = {'ce_amsoft_loss':ce_amsoft_loss}
    else:
        custom_objects = {}
    print('--- loading model at ',model_path)
    model = load_model(model_path+'best_model.hdf5', custom_objects=custom_objects)
    return model

def metrics_per_class(y_true,y_pred,N_class):
    #y_true = np.array([3,2,1,0,0,2,2,3,1,0,3,3,0,1,1,2])
    #y_pred = np.array([3,1,1,0,2,2,0,3,1,3,3,2,0,1,2,2])
    results = np.zeros((4,N_class))
    for i in range(N_class):
        true_pos = np.where(y_true==i)[0]
        true_neg = np.where(y_true!=i)[0]
        pred_pos = np.where(y_pred==i)[0]
        pred_neg = np.where(y_pred!=i)[0]

        TP = len(list(set(true_pos).intersection(set(pred_pos)))); #print(TP)
        TN = len(list(set(true_neg).intersection(set(pred_neg)))); #print(TN)
        FP = len(list(set(true_neg).intersection(set(pred_pos)))); #print(FP)
        FN = len(list(set(true_pos).intersection(set(pred_neg)))); #print(FN)

        recall = TP/(TP+FN+1e-6)
        precision = TP/(TP+FP+1e-6)
        tpr = recall
        fpr = FP/(FP+TN+1e-6)
        F1 = 2*precision*recall/(precision+recall+1e-6)
        results[:,i] = recall, precision, fpr, F1
    return results

# top1 recall_per_class: y_true in (N,) y_pred in (N,)
def recall_per_class(y_true, y_pred, index):
    tmp = []
    for jj in range(len(y_true)):
        if(y_true[jj]==index):
            tmp.append(jj)
    tmp2 = []
    for ii in tmp:
        if(np.int8(y_pred[ii])==index):
            tmp2.append(ii)
    return float(len(tmp2))/(float(len(tmp))+1e-6)

def precision_per_class(y_true, y_pred, index):
    tmp = []
    for jj in range(len(y_pred)):
        if(np.int8(y_pred[jj])==index):
            tmp.append(jj)
    tmp2 = []
    for ii in tmp:
        if(y_true[ii]==index):
            tmp2.append(ii)
    return float(len(tmp2))/(float(len(tmp))+1e-6)

def top1_f1_score_per_class(y_true, y_pred, index):
    rec = recall_per_class(y_true, y_pred, index)
    prec = precision_per_class(y_true, y_pred, index)
    f1 = 2*rec*prec/(rec+prec+1e-3)
    return f1

def calc_acc(y_true, y_pred):
    tot = min(len(y_true),len(y_pred))
    same_acct = 0
    for ii in range(tot):
        if (y_true[ii]==y_pred[ii]):
            same_acct += 1
    return same_acct/(tot+1e-6)
    

def print_recall_confusion(confusion_maxrix, labels):
    assert len(confusion_maxrix)==len(labels)
    fig = plt.figure(figsize=(3,3))
    ax = plt.subplot(1,1,1)
    for row in range(confusion_maxrix.shape[0]):
        for col in range(confusion_maxrix.shape[1]):
            c = confusion_maxrix[row][col]
            if c > 0.0:
                plt.text(col, row, "%0.2f" %(c,), color='w' if c>0.5 else 'black', 
                         va='center', ha='center', fontsize=10)        
    cax = ax.matshow(confusion_maxrix, cmap='gray_r',vmax=1)
    ax.set_xticklabels([' ']+ labels, rotation=90, fontsize=10)
    ax.set_yticklabels([' ']+ labels, fontsize=10)
    plt.show()

def print_confusion(confusion_maxrix, labels):
    assert len(confusion_maxrix)==len(labels)
    fig = plt.figure(figsize=(3,3))
    ax = plt.subplot(1,1,1)
    for row in range(confusion_maxrix.shape[0]):
        for col in range(confusion_maxrix.shape[1]):
            c = confusion_maxrix[row][col]
            if c > 0.0:
                plt.text(col, row, c, color='w' if c> np.max(confusion_maxrix)//2 else 'black', 
                         va='center', ha='center', fontsize=15)        
    cax = ax.matshow(confusion_maxrix, cmap='Blues')
    ax.set_xticklabels([' ']+ labels, rotation=90, fontsize=15)
    ax.set_yticklabels([' ']+ labels, fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def predict_one_curve(model_directory, input_curve):
    model = get_model(model_directory)
    y_pred = model.predict(input_curve)
    # convert the predicted from binary to integer
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred

