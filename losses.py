# from __future__ import division, print_function
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from functools import partial

def recall_m(y_true,y_pred):
    true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true,y_pred):
    true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true,y_pred):
    precision = precision_m(y_true,y_pred)
    recall = recall_m(y_true,y_pred)
    f1 = 2*((precision * recall)/(precision + recall+ K.epsilon()))
    return f1

def focal_loss(y_true,y_pred):
    gamma=2.
    alpha=0.25
    pt_1 = tf.where(tf.equal(y_true,1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true,0), y_pred, tf.zeros_like(y_pred))
    return -10.*K.mean(alpha*K.pow(1.-pt_1,gamma)*K.log(pt_1)) - 10.*K.mean((1-alpha)*K.pow(pt_0,gamma)*K.log(1.-pt_0))

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(
        K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def amsoftmax_loss(y_true, y_pred, margin=0.35, scale=30.0):
    m = tf.constant(margin, name='m')
    s = tf.constant(scale, name='s')
    y_pred = tf.subtract(y_pred, y_true*m) * s
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    loss = tf.reduce_mean(loss)
    return loss    
 
def ce_focal_loss(y_true,y_pred):
    alpha = 0.5
    loss1 = categorical_crossentropy(y_true, y_pred)
    gamma=2.
    alpha2=0.25
    pt_1 = tf.where(tf.equal(y_true,1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true,0), y_pred, tf.zeros_like(y_pred))
    loss2 = -K.mean(alpha2*K.pow(1.-pt_1,gamma)*K.log(pt_1)) - K.mean((1-alpha2)*K.pow(pt_0,gamma)*K.log(1.-pt_0))
    return alpha*loss1 + (1-alpha)*loss2

def ce_amsoft_loss(y_true,y_pred):
    alpha = 0.5
    loss1 = categorical_crossentropy(y_true, y_pred)
    m = tf.constant(0.35, name='m')
    s = tf.constant(30.0, name='s')
    y_pred = tf.subtract(y_pred, y_true*m) * s
    loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    loss2 = tf.reduce_mean(loss2)
    return alpha*loss1 + (1-alpha)*loss2

    
    
