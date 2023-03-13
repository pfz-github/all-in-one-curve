import os
import numpy as np 
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv1D, Dense, BatchNormalization, Activation, ReLU, Dropout
from tensorflow.keras.layers import MaxPool1D, GlobalAveragePooling1D, AveragePooling1D
from tensorflow.keras.layers import Concatenate, Dropout, Lambda, UpSampling1D, Add
from tensorflow.keras.models import Model
from dense_blocks import *
import sys
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.regularizers import l2, l1_l2

def incept_dense10(input_shape, nb_classes):
    initial_pool_width=3
    initial_pool_strides=2
    k=24
    transition_pool_size=2
    transition_pool_strides=2

    input_layer = Input(input_shape)
    print('input_shape ',input_shape)
    x1 = Conv1D(filters=32, kernel_size=5, strides=2, padding="same", name="input_conv")(input_layer)
    x1 = BatchNormalization(name="input_conv_bn")(x1)
    x1 = Activation("relu",name="input_conv_relu")(x1)
    x1 = MaxPool1D(pool_size=initial_pool_width, strides=initial_pool_strides, padding="same", name="input_maxpool")(x1)

    # add dense_inception blocks with pooling
    block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride = k, k, 3, 29, 1 
    x2 = inception_dense_block(k, block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride, reg_w=[0,0],\
                               name_blk="d_in_blk_0")(x1)
    x2 = BatchNormalization(name="d_in_blk_0_bn")(x2)
    x2 = Activation("relu",name="d_in_blk_0_relu")(x2)
    x2 = AveragePooling1D(pool_size=transition_pool_size, strides=transition_pool_strides, padding="same",name="d_in_blk_0_pool")(x2)
    
    # add the last dense_inception block
    final_block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride =  k, k, 3, 29, 1 
    x3 = inception_dense_block(k, final_block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride=1, reg_w=[0,0],\
                              name_blk="d_in_blk_1")(x2)
    x3 = BatchNormalization(name="d_in_blk_1_bn")(x3)
    x3 = Activation("relu",name="d_in_blk_1_relu")(x3)
    
    x4 = GlobalAveragePooling1D(name="global_pool")(x3)
    print('incept_dense10, global pooling in and out ',x3.shape,x4.shape)
    output_layer = Dense(nb_classes, activation="softmax", name="output_layer")(x4)
        
    model = Model(inputs=input_layer, outputs=output_layer, name="incept_dense10")
    return model


def incept_dense14(input_shape, nb_classes):
    initial_pool_width=3
    initial_pool_strides=2
    k=24
    transition_pool_size=2
    transition_pool_strides=2

    input_layer = Input(input_shape)
    print('input_shape ',input_shape)
    x1 = Conv1D(filters=32, kernel_size=5, strides=2, padding="same", name="input_conv")(input_layer)
    x1 = BatchNormalization(name="input_conv_bn")(x1)
    x1 = Activation("relu",name="input_conv_relu")(x1)
    x1 = MaxPool1D(pool_size=initial_pool_width, strides=initial_pool_strides, padding="same", name="input_maxpool")(x1)

    # add dense_inception blocks with pooling
    block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride = k, k, 3, 29, 1 
    x2 = inception_dense_block(k, block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride, reg_w=[0,0],\
                               name_blk="d_in_blk_0")(x1)
    x2 = BatchNormalization(name="d_in_blk_0_bn")(x2)
    x2 = Activation("relu",name="d_in_blk_0_relu")(x2)
    x2 = AveragePooling1D(pool_size=transition_pool_size, strides=transition_pool_strides, padding="same",name="d_in_blk_0_pool")(x2)
    
    # add dense_inception blocks with pooling
    block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride = k, k, 3, 29, 1 
    x3 = inception_dense_block(k, block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride, reg_w=[0,0],\
                               name_blk="d_in_blk_1")(x2)
    x3 = BatchNormalization(name="d_in_blk_1_bn")(x3)
    x3 = Activation("relu",name="d_in_blk_1_relu")(x3)
    x3 = AveragePooling1D(pool_size=transition_pool_size, strides=transition_pool_strides, padding="same",name="d_in_blk_1_pool")(x3)

    # add the last dense_inception block
    final_block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride =  k, k, 3, 29, 1 
    x4 = inception_dense_block(k, final_block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride=1, reg_w=[0,0],\
                              name_blk="d_in_blk_2")(x3)
    x4 = BatchNormalization(name="d_in_blk_2_bn")(x4)
    x4 = Activation("relu",name="d_in_blk_2_relu")(x4)
    
    x5 = GlobalAveragePooling1D(name="global_pool")(x4)
    output_layer = Dense(nb_classes, activation="softmax", name="output_layer")(x5)
        
    model = Model(inputs=input_layer, outputs=output_layer, name="incept_dense14")
    return model


def incept_dense18(input_shape, nb_classes):
    initial_pool_width=3
    initial_pool_strides=2
    k=24
    transition_pool_size=2
    transition_pool_strides=2

    input_layer = Input(input_shape)
    print('input_shape ',input_shape)
    x1 = Conv1D(filters=32, kernel_size=5, strides=2, padding="same", name="input_conv")(input_layer)
    x1 = BatchNormalization(name="input_conv_bn")(x1)
    x1 = Activation("relu",name="input_conv_relu")(x1)
    x1 = MaxPool1D(pool_size=initial_pool_width, strides=initial_pool_strides, padding="same", name="input_maxpool")(x1)

    # add dense_inception blocks with pooling
    block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride = k, k, 3, 29, 1 
    x2 = inception_dense_block(k, block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride, reg_w=[0,0],\
                               name_blk="d_in_blk_0")(x1)
    x2 = BatchNormalization(name="d_in_blk_0_bn")(x2)
    x2 = Activation("relu",name="d_in_blk_0_relu")(x2)
    x2 = AveragePooling1D(pool_size=transition_pool_size, strides=transition_pool_strides, padding="same",name="d_in_blk_0_pool")(x2)
    
    # add dense_inception blocks with pooling
    block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride = k, k, 3, 29, 1 
    x3 = inception_dense_block(k, block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride, reg_w=[0,0],\
                               name_blk="d_in_blk_1")(x2)
    x3 = BatchNormalization(name="d_in_blk_1_bn")(x3)
    x3 = Activation("relu",name="d_in_blk_1_relu")(x3)
    x3 = AveragePooling1D(pool_size=transition_pool_size, strides=transition_pool_strides, padding="same",name="d_in_blk_1_pool")(x3)

    # add dense_inception blocks with pooling
    block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride = k, k, 3, 29, 1 
    x4 = inception_dense_block(k, block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride, reg_w=[0,0],\
                               name_blk="d_in_blk_2")(x3)
    x4 = BatchNormalization(name="d_in_blk_2_bn")(x4)
    x4 = Activation("relu",name="d_in_blk_2_relu")(x4)
    x4 = AveragePooling1D(pool_size=transition_pool_size, strides=transition_pool_strides, padding="same",name="d_in_blk_2_pool")(x4)

    # add the last dense_inception block
    final_block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride =  k, k, 3, 29, 1 
    x5 = inception_dense_block(k, final_block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride=1, reg_w=[0,0],\
                              name_blk="d_in_blk_3")(x4)
    x5 = BatchNormalization(name="d_in_blk_3_bn")(x5)
    x5 = Activation("relu",name="d_in_blk_3_relu")(x5)
    
    x5 = GlobalAveragePooling1D(name="global_pool")(x5)
    output_layer = Dense(nb_classes, activation="softmax", name="output_layer")(x5)
        
    model = Model(inputs=input_layer, outputs=output_layer, name="incept_dense18")
    return model

def incept_dense22(input_shape, nb_classes):
    initial_pool_width=3
    initial_pool_strides=2
    k=24
    transition_pool_size=2
    transition_pool_strides=2

    input_layer = Input(input_shape)
    print('input_shape ',input_shape)
    x1 = Conv1D(filters=32, kernel_size=5, strides=2, padding="same", name="input_conv")(input_layer)
    x1 = BatchNormalization(name="input_conv_bn")(x1)
    x1 = Activation("relu",name="input_conv_relu")(x1)
    x1 = MaxPool1D(pool_size=initial_pool_width, strides=initial_pool_strides, padding="same", name="input_maxpool")(x1)

    # add dense_inception blocks with pooling
    block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride = k, k, 3, 29, 1 
    x2 = inception_dense_block(k, block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride, reg_w=[0,0],\
                               name_blk="d_in_blk_0")(x1)
    x2 = BatchNormalization(name="d_in_blk_0_bn")(x2)
    x2 = Activation("relu",name="d_in_blk_0_relu")(x2)
    x2 = AveragePooling1D(pool_size=transition_pool_size, strides=transition_pool_strides, padding="same",name="d_in_blk_0_pool")(x2)
    
    # add dense_inception blocks with pooling
    block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride = k, k, 3, 29, 1 
    x3 = inception_dense_block(k, block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride, reg_w=[0,0],\
                               name_blk="d_in_blk_1")(x2)
    x3 = BatchNormalization(name="d_in_blk_1_bn")(x3)
    x3 = Activation("relu",name="d_in_blk_1_relu")(x3)
    x3 = AveragePooling1D(pool_size=transition_pool_size, strides=transition_pool_strides, padding="same",name="d_in_blk_1_pool")(x3)

    # add dense_inception blocks with pooling
    block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride = k, k, 3, 29, 1 
    x4 = inception_dense_block(k, block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride, reg_w=[0,0],\
                               name_blk="d_in_blk_2")(x3)
    x4 = BatchNormalization(name="d_in_blk_2_bn")(x4)
    x4 = Activation("relu",name="d_in_blk_2_relu")(x4)
    x4 = AveragePooling1D(pool_size=transition_pool_size, strides=transition_pool_strides, padding="same",name="d_in_blk_2_pool")(x4)

    # add dense_inception blocks with pooling
    block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride = k, k, 3, 29, 1 
    x5 = inception_dense_block(k, block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride, reg_w=[0,0],\
                               name_blk="d_in_blk_3")(x4)
    x5 = BatchNormalization(name="d_in_blk_3_bn")(x5)
    x5 = Activation("relu",name="d_in_blk_3_relu")(x5)
    x5 = AveragePooling1D(pool_size=transition_pool_size, strides=transition_pool_strides, padding="same",name="d_in_blk_3_pool")(x5)

    # add the last dense_inception block
    final_block_size = 2
    bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride =  k, k, 3, 29, 1 
    x6 = inception_dense_block(k, final_block_size, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride=1, reg_w=[0,0],\
                              name_blk="d_in_blk_4")(x5)
    x6 = BatchNormalization(name="d_in_blk_4_bn")(x6)
    x6 = Activation("relu",name="d_in_blk_4_relu")(x6)
    
    x6 = GlobalAveragePooling1D(name="global_pool")(x6)
    output_layer = Dense(nb_classes, activation="softmax", name="output_layer")(x6)
        
    model = Model(inputs=input_layer, outputs=output_layer, name="incept_dense22")
    return model


def wide_resnet7(input_shape, nb_classes):
    drop_r = 0.05
#     reg_w = [0,1e-3]
    conv_ks = [161,81,3] 

    input_layer = Input(shape = input_shape) # in shape (?,6725,1)
    print('--input shape',input_shape)

    # Resnet BLOCK 1

    x1 = Conv1D(filters=8, kernel_size=conv_ks[0], strides=2, padding="same")(input_layer) #,kernel_regularizer=l1_l2(reg_w[0],reg_w[1])
    x1 = Dropout(drop_r)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)

    x1 = Conv1D(filters=16, kernel_size=conv_ks[1], strides=2, padding="same")(x1)
    x1 = Dropout(drop_r)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    
    x1 = Conv1D(filters=32, kernel_size=conv_ks[2], strides=2, padding="same")(x1)
    x1 = Dropout(drop_r)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)

    # expand channels for the sum
    short1 =Conv1D(filters=32, kernel_size=1, strides=8, padding='same')(input_layer)
    short1 = Dropout(drop_r)(short1)
    short1 =BatchNormalization()(short1)

    out1 = Add()([short1, x1])
    out1 = Activation('relu')(out1)
    print('--resnet block1 out shape',out1.shape)

    # Resnet BLOCK 2

    x2 = Conv1D(filters=32, kernel_size=conv_ks[0], strides=2, padding="same")(out1)
    x2 = Dropout(drop_r)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x2 = Conv1D(filters=64, kernel_size=conv_ks[1], strides=2, padding="same")(x2)
    x2 = Dropout(drop_r)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    
    x2 = Conv1D(filters=128, kernel_size=conv_ks[2], strides=2, padding="same")(x2)
    x2 = Dropout(drop_r)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    # expand channels for the sum
    short2 =Conv1D(filters=128, kernel_size=1, strides=8, padding='same')(out1)
    short2 = Dropout(drop_r)(short2)
    short2 =BatchNormalization()(short2)

    out2 = Add()([short2, x2])
    out2 = Activation('relu')(out2)
    print('--resnet block2 out shape',out2.shape)

    # Final
    avg_pool = GlobalAveragePooling1D()(out2)
    print('--avg pool out shape',avg_pool.shape)
            
    out = Dense(nb_classes,activation = 'softmax')(avg_pool)
    print('--final out shape',out.shape)
    
    model = Model(inputs = [input_layer], outputs = [out])

    return model


