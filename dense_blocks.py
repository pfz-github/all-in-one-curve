import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv1D, Dense, BatchNormalization, Activation, ReLU, Dropout
from tensorflow.keras.layers import MaxPool1D, GlobalAveragePooling1D, AveragePooling1D
from tensorflow.keras.layers import Concatenate, Dropout, Lambda, UpSampling1D, Add
from tensorflow.keras.regularizers import l2, l1_l2

def inception_block(nb_filters,inc_bottleneck_size,inc_n,inc_kernel_size,stride=1,activation='linear',name_prefix=None):
    inc_use_bottleneck = inc_bottleneck_size>0
    
    def f(x):
        if inc_use_bottleneck and int(x.shape[-1]) > inc_bottleneck_size:
            x2 = Conv1D(filters=inc_bottleneck_size, kernel_size=1, padding='same', activation=activation, use_bias=False,\
                       name=name_prefix+"_botl")(x)
        else:
            x2 = x

        kernel_size_s = [inc_kernel_size // (2 ** i) for i in range(inc_n)]
        conv_list = []
        for i in range(len(kernel_size_s)):
            conv_list.append(Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i], strides=stride, padding='same', \
                                    activation=activation, use_bias=False,name=name_prefix+str(i))(x2))

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same',name=name_prefix+"_pool")(x)
        pool_conv = Conv1D(filters=nb_filters, kernel_size=1, padding='same', activation=activation, use_bias=False,\
                           name=name_prefix+"_poolconv")(max_pool_1)
        
        conv_list.append(pool_conv)
        x3 = Concatenate(axis=2,name=name_prefix+"_conc")(conv_list)
        x3 = BatchNormalization(name=name_prefix+"_bn")(x3)
        x3 = Activation(activation='relu',name=name_prefix+"_relu")(x3)
        return x3
    return f

    
def inception_dense_block(k, num_layers, bottleneck_size, inc_bottleneck_size, inc_n, inc_kernel_size, stride, reg_w=[0,0],name_blk=None):
    """
    A single inception_dense block of the Densenet
    param k: int representing the "growth rate" of the DenseNet
    param num_layers: int representing the number of layers in the block
    param kernel_width: int representing the width of the main conv kernel
    param bottleneck_size: int representing the size of the bottleneck, as a multiple of k. Set to 0 for no bottleneck.
    return a function wrapping the entire dense block
    """
        
    def f(x):
        if stride>1:
            x = MaxPool1D(pool_size=3, strides=stride, padding='same',name=name_blk+"_maxpool")(x)

        x0 = inception_block(k,inc_bottleneck_size,inc_n,inc_kernel_size,stride=stride,name_prefix=name_blk+"_inc")(x)
        x = Concatenate(name=name_blk+"_conc0")([x,x0])
        
        for i in range(1,num_layers):     
            kernel_width = 3
            if (num_layers>2) and (i<num_layers-1):
                kernel_width = 5
            x0 = H_l(k, bottleneck_size, kernel_width, stride, reg_w,name_prefix=name_blk+"_HL_"+str(i))(x)
            x = Concatenate(name=name_blk+"_conc"+str(i))([x,x0])

        return x

    return f

def H_l(k, bottleneck_size, kernel_width, stride, reg_w=[0,0], name_prefix=None):
    '''
    A single conv layer as defined by Huang et al as H_l in the original paper
    param k: int representing the "growth rate" of the DenseNet
    param bottleneck_size: int representing the size of the bottleneck, as a multiple of k. Set to 0 for no bottleneck.
    param kernel_width: int representing the width of the main conv kernel
    return a function wrapping the keras layers for H_l
    '''
    use_bottleneck = bottleneck_size>0
    def f(x):
        if use_bottleneck:
            x = BatchNormalization(name=name_prefix+"_botl_bn")(x)
            x = Activation("relu",name=name_prefix+"_botl_relu")(x)
            if max(reg_w)>0:
                x = Conv1D(bottleneck_size, 1, strides=1, padding="same", activation="linear", use_bias=False, \
                           kernel_regularizer=l1_l2(l1=reg_w[0],l2=reg_w[1]),name=name_prefix+"_botl")(x)
            else:
                x = Conv1D(bottleneck_size, 1, strides=1, padding="same", activation="linear", use_bias=False,name=name_prefix+"_botl")(x)
                
        if max(reg_w)>0:
            x = Conv1D(k, kernel_width, strides=stride, padding="same", kernel_regularizer=l1_l2(l1=reg_w[0],l2=reg_w[1]),\
                       name=name_prefix+"_conv")(x)
        else:
            x = Conv1D(k, kernel_width, strides=stride, padding="same",name=name_prefix+"_conv")(x)
        return x
    return f


def dense_block(k, num_layers, kernel_width, bottleneck_size, reg_w=[0,0], chshuffle=False):
    """
    A single dense block of the Densenet
    param k: int representing the "growth rate" of the DenseNet
    param num_layers: int representing the number of layers in the block
    param kernel_width: int representing the width of the main conv kernel
    param bottleneck_size: int representing the size of the bottleneck, as a multiple of k. Set to 0 for no bottleneck.
    return a function wrapping the entire dense block
    """
    def f(x):
        for i in range(num_layers):
            x0 = H_l(k, bottleneck_size, kernel_width, reg_w, name_prefix=name_blk+"_HL_"+str(i))(x)
            x = Concatenate(name_prefix=name_blk+"_conc_"+str(i))([x,x0])
        if chshuffle:
            x = channel_shuffle(x,2)
        return x
    return f


# shuffle 3D tensor (N,L,Ch), assuming channel_last.
def channel_shuffle(inputs,group):
    w,in_ch = K.int_shape(inputs)[1:]
    ch_per_group = in_ch // group
    pre_shape = [-1,w,group,ch_per_group]
    dim = [0,1,3,2]
    post_shape = [-1,w,in_ch]
    
    out = Lambda(lambda z: K.reshape(z,pre_shape))(inputs)
    out = Lambda(lambda z: K.permute_dimensions(z,dim))(out)
    out = Lambda(lambda z: K.reshape(z,post_shape))(out)
    return out



