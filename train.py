import os
import numpy as np
import random
import sys
import math
import time
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.models import load_model
from my_models import *
from utils import *
from losses import *
from predict_api import get_curve_model

def set_global_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)


def create_classifier(classifier_name, input_shape, nb_classes):
    if classifier_name=='wide_resnet7':
        return wide_resnet7(input_shape, nb_classes)
    elif classifier_name=='incept_dense14':
        return incept_dense14(input_shape, nb_classes)
    elif classifier_name=='incept_dense18':
        return incept_dense18(input_shape, nb_classes)
    elif classifier_name=='incept_dense22':
        return incept_dense22(input_shape, nb_classes)
    elif classifier_name=='incept_dense10':
        return incept_dense10(input_shape, nb_classes)
    else:
        return wide_resnet7(input_shape, nb_classes)


def train():
    K.clear_session()

    root_path = "/myroot/"
    pretrain_root = "/myroot/"
    # parsing the system arguments
    archive_name = sys.argv[1]
    dataset_name = sys.argv[2]
    classifier_name=sys.argv[3]
    loss_type = sys.argv[4]
    monitor_type = sys.argv[5]
    itr = sys.argv[6]

    #True only when using pretrain_data
    pretrain_en = (archive_name.find('pretrain_data')>-1)
    print('pretrain_en: ',pretrain_en)

    # fine tune setting
    load_existing_model = True #True for fine tune, False for cold start
    include_top = False
    reduce_lr_only = True
    base_lr = 5e-3
    batch_size = 64
    nb_epochs = 1000

    if pretrain_en:
        load_existing_model = False
        reduce_lr_only = False
        base_lr = 3e-3
        batch_size = 128
        nb_epochs = 2000

    if loss_type.find('amsoft')>-1:
        classifier_name = classifier_name+"_amsoft"

    # create output directory
    if (loss_type == 'ce') or (loss_type == 'amsoft'):
         output_directory = root_path+'results/'+classifier_name+'/'+archive_name+itr+'/'+dataset_name
    else:
        output_directory = root_path+'results/'+classifier_name+'_'+loss_type+'/'+archive_name+itr+'/'+dataset_name

    if (not pretrain_en) and (load_existing_model):
        if reduce_lr_only:
            output_directory = output_directory+"_fine_reducelr"+str(base_lr)+"/"
        else:
            output_directory = output_directory+"_fine_steplr"+str(base_lr)+"/"
    else:
        if reduce_lr_only:
            output_directory = output_directory+"_reducelr"+str(base_lr)+"/"
        else:
            output_directory = output_directory+"_steplr"+str(base_lr)+"/"

    if os.path.exists(output_directory):
        print('Already done')
    else:
        print('------ Method(archive, dataset, classifier, iteration): ', archive_name, dataset_name, classifier_name, itr,' ------')
        os.makedirs(output_directory)
        os.makedirs(output_directory+'/log')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            # load the data and preprocessing
            x_train, y_train, x_test, y_test, y_true, nb_classes = load_and_preprocess(root_path,archive_name,dataset_name)
            input_shape = x_train.shape[1:]


            if (loss_type.find('amsoft')>-1):
                custom_objects = {'amsoftmax_loss':amsoftmax_loss}
                custom_metrics=['accuracy']
                loss_func = amsoftmax_loss
            elif (loss_type.find('focal')>-1):
                custom_objects = {'focal_loss':focal_loss}
                custom_metrics=['accuracy']
                loss_func = focal_loss
            elif (loss_type.find('ce_focal')>-1):
                custom_objects = {'ce_focal_loss':ce_focal_loss}
                custom_metrics=['accuracy']
                loss_func = ce_focal_loss
            else:  #default 'ce'
                custom_objects = {}
                custom_metrics=['accuracy']
                loss_func = 'categorical_crossentropy'
            monitor_term = 'val_acc'  #'loss'
            custom_objects.update({'tf':tf})


            if load_existing_model:
                if input_shape[0] == 10525:
                    model_path = pretrain_root+"results/"+classifier_name+"/pretrain_data_itr_0/len10525_steplr0.003/"
                    model = get_curve_model(model_path, loss_type)
                elif input_shape[0] == 6725:
                    model_path = pretrain_root+"results/"+classifier_name+"/pretrain_data_itr_0/len6725_steplr0.003/"
                    model = get_curve_model(model_path, loss_type)
                elif input_shape[0] == 3800:
                    model_path = pretrain_root+"results/"+classifier_name+"/pretrain_data_itr_0/len3800_steplr0.003/"
                    model = get_curve_model(model_path, loss_type)
                elif input_shape[0] == 4125:
                    model_path = pretrain_root+"results/"+classifier_name+"/pretrain_data_itr_0/len4125_steplr0.003/"
                    model = get_curve_model(model_path, loss_type)
                elif input_shape[0] == 4875:
                    model_path = pretrain_root+"results/"+classifier_name+"/pretrain_data_itr_0/len4875_steplr0.003/"
                    model = get_curve_model(model_path, loss_type)
                elif input_shape[0] == 11275:
                    model_path = pretrain_root+"results/"+classifier_name+"/pretrain_data_itr_0/len11275_steplr0.003/"
                    model = get_curve_model(model_path, loss_type)
                elif input_shape[0] == 4075:
                    model_path = pretrain_root+"results/"+classifier_name+"/pretrain_data_itr_0/len4075_steplr0.003/"
                    model = get_curve_model(model_path, loss_type)

                if not include_top:
                    print('--- nb_classes ',nb_classes)
                    x = Dense(nb_classes, activation="softmax", name="output_layer")(model.layers[-2].output)
                    model2 = Model(inputs = model.input, outputs=x)
                else:
                    if input_shape[0] == 11275:
                        model_path = root_path+"results/"+classifier_name+"/cine_lge_clin_6class_itr_0/fold1_fine_reducelr0.005/"
                        model2 = get_curve_model(model_path, loss_type)
                    elif input_shape[0] == 10525:
                        model_path = root_path+"results/"+classifier_name+"/cine_lge_6class_itr_0/fold1_fine_reducelr0.005/"
                        model2 = get_curve_model(model_path, loss_type)
                    elif input_shape[0] == 6725:
                        model_path = root_path+"results/"+classifier_name+"/cine_6class_itr_0/fold1_fine_reducelr0.005/"
                        model2 = get_curve_model(model_path, loss_type)
                    elif input_shape[0] == 3800:
                        model_path = root_path+"results/"+classifier_name+"/xjgs_lge_2class_itr_0/fold1_fine_reducelr0.005/"
                        model2 = get_curve_model(model_path, loss_type)
                    elif input_shape[0] == 4125:
                        model_path = root_path+"results/"+classifier_name+"/new_xjgs_lge_2class_itr_0/fold1_fine_reducelr0.005/"
                        model2 = get_curve_model(model_path, loss_type)
            else:
                # create classifier
                model2 = create_classifier(classifier_name, input_shape, nb_classes)
                print('--- cold start: classifier ',classifier_name,' is created.---')

            # compile and run the classifier
            model2.compile(loss=loss_func, optimizer=Adam(base_lr), metrics=custom_metrics)

            # callbacks
            if pretrain_en:
                reduce_lr = ReduceLROnPlateau(monitor=monitor_term, factor=0.5, patience=50, min_lr=1e-7)
            else:
                reduce_lr = ReduceLROnPlateau(monitor=monitor_term, factor=0.5, patience=50, min_lr=1e-6)
            file_path = output_directory+'best_model.hdf5'
            model_checkpoint = ModelCheckpoint(filepath=file_path, monitor=monitor_term, save_best_only=True)

            if pretrain_en:
                lr_schedule = LearningRateScheduler(lr_func_step_pretrain)
            else:
                lr_schedule = LearningRateScheduler(lr_func_step)

            if reduce_lr_only:
                callbacks = [reduce_lr,model_checkpoint]
            else:
                callbacks = [reduce_lr,lr_schedule,model_checkpoint]


            # train
            start_time = time.time()
            hist = model2.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, shuffle=True, \
                             verbose=True, validation_data=(x_test,y_test), callbacks=callbacks)
            duration = time.time() - start_time

            # evaluate
            model = load_model(output_directory+'best_model.hdf5',custom_objects=custom_objects)
            y_pred = model2.predict(x_test)
            y_pred = np.argmax(y_pred , axis=1)
            save_logs(output_directory, hist, y_pred, y_true, duration)

        print('DONE')


if __name__=="__main__":
#     gpuid = survey()
#     print('Choice GPU: ', gpuid)
# #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
    train()
