from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_curve, auc, roc_auc_score,matthews_corrcoef  ###计算roc和auc
from keras.callbacks import Callback,EarlyStopping
import random
from keras.layers import *
from keras import backend as K
from Method import *
import os
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef, roc_curve, auc
from keras import backend as K

def expand_dim_backend(x):
    x1 = K.reshape(x,(-1,1,100))
    return x1

def expand_dim_backend2(x):
    x1 = K.reshape(x,(-1,1,51))
    return x1

def expand_dim_backend3(x):
    x1 = K.reshape(x,(-1,1,200))
    return x1

def multiply(a):
    x = np.multiply(a[0], a[1])
    return x

def create_model(input_shape_1= [51, 21],input_shape_2= [2000, 21], unit=128, filter=100):

    X_input_1 = Input(shape=input_shape_1)
    X_input_2 = Input(shape=input_shape_2)

    X_conv_1 = Conv1D(strides=1, kernel_size=1, filters=filter, padding='same')(X_input_1)
    X_conv_1 = Activation('relu')(X_conv_1)
    X_conv_1 = Dropout(0.75)(X_conv_1)

    squeeze_1 = GlobalAveragePooling1D()(X_conv_1)
    squeeze_1 = Lambda(expand_dim_backend)(squeeze_1)

    excitation_1 = Conv1D(filters=16, kernel_size=1, strides=1, padding='valid', activation='relu')(squeeze_1)
    excitation_1 = Conv1D(filters=100, kernel_size=1, strides=1, padding='valid', activation='sigmoid')(excitation_1)

    X = Lambda(multiply)([X_conv_1, excitation_1])

    X_permute = Permute([2,1])(X)
    squeeze_2 = GlobalAveragePooling1D()(X_permute)
    squeeze_2 = Lambda(expand_dim_backend2)(squeeze_2)

    excitation_2 = Conv1D(filters=16, kernel_size=1, strides=1, padding='valid', activation='relu')(squeeze_2)
    excitation_2 = Conv1D(filters=51, kernel_size=1, strides=1, padding='valid', activation='sigmoid')(excitation_2)

    X2 = Lambda(multiply)([X_permute, excitation_2])
    X2 = Permute([2, 1])(X2)
    X2 = GlobalAveragePooling1D()(X2)
    X_Dense_1 = Dense(64, activation='relu')(X2)

    X_Bi_LSTM_1 = Bidirectional(LSTM(unit, return_sequences=False))(X)
    X_Bi_LSTM_1 = Dropout(0.75)(X_Bi_LSTM_1)
    X_Bi_LSTM_1 = BatchNormalization()(X_Bi_LSTM_1)
    X_Dense_2 = Dense(64, activation='relu')(X_Bi_LSTM_1)

    X_conv_2 = Conv1D(strides=10, kernel_size=15, filters=filter, padding='same')(X_input_2)
    X_conv_2 = Activation('relu')(X_conv_2)
    X_conv_2 = Dropout(0.75)(X_conv_2)

    squeeze_3 = GlobalAveragePooling1D()(X_conv_2)
    squeeze_3 = Lambda(expand_dim_backend)(squeeze_3)

    excitation_3 = Conv1D(filters=16, kernel_size=1, strides=1, padding='valid', activation='relu')(squeeze_3)
    excitation_3 = Conv1D(filters=100, kernel_size=1, strides=1, padding='valid', activation='sigmoid')(excitation_3)

    X3 = Lambda(multiply)([X_conv_2, excitation_3])

    X_Bi_LSTM_2 = Bidirectional(LSTM(unit, return_sequences=False))(X3)
    X_Bi_LSTM_2 = Dropout(0.75)(X_Bi_LSTM_2)
    X_Bi_LSTM_2 = BatchNormalization()(X_Bi_LSTM_2)
    X_Dense_3 = Dense(64, activation='relu')(X_Bi_LSTM_2)

    X3_permute = Permute([2,1])(X3)
    squeeze_4 = GlobalAveragePooling1D()(X3_permute)
    squeeze_4 = Lambda(expand_dim_backend3)(squeeze_4)

    excitation_4 = Conv1D(filters=16, kernel_size=1, strides=1, padding='valid', activation='relu')(squeeze_4)
    excitation_4 = Conv1D(filters=200, kernel_size=1, strides=1, padding='valid', activation='sigmoid')(excitation_4)

    X4 = Lambda(multiply)([X3_permute, excitation_4])
    X4 = Permute([2, 1])(X4)
    X4 = GlobalAveragePooling1D()(X4)
    X_Dense_4 = Dense(64, activation='relu')(X4)

    XX = Concatenate()([X_Dense_1,X_Dense_2,X_Dense_3,X_Dense_4])

    out = Dense(32)(XX)
    out = Activation('relu')(out)
    out = Dense(2)(out)
    out = Activation('softmax')(out)

    model_3 = Model(inputs=[X_input_1,X_input_2], outputs=[out])

    return model_3

def Deep_PSP_model_training(training_set_name, X_train_positive, X_train_negative, global_train_positive, global_train_negative, Y, X_val1, X_val2,Y_val,modelname,pretraining_model = None):

    num = int(len(X_train_negative)/len(X_train_positive))
    print(len(X_train_negative))
    print(len(X_train_positive))

    if num > 25:
        num = 25

    for i in range(num):
        inputfile = (training_set_name)
        _, input_sequence = file2str(inputfile)

        for kk in range(len(input_sequence)):
            input_sequence[kk] = input_sequence[kk].translate(str.maketrans('', '', '#'))

        globel_input_sequence = []
        for kk in range(len(input_sequence)):
            result_index = str2dic(input_sequence[kk])
            globel_input_sequence.append(result_index)
        input_sequence = pad_sequences(globel_input_sequence,maxlen = 2000)

        if len(X_train_positive) * (i + 1) < len(X_train_negative):
            X_train = np.vstack((X_train_positive, X_train_negative[len(X_train_positive) * i:len(X_train_positive) * (i + 1)]))
            X_train2 = np.vstack((input_sequence[global_train_positive], input_sequence[global_train_negative[len(X_train_positive) * i:len(X_train_positive) * (i + 1)]]))

        else:

            X_train = np.vstack((X_train_positive, X_train_negative[len(X_train_negative) - len(X_train_positive):]))
            X_train2 = np.vstack((input_sequence[global_train_positive], input_sequence[global_train_negative[len(X_train_negative) - len(X_train_positive):]]))

        print('This is the %d-th iteration'%(i+1))

        print('The number of training set is %d'%(len(X_train)),'The number of validation set is %d'%(len(X_val1)))

        Y = [0]*len(X_train_positive)+[1]*len(X_train_positive)
        Y = to_categorical(Y)

        model = create_model([51, 21],[2000,21],128,100)

        if pretraining_model == True:
            model.load_weights('model_weight\weights_pretraining')

        # model.summary()
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0)
        model.compile(loss='categorical_crossentropy',optimizer = adam,metrics = ['accuracy'])

        filepath1 = 'model_weight' + '\\' + modelname

        if not os.path.exists(filepath1):  # 判断是否存在
            os.makedirs(filepath1)  # 不存在则创建
        filepath2 = filepath1 + '\\' + 'weights%d' % (i + 1)

        # checkpoint = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
        #                              mode='min')

        checkpoint = ModelCheckpoint(filepath2, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')

        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')
        callbacks_list = [checkpoint, earlystopping]

        X_train_ = to_categorical(X_train)
        X_train2_ = to_categorical(X_train2)

        model.fit([X_train_, X_train2_],Y,validation_data=([X_val1, X_val2],Y_val),
              nb_epoch = 1000,callbacks=callbacks_list, batch_size = 1024,verbose = 2,shuffle = True)
        K.clear_session()

def Deep_PSP_model_testing(X_test, X_test_2, modelname,X_val1, X_val2,Y_val):

    weight_file = 'model_weight' +'\\'+ modelname

    X_predict_test = np.zeros((len(X_test),2*len(os.listdir(weight_file))))
    X_predict_val = np.zeros((len(X_val1), 2 *len(os.listdir(weight_file))))

    X_test = to_categorical(X_test)
    X_test_2 = to_categorical(X_test_2)
    y_val = [np.argmax(one_hot) for one_hot in Y_val]
    # len(os.listdir(weight_file))

    for i in range(len(os.listdir(weight_file))):

        print(i)

        model = create_model([51, 21],[2000,21],128,100)

        print(weight_file +'\\' + 'weights%d'%(i+1))
        model.load_weights(weight_file +'\\' + 'weights%d'%(i+1))

        X_predict_test[:,i*2:(i+1)*2] = model.predict([X_test, X_test_2])
        X_predict_val[:, i * 2:(i + 1) * 2] = model.predict([X_val1, X_val2])

        K.clear_session()

    lr = LogisticRegression(C=0.1)

    lr.fit(X_predict_val, y_val)
    predict = lr.predict(X_predict_test)
    predict_probe = lr.predict_proba(X_predict_test)

    return predict, predict_probe
