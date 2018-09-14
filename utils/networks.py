from keras.models import Sequential, Model
from keras.layers import (ConvLSTM2D, BatchNormalization, Convolution3D, Convolution2D,Conv2D,
                          TimeDistributed, MaxPooling2D, UpSampling2D, Input, merge, Cropping2D, concatenate)

from utils.keras_extensions import (categorical_crossentropy_3d_w, softmax_3d, softmax_2d)
from keras import losses
from keras.optimizers import Adam



def class_net_fcn_2p_lstm(input_shape):
    #This is not a sequential model because sequential models are specifically for
    # linear sequences
    c = 12
    input_img = Input(input_shape, name='input')

    # 192 216 12
    x = ConvLSTM2D(filters=c, kernel_size=(3,3), padding='same', return_sequences=True,activation='tanh')(input_img)
    x = ConvLSTM2D(filters=c, kernel_size=(3,3), padding='same', return_sequences=True,activation='tanh')(x)
    c1 = ConvLSTM2D(filters=c, kernel_size=(3,3), padding='same', return_sequences=True,activation='tanh')(x)

    # pool: 96 108 24
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)
    x = ConvLSTM2D(filters=(2 * c), kernel_size=(3,3), padding='same', return_sequences=True,activation='tanh')(x)
    x = ConvLSTM2D(filters=(2 * c), kernel_size=(3,3), padding='same', return_sequences=True,activation='tanh')(x)
    c2 = ConvLSTM2D(filters=(2 * c), kernel_size=(3,3), padding='same', return_sequences=True,activation='tanh')(x)

    # pool: 48 54 24
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
    x = ConvLSTM2D(filters=(2 * c), kernel_size=(3,3), padding='same', return_sequences=True,activation='tanh')(x)
    x = ConvLSTM2D(filters=(2 * c), kernel_size=(3,3), padding='same', return_sequences=True,activation='tanh')(x)
    c3 = ConvLSTM2D(filters=(2 * c), kernel_size=(3,3), padding='same', return_sequences=True,activation='tanh')(x)

    # pool: 24 27 24
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c3)
    x = ConvLSTM2D(filters=(4 * c), kernel_size=(3,3), padding='same', return_sequences=True,activation='tanh')(x)
    x = ConvLSTM2D(filters=(4 * c), kernel_size=(3,3), padding='same', return_sequences=True,activation='tanh')(x)
    c4 = ConvLSTM2D(filters=(4 * c), kernel_size=(3,3), padding='same', return_sequences=True,activation='tanh')(x)

    # upsl: 48 54 24
    upsl_1 = TimeDistributed(UpSampling2D((2, 2)))(c4)
    #x = TimeDistributed(Conv2D(filters=c, kernel_size=(2,2), padding='same',activation='relu'))(x)
    x = merge([c3, upsl_1], mode='concat')
    x = TimeDistributed(Conv2D(filters=c, kernel_size=(3,3), padding='same',activation='relu'))(x)
    x = TimeDistributed(Conv2D(filters=c, kernel_size=(3,3), padding='same',activation='relu'))(x)

    # upsl: 96 108 24
    upsl_2 = TimeDistributed(UpSampling2D((2, 2)))(x)
    #x = TimeDistributed(Conv2D(filters=c, kernel_size=(2,2), padding='same',activation='relu'))(x)
    x = merge([c2, upsl_2], mode='concat')
    x = TimeDistributed(Conv2D(filters=3, kernel_size=(3,3), padding='same',activation='relu'))(x)
    x = TimeDistributed(Conv2D(filters=3, kernel_size=(3,3), padding='same',activation='relu'))(x)

    # upsl: 192 216 24
    upsl_3 = TimeDistributed(UpSampling2D((2, 2)))(x)
    #x = TimeDistributed(Conv2D(filters=c, kernel_size=(2,2), padding='same',activation='relu'))(x)
    x = merge([c1, upsl_3], mode='concat')
    x = TimeDistributed(Conv2D(filters=3, kernel_size=(3,3), padding='same',activation='relu'))(x)
    x = TimeDistributed(Conv2D(filters=3, kernel_size=(3,3), padding='same',activation='relu'))(x)


    # upsl: 384 432
    upsl_4 = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(filters=2, kernel_size=(3,3), padding='same',activation='relu'))(upsl_4)
    output = TimeDistributed(Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid'), name='output')(x)
    model = Model(input_img, output=[output])
    model.compile(loss='binary_crossentropy', optimizer = 'adadelta', metrics = ['binary_accuracy'])
    return model
