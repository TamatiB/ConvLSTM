from keras.models import Sequential, Model
from keras.layers import (ConvLSTM2D, BatchNormalization, Convolution3D, Convolution2D,Conv2D,
                          TimeDistributed, MaxPooling2D, UpSampling2D, Input, merge, Cropping2D, concatenate)

from utils.keras_extensions import (categorical_crossentropy_3d_w, softmax_3d, softmax_2d)
from keras import losses
from keras.optimizers import Adam

def binary_net(input_shape):
    net = Sequential()
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3, input_shape=input_shape,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(Convolution3D(nb_filter=1, kernel_dim1=1, kernel_dim2=3,
                          kernel_dim3=3, activation='sigmoid',
                          border_mode='same', dim_ordering='tf'))
    net.compile(loss='binary_crossentropy', optimizer='adadelta')
    return net


def class_net(input_shape):
    c = 3
    net = Sequential()
    net.add(ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, input_shape=input_shape,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=4 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=8 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(Convolution3D(nb_filter=3, kernel_dim1=1, kernel_dim2=3,
                          kernel_dim3=3, activation=softmax_3d(class_dim=-1),
                          border_mode='same', dim_ordering='tf'))
    net.compile(loss=categorical_crossentropy_3d_w(4, class_dim=-1), optimizer='adadelta')
    return net


def class_net_ms(input_shape):
    c = 12
    net = Sequential()
    net.add(ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, input_shape=input_shape,
                       border_mode='same', return_sequences=True))
    net.add(ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(TimeDistributed(MaxPooling2D((2, 2), (2, 2))))

    net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    # net.add(TimeDistributed(MaxPooling2D((2, 2), (2, 2))))

    net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    # net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
    #                    border_mode='same', return_sequences=True))
    # net.add(TimeDistributed(UpSampling2D((2, 2))))
    # net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
    #                    border_mode='same', return_sequences=True))

    net.add(TimeDistributed(UpSampling2D((2, 2))))
    net.add(Convolution3D(nb_filter=3, kernel_dim1=1, kernel_dim2=3,
                          kernel_dim3=3, activation=softmax_3d(class_dim=-1),
                          border_mode='same', dim_ordering='tf'))
    net.compile(loss=categorical_crossentropy_3d_w(4, class_dim=-1), optimizer='adadelta')
    return net


def class_net_fcn_1p(input_shape):
    c = 12
    input_img = Input(input_shape, name='input')

    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(input_img)
    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)
    c1 = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    x = TimeDistributed(Convolution2D(2*c, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Convolution2D(2*c, 3, 3, border_mode='same'))(x)
    c2 = TimeDistributed(Convolution2D(2*c, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(c2)
    x = merge([c1, x], mode='concat')
    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)
    output = Convolution3D(nb_filter=3, kernel_dim1=1, kernel_dim2=3,
                           kernel_dim3=3, activation=softmax_3d(class_dim=-1),
                           border_mode='same', dim_ordering='tf', name='output')(x)
    model = Model(input_img, output=[output])
    model.compile(loss=categorical_crossentropy_3d_w(4, class_dim=-1), optimizer='adadelta')

    return model


def class_net_fcn_1p_lstm(input_shape):
    c = 12
    input_img = Input(input_shape, name='input')
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(input_img)
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c1 = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    x = ConvLSTM2D(nb_filter=2*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    x = ConvLSTM2D(nb_filter=2*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c2 = ConvLSTM2D(nb_filter=2*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(c2)
    x = merge([c1, x], mode='concat')

    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)

    # x = TimeDistributed(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 396, 440), border_mode='valid'))(x)

    output = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same', activation=softmax_2d(-1)), name='output')(x)

    model = Model(input_img, output=[output])
    model.compile(loss=categorical_crossentropy_3d_w(2, class_dim=-1), optimizer='adadelta')
    return model


def class_net_fcn_2p_lstm(input_shape):
    #This is not a sequential model because sequential models are specifically for
    # linear sequences
    c = 12
    input_img = Input(input_shape, name='input')
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(input_img)
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c1 = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c2 = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c3 = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(c3)
    x = merge([c2, x], mode='concat')
    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = merge([c1, x], mode='concat')
    #x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)

    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)

    #output = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same', activation=softmax_2d(-1)), name='output')(x)
    #model = Model(input_img, output=[output])
    #model.compile(loss=categorical_crossentropy_3d_w(2, class_dim=-1), optimizer='adam')

    x = TimeDistributed(Convolution2D(2, 3, 3, border_mode='same'))(x)
    output = TimeDistributed(Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid'), name='output')(x)
    model = Model(input_img, output=[output])
    model.compile(loss='binary_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])
    return model


# def class_net_fcn_2p_lstm(input_shape):
#     #This is not a sequential model because sequential models are specifically for
#     # linear sequences
#     c = 12
#     input_img = Input(input_shape, name='input')
#     x = ConvLSTM2D(nb_filter=12, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(input_img)
#     x = ConvLSTM2D(nb_filter=12, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#     c1 = ConvLSTM2D(nb_filter=12, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#
#     x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)
#
#     x = ConvLSTM2D(nb_filter=24, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#     x = ConvLSTM2D(nb_filter=24, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#     c2 = ConvLSTM2D(nb_filter=24, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#
#     x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
#     x = ConvLSTM2D(nb_filter=24, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#     x = ConvLSTM2D(nb_filter=24, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#     c3 = ConvLSTM2D(nb_filter=24, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#
#     x = TimeDistributed(UpSampling2D((2, 2)))(c3)
#     x = merge([c2, x], mode='concat')
#     x = TimeDistributed(Convolution2D(2, 3, 3, border_mode='same'))(x)
#
#     x = TimeDistributed(UpSampling2D((2, 2)))(x)
#     x = merge([c1, x], mode='concat')
#     # x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)
#
#
#     x = TimeDistributed(UpSampling2D((2, 2)))(x)
#     x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
#     x = TimeDistributed(UpSampling2D((2, 2)))(x)
#
#     output = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same', activation=softmax_2d(-1)), name='output')(x)
#     model = Model(input_img, output=[output])
#     model.compile(loss=categorical_crossentropy_3d_w(2, class_dim=-1), optimizer='adadelta')
#
#     #x = TimeDistributed(Convolution2D(2, 3, 3, border_mode='same'))(x)
#     #output = TimeDistributed(Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid'), name='output')(x)
#     #model = Model(input_img, output=[output])
#     #model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics = ['accuracy'])
#     return model


def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)


# def class_net_fcn_2p_lstm(input_shape):
#     #This is not a sequential model because sequential models are specifically for
#     # linear sequences
#     #c = 12
#     concat_axis = 3
#     input_img = Input(input_shape, name='input')
#
#     x = ConvLSTM2D(nb_filter=16, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(input_img)
#     c1 = ConvLSTM2D(nb_filter=16, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#     x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)
#
#     x = ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#     c2 = ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#     x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
#
#     x = ConvLSTM2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#     c3 = ConvLSTM2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#     x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c3)
#
#     x = ConvLSTM2D(nb_filter=128, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#     c4 = ConvLSTM2D(nb_filter=128, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#     x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c4)
#
#     c5 = ConvLSTM2D(nb_filter=256, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
#
#     up_samp6 = TimeDistributed(UpSampling2D((2, 2)))(c5)
#     up_samp6 = TimeDistributed(Convolution2D(128, 2, 2, border_mode='same'))(up_samp6)
#
#     x = TimeDistributed(Convolution2D(128, 3, 3, border_mode='same'))(up_samp6)
#     x = TimeDistributed(Convolution2D(128, 3, 3, border_mode='same'))(x)
#
#     up_samp = TimeDistributed(UpSampling2D((2, 2)))(x)
#     up_samp = TimeDistributed(Convolution2D(64, 2, 2, border_mode='same'))(up_samp)
#
#     x = TimeDistributed(Convolution2D(64, 3, 3, border_mode='same'))(up_samp)
#     x = TimeDistributed(Convolution2D(64, 3, 3, border_mode='same'))(x)
#
#     up_samp = TimeDistributed(UpSampling2D((2, 2)))(x)
#     up_samp = TimeDistributed(Convolution2D(32, 2, 2, border_mode='same'))(up_samp)
#
#     x = TimeDistributed(Convolution2D(32, 3, 3, border_mode='same'))(up_samp)
#     x = TimeDistributed(Convolution2D(32, 3, 3, border_mode='same'))(x)
#     # up_samp = TimeDistributed(UpSampling2D((2, 2)))(x)
#     # up_samp = TimeDistributed(Convolution2D(64, 2, 2, border_mode='same'))(up_samp)
#     #
#     # x = TimeDistributed(Convolution2D(64, 3, 3, border_mode='same'))(x)
#     # x = TimeDistributed(Convolution2D(64, 3, 3, border_mode='same'))(x)
#     x = TimeDistributed(Convolution2D(2, 3, 3, border_mode='same'))(x)
#     x = TimeDistributed(UpSampling2D((2, 2)))(x)
#
#     #output = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same', activation=softmax_2d(-1)), name='output')(x)
#     #model = Model(input_img, output=[output])
#     #model.compile(loss=categorical_crossentropy_3d_w(2, class_dim=-1), optimizer='adam')
#
#     #x = TimeDistributed(Convolution2D(2, 3, 3, border_mode='same'))(x)
#     output = TimeDistributed(Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid'), name='output')(x)
#     model = Model(input_img, output=[output])
#     model.compile(loss='binary_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
#     return model
