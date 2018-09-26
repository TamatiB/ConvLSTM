""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
import numpy as np
import pylab as plt

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.
def convlstm_unet():
    input_shape = (None, 256, 400, 1)
    input = Input(input_shape, name='input')

    x = ConvLSTM2D(filters=20, kernel_size=(3, 3),
                       input_shape=input_shape,
                       padding='same', return_sequences=True)(input)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=20, kernel_size=(3, 3),
                       padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=20, kernel_size=(3, 3),
                       padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)

    x = TimeDistributed(Conv2D(filters=2, kernel_size=(3,3), padding='same',activation='relu'))(x)
    output = TimeDistributed(Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid'), name='output')(x)
    # output = Conv3D(filters=1, kernel_size=(3, 3, 3),
    #                    activation='sigmoid',
    #                    padding='same', data_format='channels_last')(x)


    model = Model(inputs = [input], output=[output])
    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    return model


noisy_movies = np.load("image_sequences.npy")
shifted_movies = np.load("labels.npy")

print("Movies loaded")

model = convlstm_unet()

print("Model loaded")
print("Fitting to model...")

model.fit(noisy_movies[:800], shifted_movies[:800], batch_size=10,
        epochs=50, validation_split=0.05)

# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
which = 900
track = noisy_movies[which][:7, ::, ::, ::]

for j in range(16):
    new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)


# # And then compare the predictions
# # to the ground truth
# track2 = noisy_movies[which][::, ::, ::, ::]
# for i in range(15):
#     fig = plt.figure(figsize=(10, 5))
#
#     ax = fig.add_subplot(121)
#
#     if i >= 7:
#         ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
#     else:
#         ax.text(1, 3, 'Initial trajectory', fontsize=20)
#
#     toplot = track[i, ::, ::, 0]
#
#     plt.imshow(toplot)
#     ax = fig.add_subplot(122)
#     plt.text(1, 3, 'Ground truth', fontsize=20)
#
#     toplot = track2[i, ::, ::, 0]
#     if i >= 2:
#         toplot = shifted_movies[which][i - 1, ::, ::, 0]
#
#     plt.imshow(toplot)
#     plt.savefig('%i_animate.png' % (i + 1))
