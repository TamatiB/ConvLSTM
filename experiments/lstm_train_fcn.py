import matplotlib
matplotlib.use('agg')

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
from keras import backend as K
K.set_image_dim_ordering('tf')

#i added this
#import sys
#user_home = os.path.expanduser('~')
#print(user_home)
#sys.path.insert(0, user_home + 'Documents/ConvLSTM/utils/networks.py')
#sys.path.insert(0, user_home + 'Documents/ConvLSTM/utils/data_preprocesing.py')
#import sys
#sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('.')
#from ConvLSTM import __version__
from utils.data_preprocessing import load_data, list_sequences, load_splits
from utils.networks import class_net_fcn_2p_lstm
#from utils.networks import class_net_fcn_2p_lstm



def net_summary(net):
    import sys
    from io import StringIO
    # Temporarily redirect stdout, print net summary and then restore stdout
    msg = StringIO()
    out = sys.stdout
    sys.stdout = msg
    net.summary()
    sys.stdout = out
    return msg.getvalue()


def seq_augment(seq_X, seq_y):
    # TODO: ugly hack, consider to rewrite the sequence generator so that it fits with hal
    from hal.augmentation.geometric_transforms import random_rigid_transform, apply_geometric_transform
    height, width = seq_y.shape[1:3]
    _y, _x = np.mgrid[:height, :width].astype('f')
    _x, _y = random_rigid_transform(_x, _y, scale=0.1, shift=(5.0, 5.0))
    for k in range(seq_y.shape[0]):
        frame = seq_y[k, ...].transpose([2, 0, 1])
        frame = apply_geometric_transform(frame, _x, _y, interp='nearest')
        seq_y[k, ...] = frame.transpose([1, 2, 0])
    from scipy import ndimage
    _x = .25 * ndimage.zoom(_x, 0.25, order=3)
    _y = .25 * ndimage.zoom(_y, 0.25, order=3)
    for k in range(seq_X.shape[0]):
        frame = seq_X[k, ...].transpose([2, 0, 1])
        frame = apply_geometric_transform(frame, _x, _y, interp='cubic')
        seq_X[k, ...] = frame.transpose([1, 2, 0])
    return seq_X, seq_y


class SequenceGenerator():
#class SequenceGenerator(keras.utils.Sequence):

    def __init__(self, sequences, seq_length, seq_per_seq=10, shuffle=True, step=1, jitter=True, augment=None):
        self.seq_length = seq_length
        self.step = step
        self.Xs = []
        self.ys = []
        #self.ys1 = []
        for s in sequences:
            print(s)
            X, y = load_data(s, start=0, step=1, binary=True)
            #print(X.shape)
            self.Xs.append(X)
            self.ys.append(y)
            #self.ys1.append(y)
        #print(len(self.Xs))
        self.nb_elements = seq_per_seq * len(sequences)
        self.shuffle = shuffle
        self.jitter = jitter
        self.augment = augment
        #print("Number of elements: \n")
        print(self.nb_elements)
        #print("Sequence length" + str(self.seq_length) + "\n")
        #print("Sequence per sequence" + str(seq_per_seq)+ "\n")

    def generate_batch(self, batch_size):
        #count = 0
        while True:
            #print("\n Batch number: ")
            #print(count)
            #count+= 1

            for ib in range(self.nb_elements // batch_size):
                #seq_per_seq * len sequences //batch_size
                Xar = []
                yar = []
                #y1 = []

                for j in range(batch_size):
                    if self.shuffle:
                        iter = np.random.randint(0, len(self.Xs))
                    else:
                        iter = 0
                    #print("\n Data number")
                    #print(iter)
                    seq_X = self.Xs[iter]
                    #print(seq_X.shape)
                    seq_y = self.ys[iter]
                    #seq_y1 = self.ys1[i]
                    if self.shuffle:
                        # select random frame range in dataset iter
                        s = np.random.randint(0, len(seq_X) - self.seq_length * self.step)
                    else:
                        s = ib
                    indices = np.arange(s, s + self.seq_length * self.step, self.step) #returns evenly spaced array (skipping frames according to step)
                    if self.jitter is not None:
                        # Perturb indices by introducing random time jitter
                        indices += np.random.randint(-self.jitter, self.jitter + 1, indices.shape)
                        # Clip to avoid out of bound indices
                        indices = np.clip(indices, 0, len(seq_X) - self.seq_length * self.step)
                    seq_X = seq_X[indices]
                    seq_y = seq_y[indices]
                    #seq_y1 = seq_y1[indices]
                    if self.augment:
                        seq_X, seq_X = seq_augment(seq_X, seq_X)
                    #print("\n seq in inner for loop")
                    #print(seq_X.shape)
                    #print("array apended")
                    Xar.append(seq_X)
                    #print(seq_y.shape)

                    #This was supposed to help me make it many to 1
                    #seq_y = np.expand_dims(seq_y[0], axis = 0)
                    yar.append(seq_y)
                    #print(yar)
                    #print("\n Length x: " + str(len(Xar)))
                #count += 1
                    #y1.append(seq_y1)
                #print("\n count of ib, so nb/batches ")
                #print(count)
                #print("\n X after inner for loop")
                #print(len(X))
                #print(type(X))
                #X = np.array(X).astype('f') / 255
                # dont need the 255 over here though
                #print("\n Length x: " + str(len(Xar)))
                #print(yar)
                X = np.array(Xar).astype('f') #convert to float
                y = np.array(yar)
                #print(yar)



                #print("\n X: ",X.shape)
                #print("\n Y: ",y.shape)
                #print("\n Y1: ",y1.shape)

                yield ({'input': X}, {'output': y})

    def nb_batches(self, batch_size):
        return self.nb_elements // batch_size


#class ModelCheckpointsCallback(Callback):
class ModelCheckpointsCallback():

    def __init__(self, net, output_dir):
        super().__init__()
        self.net = net
        self.output_dir = output_dir
        self.training_loss_history = []
        self.validation_loss_history = []

    def on_end_epoch(self):
        if len(self.validation_loss_history) > 0:
            if self.state.validation_loss < np.min(self.validation_loss_history):
                self.net.save_weights('%s/weights_%06d_%d.h5' % (self.output_dir, self.state.epoch, int(self.state.validation_loss)))

        self.training_loss_history.append(self.state.training_loss)
        self.validation_loss_history.append(self.state.validation_loss)


def generate_results(net, generator, output_dir, out_prefix):
    for i, b in enumerate(generator.generate_batch(1)):
        img = [x for x in np.squeeze(b.input)]
        t = [x for x in np.squeeze(b.output.astype('f'))]
        r = [x for x in np.squeeze(net.predict(b.input))]
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.subplot(3, 1, 1)
        plt.imshow(np.hstack(img), cmap='gray')
        plt.subplot(3, 1, 2)
        plt.imshow(np.hstack(t), cmap='gray')
        plt.subplot(3, 1, 3)
        plt.imshow(np.hstack(r), cmap='gray')
        plt.savefig('%s/%s_%03d.png' % (output_dir, out_prefix, i))
        plt.close()


def train_model(network, sequences, sequences_test, nb_epochs=10, seq_step=2, seq_per_seq=10,
                output_dir=None, file_suffix='', jitter=None, augment=None):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Setup data generators
    train = SequenceGenerator(sequences, seq_length = 10)
    #val = SequenceGenerator(sequences_test, seq_length, seq_per_seq=seq_per_seq, step=seq_step)
    input_shape = [None, ] + list(train.Xs[0].shape[1:])
    print("The input shape for the model is : " + str(input_shape))
    # batch size should be a number your data is divisible by
    a = train.generate_batch(2)
    #v = val.generate_batch(2)
    #print(type(a))
    #for i in a:
        #print (i)
    #next(a)
    #print("Training samples    " + ltrain) + "\n")
    # Setup model and train
    model = network(input_shape)
    net = model
    print(net_summary(net))

    #model.fit(train, nb_epochs=nb_epochs, batch_size=1, val_data_generator=val)
    #model.fit_generator(a, epochs=nb_epochs, steps_per_epoch=1, validation_data=val)
    model.fit_generator(a, epochs=nb_epochs, steps_per_epoch=15, verbose=1 )
    #model.fit_generator(generator = train, validation_data=val, use_multiprocessing=True, workers=6)
    #print((sequences_test[0]).shape())
    #res = model.predict(load_data(sequences_test[0]))

    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    #res2 = model.predict_generator(v, steps=5)

    #print(type(res))
    #print(shape(res))

    #model.evaluate_generator(v, steps=None, max_queue_size=10)

    #loss = model._state.validation_loss

    # Save output
    # if output_dir is not None:
    #     net.save_weights(os.path.join(output_dir, 'weights' + file_suffix + '.h5'), overwrite=True)
    #     plt.ylim([0, 100000])
    #     plt.grid(True)
    #     plt.savefig(os.path.join(output_dir, 'loss' + file_suffix + '.png'))
    #     plt.close()
    #     generate_results(net, train, output_dir, 'train' + file_suffix)
    #     generate_results(net, val, output_dir, 'val' + file_suffix)

    #return net, loss

if __name__ == '__main__':
    # Load data split

    input_dir = os.path.expanduser('~/Documents/ConvLSTM/Databinary/')
    print("Fetching data from " + input_dir)
    #split = load_splits(os.path.expanduser('~/Documents/ConvLSTM/Data/split.txt'))[0]
    #sequences = [os.path.join(input_dir, f) for f in split['train']]
    #sequences_test = [os.path.join(input_dir, f) for f in split['test']]
    sequences = [input_dir + 'substack500_7fps_01.avi',
    input_dir + 'substack500_7fps_02.avi',
    input_dir + 'substack500_7fps_03.avi',
    input_dir + 'substack500_7fps_04.avi',
    input_dir + 'substack500_7fps_05.avi',
    input_dir + 'substack500_7fps_06.avi',
    input_dir + 'substack500_7fps_07.avi',
    input_dir + 'substack500_7fps_08.avi',
    input_dir + 'substack500_7fps_09.avi',
    input_dir + 'substack500_7fps_10.avi']
    #sequences = [input_dir + '/substack500_7fps_0.avi']
    sequences_test = [input_dir + 'substack500_7fps_00.avi', input_dir + 'substack500_7fps_01.avi']
    print("Number of sequences in train set " + str(len(sequences)))
    print("NUmber of sequences in test set " + str(len(sequences_test)))

    network = class_net_fcn_2p_lstm  # input_shape = (None, 96, 108, 1)
    train_model(network, sequences, sequences_test, output_dir='tmp')
