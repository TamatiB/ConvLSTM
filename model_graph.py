
from keras.utils import plot_model
from utils.networks import class_net_fcn_2p_lstm, binary_net
from utils import unet_model
network = class_net_fcn_2p_lstm
model = network([None, 96 , 108, 1])
print(model.summary())

#model = unet_model.unet()
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# # Many to one lstm block but this may just be numbers
#   model = Sequential()
#     model.add(LSTM(1, input_shape=(timesteps, data_dim)))
