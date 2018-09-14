from keras.models import model_from_json
from experiments.lstm_train_fcn import SequenceGenerator
from utils.networks import class_net_fcn_2p_lstm
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np


def display_image(image, size):
    fig = plt.figure(figsize=size)
    x = len(image)
    y = len(image[0])
    img = np.zeros((x, y))
    for row in range(x):
        for col in range(y):
            img[row,col] = image[row][col][0]

    plt.imshow(img, cmap = 'gray')
    plt.show()


network = class_net_fcn_2p_lstm
model = network([None, 192, 216, 1])
model.load_weights("model.h5")
print("Loaded model from disk")


sequences_test = ['/home/pelonomi/Documents/ConvLSTM/Data/substack500_7fps_2.avi']
val = SequenceGenerator(sequences_test, seq_length=10, seq_per_seq=5, step=1)
v = val.generate_batch(1)
res = model.predict_generator(v, steps=1)

# #1d imaging
# item_count = 0
# image_count = 0
# size = (5,5)
# for item in res:
#     for image in item:
#         display_image(image, size)
#         image_count +=1
#         print(item_count,image_count)
#     item_count +=1

size = (18,18)
image = res[0][5]

print("Prediction made")
x = len(image)
y = len(image[0])
img = np.zeros((x, y))
for row in range(x):
    for col in range(y):
        img[row,col] = image[row][col][0]

plt.figure(figsize=size)
plt.imshow(img, cmap = 'gray')
plt.show()
#display_image(res[0][5],size )
