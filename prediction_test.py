

from experiments.lstm_train_fcn import SequenceGenerator


sequences_test = ['/home/pelonomi/Documents/ConvLSTM/Data/substack500_7fps_2.avi']
val = SequenceGenerator(sequences_test, seq_length=10, seq_per_seq=10, step=5)
v = val.generate_batch(2)


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
