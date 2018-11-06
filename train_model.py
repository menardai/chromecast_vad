import numpy as np
import datetime

from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from vad_model import VadModel


def load_dataset(x_filename, y_filename):
    print('loading dataset X ...')
    X = np.load(x_filename)
    print('loading dataset Y ...')
    Y = np.load(y_filename)
    return X, Y

# load and split dataset
X, Y = load_dataset('data/dev_set_2500_x.npy', 'data/dev_set_2500_y.npy')
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20, random_state=42)

# load model
vad = VadModel(architecture_filename='models/model_architecture.json',
               weights_filename='models/vad_06_11_2018_weights_250_0005.h5')

# compile
#lr_list = [0.00005, 0.0001, 0.0002, 0.0005]
lr_list = [0.0005]
for lr in lr_list:
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.01)
    vad.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    experiment_name = 'gru_{}_lr_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M"), lr)
    tbCallBack = TensorBoard(log_dir='./logs/'+experiment_name,
                             histogram_freq=0, write_graph=False, write_images=False)

    # train
    vad.model.fit(X_train, Y_train, batch_size=200, epochs=100, validation_data=(X_val, Y_val), callbacks=[tbCallBack])
    vad.model.save_weights("models/vad_06_11_2018_weights_350_0005.h5")
