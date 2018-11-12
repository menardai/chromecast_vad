import numpy as np
import datetime
import glob

from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from vad_model import VadModel
from dataset import SpectrogramDataGenerator


def train_10000_split_samples():
    # load and split dataset
    x_filenames = sorted(glob.glob('/media/ai/backup/datasets/voice_activity_detection/dev_set/x_spectrogram_*.npy'))
    y_filenames = sorted(glob.glob('/media/ai/backup/datasets/voice_activity_detection/dev_set/y_*.npy'))

    print('number of samples (to split in train/val) =', len(x_filenames))
    print(x_filenames[0])
    print(y_filenames[0])

    X_filename_train, X_filename_val, Y_filename_train, Y_filename_val = train_test_split(
        x_filenames, y_filenames, test_size=0.10, random_state=42)

    batch_size = 250
    training_generator = SpectrogramDataGenerator(X_filename_train, Y_filename_train, batch_size)
    val_generator = SpectrogramDataGenerator(X_filename_val, Y_filename_val, batch_size)

    # load model
    vad = VadModel()
    #vad = VadModel(architecture_filename='models/model_architecture.json')

    # compile
    #lr_list = [0.00005, 0.0001, 0.0002, 0.0005]
    lr_list = [0.0005]
    for lr in lr_list:
        opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.01)
        vad.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

        experiment_name = 'gru_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M"), lr)
        tbCallBack = TensorBoard(log_dir='./logs/'+experiment_name,
                                 histogram_freq=0, write_graph=False, write_images=False)

        # train
        vad.model.fit_generator(
            epochs=75,

            generator=training_generator,
            steps_per_epoch=len(training_generator),

            validation_data=val_generator,
            validation_steps=len(val_generator),

            callbacks=[tbCallBack])

        vad.model.save_weights("models/vad_09_11_2018_weights.h5")


if __name__ == '__main__':
    train_10000_split_samples()
