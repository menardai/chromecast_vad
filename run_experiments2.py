import numpy as np
import datetime
import glob

from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import load_model

from vad_model import VadModel
from dataset import SpectrogramDataGenerator


def start_experiments():
    # load and split dataset
    x_filenames = sorted(glob.glob('data/dev_set/x_spectrogram_*.npy'))
    y_filenames = sorted(glob.glob('data/dev_set/y_*.npy'))

    print('number of samples (to split in train/val) =', len(x_filenames))
    print(x_filenames[0])
    print(y_filenames[0])

    X_filename_train, X_filename_val, Y_filename_train, Y_filename_val = train_test_split(
        x_filenames, y_filenames, test_size=0.10, random_state=42)

    batch_size = 500
    training_generator = SpectrogramDataGenerator(X_filename_train, Y_filename_train, batch_size)
    val_generator = SpectrogramDataGenerator(X_filename_val, Y_filename_val, batch_size)

    dropout_rates = [
        [0.25, 0.50, 0.50],
        [0.25, 0.25, 0.80],
        #[0.40, 0.60, 0.60],
        #[0.60, 0.30, 0.30],
        #[0.60, 0.30, 0.30],
        #[0.15, 0.50, 0.50],
        #[0.15, 0.70, 0.70],
        #[0.15, 0.70, 0.30],
        #[0.50, 0.50, 0.50],
        #[0.30, 0.80, 0.50],
    ]

    model_weights = [
        'many_to_one_2018-11-26_13h33_d0.25_0.5_0.5_u128_256_0',
        'many_to_one_2018-11-26_14h48_d0.25_0.25_0.8_u128_256_0',
    ]

    units_list = [
        {'lstm': 128, 'dense': 256},
    ]

    for i, dropout_rate in enumerate(dropout_rates):
        for units in units_list:

            vad = VadModel(dropout_rates=dropout_rate, lstm_units=units['lstm'], dense_units=units['dense'])

            vad.model = load_model('models/{}.h5'.format(model_weights[i]))

            lr_list = [0.000075]
            for lr_index, lr in enumerate(lr_list):

                opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.01)
                experiment_name = vad.train(50, training_generator, val_generator, opt, units, vad, dropout_rate, lr_index)

                vad.save('models/{}.h5'.format(experiment_name))


if __name__ == '__main__':
    start_experiments()
