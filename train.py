
from __future__ import division
from __future__ import print_function

from model import get_unet_3d
from util import VolSegDataManager
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import os

cwd = os.getcwd()

steps_per_epoch = 1
epochs = 1

checkpoint_path = cwd + '/save/weights.{epoch:02d}-{val_loss:.2f}.hdf5'

#class_weights = {0: 0., 1: 1., 2: 1., 3: 1.}

dims = 176, 208, 176, 1

model = get_unet_3d(*dims)
data = VolSegDataManager(cwd + '/data/scan',
                         cwd + '/data/seg',
                         vol_prefix='scan_',
                         seg_prefix='seg_',
                         test_fract=0.34)
data.data_aug = True

generator_train = data.get_train_batch_generator()
generator_test = data.get_test_batch_generator()

mod_checkpt = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0,
                                save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)

model.fit_generator(generator_train, steps_per_epoch, epochs=epochs, verbose=1,
                    callbacks=[mod_checkpt], validation_data=generator_test, validation_steps=1,
                    class_weight=None, max_q_size=1, workers=1, pickle_safe=False, initial_epoch=0)

print('done')
