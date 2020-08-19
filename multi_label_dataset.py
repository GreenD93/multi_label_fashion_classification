from random import randint
import argparse

import cv2
import numpy as np

from tensorflow.keras.utils import Sequence
import tensorflow as tf

#------------------------------------------------------
# MultiDatasetGenerator
# https://hwiyong.tistory.com/241
class MultiLabelDatasetGenerator(Sequence):

    #---------------------------------------------
    # constructor
    def __init__(self, X, Y, batch_size=16, target_size=(224,224), 
                 mode='train', shuffle=True):

        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle

        self.img_paths = X.tolist()
        self.label_list = Y
        self.label_cnt = len(self.label_list[0])
        
        if mode == 'train':
            print('train gen data length : {0}'.format(len(self.img_paths)))
            self.aug = True
        else:
            print('val gen data length : {0}'.format(len(self.img_paths)))
            self.aug = False
        
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.img_paths)) / self.batch_size)

    def __data_generation(self, indexes):

        X = np.empty((self.batch_size, *self.target_size, 3))
        y = [self.label_list[0][indexes], self.label_list[1][indexes]]

        # Generate data
        for i, idx in enumerate(indexes):

            img_path = self.img_paths[idx]

            img = self._read_img(img_path)

            # data augmentation
            if self.aug:
                img = self._get_augmentated_img(img)

            X[i, ] = img

        return X, y

    def __getitem__(self, index):
        # Generate one batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def _read_img(self, img_path):
        
        # bgr to rgb
        img = cv2.imread(img_path)[:,:,::-1]
        
        # dowm sampling -> INTER_AREA
        img = cv2.resize(img, dsize=self.target_size, interpolation=cv2.INTER_AREA)

        # normalization
        img = np.array(img, dtype="float") / 255.0

        return img

    def _get_augmentated_img(self, img,
                              rotation_range=25,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_intensity=0.2,
                              zoom_range=0.8,
                              horizontal_flip=True,
                              vertical_flip=True):


        img = tf.keras.preprocessing.image.random_rotation(img, rg=30, 
                                                           row_axis=0, col_axis=1, channel_axis=2)

        img = tf.keras.preprocessing.image.random_shift(img, 0.1, 0.1, 
                                                        row_axis=0, col_axis=1, channel_axis=2)

        img = tf.keras.preprocessing.image.random_shear(img, intensity=shear_intensity, 
                                                        row_axis=0, col_axis=1, channel_axis=2)

        img = tf.keras.preprocessing.image.random_zoom(img, zoom_range=(zoom_range, zoom_range), 
                                                       row_axis=0, col_axis=1, channel_axis=2)

        if horizontal_flip and randint(0,1):
            img = np.flip(img, axis=0)
            
#         if vertical_flip and randint(0,1):
#             img = np.flip(img, axis=1)
            
        return img
