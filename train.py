import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import cv2
import pickle

import random
import glob
import os

from multi_label_dataset import MultiLabelDatasetGenerator
from fashion_net import FashionNet

random.seed(42)

#------------------------------------------------------
# model params
BS = 32

HEIGHT = 224
WIDTH  = 224
EPOCHS = 10
INIT_LR = 1e-4
#------------------------------------------------------


#------------------------------------------------------
# read data
img_types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
img_paths = [ img_path for img_type in img_types for img_path in glob.glob(f'dataset/*/{img_type}') ]
img_paths = np.array(img_paths)

random.shuffle(img_paths)

labels = [img_path.split(os.path.sep)[-2].split('_') for img_path in img_paths]

#------------------------------------------------------
# encode label to one-hot
enc = OneHotEncoder(handle_unknown='ignore')

color_label = np.array(labels)[:, 0]
color_label = np.expand_dims(color_label, axis=1)
color_label = enc.fit_transform(color_label).toarray()
print(enc.get_feature_names())

#------------------------------------------------------
# save color encoder
print("[INFO] save color encoder network...")
with open('model/color_en.pickle','wb') as f:
    f.write(pickle.dumps(enc))
    
enc = OneHotEncoder(handle_unknown='ignore')
type_label = np.array(labels)[:, 1]
type_label = np.expand_dims(type_label, axis=1)
type_label = enc.fit_transform(type_label).toarray()
print(enc.get_feature_names())

#------------------------------------------------------
# save type encoder
print("[INFO] save type encoder network...")
with open('model/type_en.pickle','wb') as f:
    f.write(pickle.dumps(enc))

#------------------------------------------------------
# count n_class
n_color =  len(color_label[0])
n_type = len(type_label[0])

#------------------------------------------------------
# split train/test data
indexes = np.arange(len(img_paths))
np.random.shuffle(indexes)

split_point = int(len(img_paths) * 0.8)

train_indexes = indexes[:split_point]
test_indexes = indexes[split_point:]

train_X = img_paths[train_indexes]
test_X  = img_paths[test_indexes]

train_Y = [color_label[train_indexes], type_label[train_indexes]] 
test_Y  = [color_label[test_indexes], type_label[test_indexes]]

#------------------------------------------------------
# load data
train_gen = MultiLabelDatasetGenerator(X=train_X, Y=train_Y, batch_size=BS, 
                                  target_size=(HEIGHT, WIDTH), mode='train')

test_gen = MultiLabelDatasetGenerator(X=test_X, Y=test_Y, batch_size=BS, 
                                 target_size=(HEIGHT, WIDTH), mode='val')

#------------------------------------------------------
# load model
model = FashionNet(HEIGHT, WIDTH, n_color, n_type)
model = model.build()
model.summary()

model.compile(loss={'color': 'categorical_crossentropy', 'type': 'categorical_crossentropy'},
              optimizer=tf.keras.optimizers.RMSprop(lr=INIT_LR),
              metrics=['accuracy'])

#------------------------------------------------------
# train model
history = model.fit(
            train_gen,
            validation_data=test_gen,
            steps_per_epoch=len(train_X) // BS,
            epochs=EPOCHS, verbose=1)

#------------------------------------------------------
# save model
print("[INFO] serializing network...")
model.save('model/fashion_net.h5', save_format="h5")
