from tensorflow.keras.models import load_model
import cv2

import argparse
import pickle

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('img_path', type=str, help='input img path')
parser.add_argument('model_path', type=str, help='input model path')

args = parser.parse_args()

img_path = args.img_path
model_path = args.model_path


#------------------------------------------------------
# read data
img = read_img(img_path)
np_img = np.expand_dims(img, axis=0)

#------------------------------------------------------
# load model
print("[INFO] loading network...")
model = load_model(model_path)

#------------------------------------------------------
# predict
print("[INFO] classifying image...")
proba_list = model(np_img)

#------------------------------------------------------
# draw_result
img = draw_result(img, proba_list)
cv2.imshow('Output', img)
cv2.waitKey(0)
