import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Conv2D, Flatten, Dropout, Dense, Input
from tensorflow.keras.models import Sequential

# https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/
# https://kapernikov.com/multi-label-classification-with-keras/
class FashionNet():
    
    def __init__(self, height, width, n_color, n_type):
        
        self.shape = (height, width, 3)
        
        self.feature_extractor = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, 
                                              input_shape=(height, width, 3))
        self.feature_extractor.trainable = False
        
        self.flatten_layer = Flatten()
        
        self.color_model = Sequential([
                                Dense(128),
                                BatchNormalization(),
                                Activation('relu'),
                                Dense(n_type, activation='softmax')
                            ], name='color')
        
        self.type_model = Sequential([
                                Dense(256),
                                BatchNormalization(),
                                Activation('relu'),
                                Dense(n_type, activation='softmax')
                            ], name='type')
        
    def build(self):
        
        input_images = tf.keras.Input(shape=self.shape, dtype='float32', name='images')
        
        feature_vector = self.feature_extractor(input_images)
        feature_vector = self.flatten_layer(feature_vector)
        
        color_output = self.color_model(feature_vector)
        type_output = self.type_model(feature_vector)
        
        model = Model(input_images, [color_output, type_output])
        
        return model