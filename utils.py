import matplotlib.pyplot as plt
import numpy as np
import cv2

import pickle

TARGET_SIZE = (224,224)

color_en = pickle.loads(open('model/color_en.pickle','rb').read())
type_en = pickle.loads(open('model/type_en.pickle','rb').read())

color_en_list = color_en.get_feature_names()
type_en_list = type_en.get_feature_names()

def plot_training_history(history):
    color_acc = history.history['color_accuracy']
    val_color_acc = history.history['val_color_accuracy']
    color_loss = history.history['color_loss']
    val_color_loss = history.history['val_color_loss']
    type_acc = history.history['type_accuracy']
    val_type_acc = history.history['val_type_accuracy']
    type_loss = history.history['type_loss']
    val_type_loss = history.history['val_type_loss']
    epochs = range(len(color_acc))
 
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, color_acc, 'bo', label='Color training acc')
    plt.plot(epochs, val_color_acc, 'b', label='Color validation acc')
    plt.plot(epochs, type_acc, 'ro', label='Type training acc')
    plt.plot(epochs, val_type_acc, 'r', label='Type validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
 
    plt.subplot(1, 2, 2)
    plt.plot(epochs, color_loss, 'bo', label='Color training loss')
    plt.plot(epochs, val_color_loss, 'b', label='Color validation loss')
    plt.plot(epochs, type_loss, 'ro', label='Type training loss')
    plt.plot(epochs, val_type_loss, 'r', label='Type validation loss')
    plt.title('Training and validation loss')
    plt.legend()
 
    plt.show()
    
def read_img(img_path):

    # bgr to rgb
    img = cv2.imread(img_path)[:,:,::-1]

    # dowm sampling -> INTER_AREA
    img = cv2.resize(img, dsize=TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # normalization
    img = np.array(img, dtype="float") / 255.0
    
    return img

def show_imgs(imgs):
    count = len(imgs)
    i = 100 + count * 10
    for img in imgs:
        i += 1
        plt.subplot(i)
        plt.axis('off')
        plt.imshow(img)
    plt.show()
    pass

def draw_result(img, proba_list):
    def get_class(proba_list):
        sort_proba = np.argsort(proba_list)[:,2]
        color, cloth_type = sort_proba

        # color
        color = color.item(0)
        color_proba = proba_list[0][color].numpy().item(0)
        color_proba = np.round(color_proba, 3)
        
        color = color_en_list[color].split('_')[1]


        # type
        cloth_type = cloth_type.item(0)
        type_proba = proba_list[1][cloth_type].numpy().item(0)
        type_proba = np.round(type_proba, 3)
        
        cloth_type = type_en_list[cloth_type].split('_')[1]
        class_result = {'class':[color, cloth_type], 'proba':[color_proba, type_proba]}
        print(class_result)
        
        return class_result
    
    class_dict = get_class(proba_list)
    
    for i in range(0, len(class_dict['class'])):
        string = '{0} : {1}'.format(class_dict['class'][i], class_dict['proba'][i])
        cv2.putText(img, string, (10, (i * 20) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 1, 0), 1)
        
    return img
    