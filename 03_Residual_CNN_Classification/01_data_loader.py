import os
import cv2
from sklearn.preprocessing import LabelEncoder
import numpy as np

#per rimuovere le info e i warning di tensorflow (Gemini)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


#-------------
#preprocessing
#-------------
def preprocessing(im):
    #conversione da BGR a RGB
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    #ridimensionamento a 128x128
    im = cv2.resize(im, (128, 128))

    #conversione a float32
    im = im.astype('float32')

    #zero-centering con i valori di media del VGG (ricavati da Gemini)
    VGG_MEAN_RGB = np.array([123.68, 116.779, 103.939])
    im -= VGG_MEAN_RGB

    return im

#----------------------------------------------------
#caricamento del dataset preprocessato in array numpy
#----------------------------------------------------
def load_dataset(path):
    X = []
    y = []

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                im_path = os.path.join(folder_path, file)

                if os.path.isfile(im_path):
                    im = cv2.imread(im_path)

                    if im is not None:
                        im = preprocessing(im)
                        X.append(im)
                        y.append(folder)
        print(f"{folder} completata")

    X, y = np.array(X), np.array(y)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    y = to_categorical(y)

    classes_names = encoder.classes_

    print(f"Dataset {path} caricato\n")
    return X, y, classes_names

#-----------------------------------
#salvataggio del dataset in file npy
#-----------------------------------
def save_dataset(X, y, classes, name):
    save_dir = 'dataset'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Cartella {save_dir} creata")

    np.save(os.path.join(save_dir, f'X_{name}.npy'), X)
    np.save(os.path.join(save_dir, f'y_{name}.npy'), y)

    txt_path = os.path.join(save_dir, f"labels_{name}.txt")
    with open(txt_path, 'w') as f:
        for i, class_name in enumerate(classes):
            f.write(f"{i}: {class_name}\n")
    
    print(f"{name} memorizzato")



if __name__ == "__main__":
    X_aid, y_aid, classes_aid = load_dataset('AID')
    X_uc, y_uc, classes_uc = load_dataset('UCMerced_LandUse/Images')

    save_dataset(X_aid, y_aid, classes_aid, 'aid')
    save_dataset(X_uc, y_uc, classes_uc, 'uc')