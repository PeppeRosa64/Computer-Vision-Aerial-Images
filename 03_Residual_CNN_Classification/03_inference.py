import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

#per rimuovere le info e i warning di tensorflow (Gemini)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


#------------------------------------------------
#caricamento modello migliore e nomi delle classi
#------------------------------------------------
def load_model_classes():
    if not os.path.exists("dataset/labels_uc.txt"):
        print("ERRORE")
        print("labels_uc non trovato. Eseguire 01_data_loader.py")
        sys.exit(1)

    if not os.path.exists("best_model.keras"):
        print("ERRORE")
        print("Nessun modello trovato. Eseguire 02_run_experiment.py")
        sys.exit(1)

    classes_names = []

    with open("dataset/labels_uc.txt", 'r') as f:
        for line in f:
            label = line.strip().split(': ')
            if len(label)==2:
                classes_names.append(label[1])

    best_model = tf.keras.models.load_model("best_model.keras")

    return best_model, classes_names


#-------------------------------------
#caricamento e verifica di un'immagine
#-------------------------------------
def load_image():
    path = input("Inserisci path immagine --> ")

    #verifica che il path inserito sia un file
    if not os.path.isfile(path):
        print("ERRORE - Nessun file trovato")
        return None
    
    #verifica che il file sia un'immagine
    img = cv2.imread(path)
    if img is None:
        print("ERRORE - il file non è un'immagine valida")

    return img


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


#---------------
#classificazione
#---------------
def classify_image(im_raw, im_pre, model, classes_names):
    im = np.expand_dims(im_pre, axis=0)
    pred_prob = model.predict(im, verbose=0)[0]   #restituisce una lista di liste, in questo caso contiene una sola lista che viene estratta con [0]

    top5_index = np.argsort(pred_prob)[-5:][::-1]   #prendo le 5 classi con più alta probabilità, tramite gli indici delle probabilità maggiori
    top5_classes = [classes_names[i] for i in top5_index]
    top5_probs = [pred_prob[i]*100 for i in top5_index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{top5_classes[0].upper()}: {top5_probs[0]:.2f}%", fontsize=18, fontweight='bold')

    im_rgb = cv2.cvtColor(im_raw, cv2.COLOR_BGR2RGB)
    axes[0].imshow(im_rgb)
    axes[0].axis('off')
    axes[0].set_title("Immagine in Input", fontsize=14)

    y_pos = np.arange(len(top5_classes))
    bars = axes[1].barh(y_pos, top5_probs, edgecolor='black')

    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top5_classes, fontsize=12)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Confidenza della Rete', fontsize=12)
    axes[1].set_xlim([0, 100])
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[1].text(width + 1.5, bar.get_y() + bar.get_height()/2, 
                     f"{top5_probs[i]:.1f}", 
                     va='center', fontsize=11, color='black')
                     
    plt.tight_layout()
    fig.subplots_adjust(top=0.82)
    plt.show()
    plt.close()



if __name__=="__main__":
    best_model, classes_names = load_model_classes()

    while(True):
        im = load_image()
        if im is None:
            continue

        im_pre = preprocessing(im)
        classify_image(im, im_pre, best_model, classes_names)

        flag = input("\nVuoi continuare [y/n]? ")
        if flag == 'n':
            break