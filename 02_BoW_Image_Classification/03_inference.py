import os
import cv2
import pickle
import numpy as np
from sklearn.preprocessing import normalize

#---------------------------------------
#funzione di preproccesing dell'immagine
#---------------------------------------
def preprocessing(im):
    #conversione in scala dei grigi
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    #equalizzazione dell'istogramma per aumentare il contrasto
    im_eq = cv2.equalizeHist(im_gray)

    return im_eq


#----------------------------------------------
#funzione per caricare e verificare un'immagine
#----------------------------------------------
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

#----------------------------------------------------------
#funzione per caricare il modello e il vocabolario migliore
#----------------------------------------------------------
def load_model():
    #controllo esistenza dei file necessari
    if not os.path.exists("best_model.pkl") or not os.path.exists("best_vocab.pkl"):
        print("NESSUN MODELLO O VOCABOLARIO TROVATO. Assicurarsi di eseguire 02_train_classifier.py")
        return None, None
    
    with open("best_model.pkl", "rb") as f:
        clf = pickle.load(f)
    
    with open("best_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    return clf, vocab


#------------------------------------
#funzione per classificare l'immagine
#------------------------------------
def classify_image(im, obj, clf, vocab):
    im_processed = preprocessing(im)

    #calcolo descrittori e normalizzazione
    kp, descr = obj.detectAndCompute(im_processed, None)
    if descr is None:
        print("IMPOSSIBILE CLASSIFICARE L'IMMAGINE")
        return
    descr = normalize(descr, norm="l2")

    #calcolo istogramma
    visual_words = vocab.predict(descr)
    k = vocab.n_clusters
    histogram, _ = np.histogram(visual_words, bins=np.arange(k+1))
    histogram = histogram.reshape(1, -1)   #conversione in array 2D
    histogram = normalize(histogram, norm="l1")

    #classificazione
    label = clf.predict(histogram)
    print(f"--> {label[0]} <--")



if __name__ == "__main__":
    obj = cv2.SIFT_create()
    clf, vocab = load_model()

    if clf is not None:
        while(True):
            im = load_image()
            if im is None:
                continue

            classify_image(im, obj, clf, vocab)

            flag = input("\nVuoi continuare [y/n]? ")
            if flag == 'n':
                break