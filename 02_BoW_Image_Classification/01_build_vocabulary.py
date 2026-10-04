import os
import cv2
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
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

#-------------------------------------------------------------------------------------
#funzione per la costruzione della matrice deii descrittori delle immagini del dataset
#-------------------------------------------------------------------------------------
def get_matrix_features(dataset_path):
    #istanziazione oggetto SIFT
    obj = cv2.SIFT_create()

    #lista descrittori
    features = []

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                im_path = os.path.join(folder_path, file)

                if os.path.isfile(im_path):
                    im = cv2.imread(im_path)

                    if im is not None:
                        im_processed = preprocessing(im)
                        kp, descr = obj.detectAndCompute(im_processed, None)

                        if descr is not None:
                            descr = descr[:500]    #estrazione di 500 descrittori per immagini (causa limiti di calcolo)
                            features.append(descr)
                            print(f"Descrittori estratti per {file}")
    
    #creazione matrice di descrittori
    matrix_features = np.vstack(features)
    print(f"\nDimensione matrice descrittori --> {matrix_features.shape}\n")

    #normalizzazione L2 della matrice
    matrix_features = normalize(matrix_features, norm="l2")

    return matrix_features


#----------------------------------------
#funzione per la creazione dei vocabolari
#----------------------------------------
def create_vocabs(matrix_features):
    vocab_dim = [50, 100, 500]

    for k in vocab_dim:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)   #uso alternativo a KMeans per velocizzare il processo
        kmeans.fit(matrix_features)

        vocab_name = f"vocab_k{k}.pkl"
        with open(vocab_name, "wb") as f:
            pickle.dump(kmeans, f)

        if os.path.exists(f"vocab_k{k}.pkl"):
            print(f"Vocabolario vocab_k{k}.pkl salvato correttamente")



if __name__ == "__main__":
    matrix_features = get_matrix_features("AID")
    create_vocabs(matrix_features)