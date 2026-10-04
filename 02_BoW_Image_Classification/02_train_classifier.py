import os
import cv2
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


#---------------------------------------
#funzione di preproccesing dell'immagine
#---------------------------------------
def preprocessing(im):
    #conversione in scala dei grigi
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    #equalizzazione dell'istogramma per aumentare il contrasto
    im_eq = cv2.equalizeHist(im_gray)

    return im_eq


#-----------------------------------------
#funzione per l'estrazione dei descrittori
#-----------------------------------------
def get_features(dataset_path):
    #istanziazione oggetto SIFT
    obj = cv2.SIFT_create()

    #lista descrittori
    features = []

    #lista etichette per campione (condividono lo stesso indice del campione di riferimento)
    labels = []

    print("\nCalcolo descrittori in corso...")

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
                            descr = normalize(descr, norm="l2")   #normalizzazione coerente con quella del vocabolario
                            features.append(descr)
                            labels.append(folder)

    print("Fatto!")
    return features, labels


#--------------------------------------------------------------------------
#funzione per la costruzione della matrice degli istogrammi per vocabolario
#--------------------------------------------------------------------------
def get_matrix_histograms(features, labels, k):
    vocab_name = f"vocab_k{k}.pkl"

    #controllo generazione vocabolario
    if not os.path.exists(vocab_name):
        print("VOCABOLARIO INESISTENTE, assicurasi di eseguire 01_build_vocabulary.py")
        return None, None

    #caricamento vocabolario
    with open(vocab_name, "rb") as f:
        vocab = pickle.load(f)

    #matrice di istogrammi normalizzati L1
    X = []

    print("Costruzione matrice in corso...")

    for descr in features:
        visual_words = vocab.predict(descr)
        histogram, _ = np.histogram(visual_words, bins=np.arange(k+1))   #il secondo valore return bins_edges non serve
        X.append(histogram)

    X = np.vstack(X)   #creazione matrice 2D
    y = np.array(labels)   #creazione vettore di etichette

    X = normalize(X, norm="l1")   #normalizzazione L1 sulla matrice (frequenze relative)

    print("Fatto!")
    return X, y


#----------------------------------------------
#funzione di training e valutazione del modello
#----------------------------------------------
def evaluate_model(X, y, clf, clf_name, k):
    cv_stratified = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    #valutazione modello
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']   #metriche di valutazione
    score = cross_validate(clf, X, y, cv=cv_stratified, n_jobs=-1, scoring=metrics)   #score della 3-fold cross validation

    acc = score["test_accuracy"].mean()
    prec = score["test_precision_macro"].mean()
    recall = score["test_recall_macro"].mean()
    f1 = score["test_f1_macro"].mean()

    print(f"Accuracy: {acc:.2f} | Precision: {prec:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")

    #matrice di confusione
    y_pred = cross_val_predict(clf, X, y, cv=cv_stratified, n_jobs=-1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, 
        ax=ax, 
        cmap="Blues", 
        xticks_rotation='vertical'
    )
    
    plt.title(f"Matrice di Confusione - {clf_name} (K={k})")
    plt.tight_layout()
    
    fixed_name = clf_name.replace(" ", "_")   #nome con underscore (es. "CM_Random_Forest_K50.png")
    plt.savefig(f"CM_{fixed_name}_K{k}.png", dpi=200)
    plt.close()

    return f1


#---------------------------------------------
#funzione per la selezione del miglior modello
#---------------------------------------------
def get_best_model(dataset_path):
    k_values = [50, 100, 500]

    clf_to_test = ["SVM con kernel RBF", "Random Forest"]
    
    best_model = {"clf_name": None, "clf": None, "k": 0, "score": 0, "X": None, "y": None}   #dizionario per memorizzare il modello migliore (clf_name, clf, k, score, X, y)

    #calcolo descrittori ed etichette
    features, labels = get_features(dataset_path)

    for k in k_values:
        print(f"\n\t--> K = {k} <--")
        X, y = get_matrix_histograms(features, labels, k)

        if X is None: continue   #salta l'iterazione in caso il vocabolario non esista

        for clf_name in clf_to_test:
            print(f"\n{clf_name}")

            #istanziazione oggetto classificatori
            if clf_name == "SVM con kernel RBF":
                clf = SVC(kernel='rbf', class_weight='balanced', C=100)
            else:
                clf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced', max_depth=None)

            #valutazione modello
            score = evaluate_model(X, y, clf, clf_name, k)

            if score > best_model["score"]:
                best_model |= {"clf_name": clf_name, 
                               "clf": clf,
                                "k": k, 
                                "score": score, 
                                "X": X, 
                                "y": y}

    print("\n+-----------------+")
    print("| MIGLIOR MODELLO |")
    print("+-----------------+")
    print(f"clf --> {best_model['clf_name']} \nk --> {best_model['k']} \nscore -->  {best_model['score']:.2f}\n")          
    return best_model


#------------------------------------------------
#funzione per il salvataggio del modello migliore
#------------------------------------------------
def save_best_model(best_model):
    #controllo che la procedura di calcolo del modello sia riuscita
    if best_model['clf'] is None:
        print("Nessun modello trovato. Assicurarsi di avere calcolato i vocabolari.")
        return
    
    print("Addestramento in corso...")
    clf = best_model["clf"]
    X = best_model["X"]
    y = best_model["y"]

    clf.fit(X, y)   #addestramento su tutto il dataset
    print("Fatto!")


    #salvataggio modello
    with open("best_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    if os.path.exists("best_model.pkl"):
        print("best_model.pkl salvato correttamente")

    #copia del vocabolario migliore
    vocab_name = f"vocab_k{best_model['k']}.pkl"
    shutil.copy(vocab_name, "best_vocab.pkl")
    if os.path.exists("best_vocab.pkl"):
        print("best_vocab.pkl salvato correttamente")
    


if __name__ == "__main__":
    #+-------------------+
    #| SELEZIONE MODELLO |
    #+-------------------+
    print("Scelgo il modello migliore...")
    best_model = get_best_model("UCMerced_LandUse/Images")

    #+---------------------+
    #| SALVATAGGIO MODELLO |
    #+---------------------+
    save_best_model(best_model)