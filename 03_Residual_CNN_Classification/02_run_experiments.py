import os
import sys
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

#per rimuovere le info e i warning di tensorflow (Gemini)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, callbacks
from tensorflow.keras.optimizers import Adam


#--------------------------------------
#controllo esecuzione 01_data_loader.py
#--------------------------------------
def check_data_loading():
    file_needed = [
        "X_aid.npy",
        "X_uc.npy",
        "y_aid.npy",
        "y_uc.npy",
        "labels_aid.txt",
        "labels_uc.txt"
    ]

    for file in file_needed:
        if not os.path.exists(f"dataset/{file}"):
            print("ERRORE")
            print(f"{file} mancante. Eseguire 01_data_loader.py!")
            sys.exit(1)

    print("File necessari trovati, avvio esperimenti")


#--------------------------------
#implementazione blocco residuale
#--------------------------------
def residual_block(x, filters, stride=1):
    shortcut = x

    x = layers.Conv2D(filters=filters, kernel_size=3, strides=stride, padding='same')(x)   #mantiene le stesse dimensioni o le dimezza
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)   #mantiene le stesse dimensioni
    x = layers.BatchNormalization()(x)

    #nel caso di numero diverso di canali o riduzione delle dimensioni, viene applicato una convoluzione con kernel_size=1 per far coincidere le dimensioni dell'input e dell'output
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters=filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)   #è messa dopo la somma per permettere pure addizioni negative
    return x

#-----------------------------
#implementazione ResNet custom
#-----------------------------
def custom_resnet(n_classes, is_strategy2=False):
    inputs = Input(shape=(128, 128, 3))   #costruzione tensore

    #data augmentation (in uso solo per la strategia 2)
    if is_strategy2:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip('horizontal_and_vertical'),
            layers.RandomRotation(0.2)
        ])

        x = data_augmentation(inputs)
    else:
        x = inputs

    #convoluzione
    x = layers.Conv2D(filters=32, kernel_size=7, strides=2, padding='same')(x)   #output: (64, 64, 32)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)   #output: (32, 32, 32)

    #blocchi residuali
    x = residual_block(x, 64)   #output: (32, 32, 64)
    x = residual_block(x, 64)   #output: (32, 32, 64) in questo caso non si attiva l'if del blocco residuale
    x = residual_block(x, 128, 2)   #output: (16, 16, 128)

    #strati densi
    x = layers.Flatten(name='flatten_layer')(x)
    x = layers.Dense(256, activation='relu')(x)   #32768 - 256
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(n_classes, activation='softmax', name=f'{n_classes}-classes')(x)

    return Model(inputs=inputs, outputs=output)


#-------------------------
#split di UCMerced_LandUse
#-------------------------
def split_ucmerced():
    X_uc = np.load('dataset/X_uc.npy', mmap_mode='r')
    y_uc = np.load('dataset/y_uc.npy', mmap_mode='r')

    X_train, X_te, y_train, y_te = train_test_split(X_uc, y_uc, test_size=0.15, stratify=y_uc.argmax(axis=1), random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15/0.85, stratify=y_train.argmax(axis=1), random_state=42)

    return X_tr, y_tr, X_val, y_val, X_te,  y_te


#+----------------------+
#| funzioni strategia 1 |
#+----------------------+
#-----------------
#pre-addestramento
#-----------------
def pretrain_on_aid():
    print("\n\n+-------------+")
    print("| STRATEGIA 1 |")
    print("+-------------+\n")
    print("AVVIO PRE-ADDESTRAMENTO")

    X_aid = np.load('dataset/X_aid.npy', mmap_mode='r')
    y_aid = np.load('dataset/y_aid.npy', mmap_mode='r')

    X_tr, X_val, y_tr, y_val = train_test_split(X_aid, y_aid, test_size=0.3, stratify=y_aid.argmax(axis=1), random_state=42)

    #costruzione modello custom
    model = custom_resnet(n_classes=30)
    model.summary()

    #compilazione
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
    checkpoint = callbacks.ModelCheckpoint(filepath='pretrained_aid.keras', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    #addestramento
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping, checkpoint])
    
    if os.path.exists('pretrained_aid.keras'):
        print("COMPLETATO!")
    else:
        print("ERRORE nel salvataggio del modello")

#-----------
#fine-tuning
#-----------
def finetune(X_tr, y_tr, X_val, y_val):
    print("\nAVVIO FINE-TUNING")

    pretrained_model = tf.keras.models.load_model('pretrained_aid.keras')

    #uso della prima strategia di fine-tuning: congelare i layer convoluzionali e addestrare la MLP finale
    flatten_layer = pretrained_model.get_layer('flatten_layer')   #estrazione del layer flatten
    feature_extractor = Model(inputs=pretrained_model.input, outputs=flatten_layer.output)   #modello con ingresso tensori (128, 128, 3) e in uscita il risultato del layer flatten

    #congelamento dei layer
    for layer in feature_extractor.layers:
        layer.trainable = False
    
    #implementazione data augmentation e MLP
    inputs = Input(shape=(128, 128, 3))

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.2)
    ])
    x = data_augmentation(inputs)
    x = feature_extractor(x)

    x = layers.Dense(256, activation='relu')(x)   #32768 - 256
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(21, activation='softmax', name='21-classes')(x)

    #costruzione modello
    finetuned_model = Model(inputs=inputs, outputs=output, name='resnet_finetuned')
    finetuned_model.summary()

    #compilazione modello con monitoraggio su F1-score
    f1_score = tf.keras.metrics.F1Score(average='macro', name='f1_macro')

    finetuned_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy', f1_score])

    early_stopping = callbacks.EarlyStopping(monitor='val_f1_macro', mode='max', patience=10, restore_best_weights=True, verbose=1)
    checkpoint = callbacks.ModelCheckpoint(filepath='finetuned_model.keras', monitor='val_f1_macro', mode='max', save_best_only=True, verbose=1)
    
    #addestramento
    finetuned_model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping, checkpoint])
    
    if os.path.exists('finetuned_model.keras'):
        print("COMPLETATO!")
    else:
        print("ERRORE nel salvataggio del modello")


#+----------------------+
#| funzioni strategia 2 |
#+----------------------+
#----------------------------------------------
#addestramento direttamente su UCMerced_LandUse
#----------------------------------------------
def train_uc(X_tr, y_tr, X_val, y_val):
    print("\n\n+-------------+")
    print("| STRATEGIA 2 |")
    print("+-------------+\n")
    print("AVVIO ADDESTRAMENTO")

    #costruzione modello custom
    model = custom_resnet(n_classes=21, is_strategy2=True)
    model.summary()

    f1_score = tf.keras.metrics.F1Score(average='macro', name='f1_macro')

    #compilazione
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy', f1_score])

    early_stopping = callbacks.EarlyStopping(monitor='val_f1_macro', mode='max', patience=10, restore_best_weights=True, verbose=1)
    checkpoint = callbacks.ModelCheckpoint(filepath='trained_uc.keras', monitor='val_f1_macro', mode='max', save_best_only=True, verbose=1)

    #addestramento
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping, checkpoint])
    
    if os.path.exists('trained_uc.keras'):
        print("COMPLETATO!")
    else:
        print("ERRORE nel salvataggio del modello")


#-------------
#model section
#-------------
def select_model(X_val, y_val):
    print("\n\n+-----------------+")
    print("| MODEL SELECTION |")
    print("+-----------------+")

    model_stat_1 = tf.keras.models.load_model("finetuned_model.keras")
    model_stat_2 = tf.keras.models.load_model("trained_uc.keras")

    #evaluate ritorna una lista contenente la loss e le metriche del modello [loss, accuracy, f1_macro]
    f1_stat_1 = model_stat_1.evaluate(X_val, y_val, verbose=0)[2]
    f1_stat_2 = model_stat_2.evaluate(X_val, y_val, verbose=0)[2]

    print(f"F1-score strategia 1 --> {f1_stat_1:.2f}")
    print(f"F1-score strategia 2 --> {f1_stat_2:.2f}")

    if f1_stat_1 >= f1_stat_2:
        print("Strategia 1 selezionata")
        best_model_path = "finetuned_model.keras"
    else:
        print("Strategia 2 selezionata")
        best_model_path = "trained_uc.keras"

    shutil.copy(best_model_path, "best_model.keras")
    if os.path.exists("best_model.keras"):
        print("SALVATAGGIO BEST_MODEL COMPLETATO")
    else:
        print("ERRORE nel salvataggio del modello")


#--------------------------------
#estrazione dei nomi delle classi
#--------------------------------
def load_classes_names():
    classes_names = []

    with open("dataset/labels_uc.txt", 'r') as f:
        for line in f:
            label = line.strip().split(': ')
            if len(label)==2:
                classes_names.append(label[1])

    return classes_names


#---------------------------------------------
#valutazione sul test set del modello migliore
#---------------------------------------------
def evaluate_test_set(X_te, y_te):
    print("\n\n+--------------------------+")
    print("| VALUTAZIONE SUL TEST SET |")
    print("+--------------------------+")

    best_model = tf.keras.models.load_model("best_model.keras")

    y_prob = best_model.predict(X_te)
    y_pred_prob = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_te, axis=1)

    #metriche
    acc = accuracy_score(y_true, y_pred_prob)
    prec = precision_score(y_true, y_pred_prob, average='macro')
    recall = recall_score(y_true, y_pred_prob, average='macro')
    f1 = f1_score(y_true, y_pred_prob, average='macro')

    print(f"Accuracy --> {acc:.2f}")
    print(f"Precision --> {prec:.2f}")
    print(f"Recall --> {recall:.2f}")
    print(f"F1-score --> {f1:.2f}")

    #matrice di confusione
    classes_names = load_classes_names()

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred_prob,
        ax=ax,
        display_labels=classes_names,
        cmap="Blues", 
        xticks_rotation='vertical'
    )
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig("Confusion_matrix.png", dpi=200)
    plt.show()
    plt.close()



if __name__ == "__main__":
    #controllo esistenza file necessari
    check_data_loading()

    #split UCMerced_LandUse
    X_tr, y_tr, X_val, y_val, X_te,  y_te = split_ucmerced()

    #+-------------+
    #| strategia 1 |
    #+-------------+
    pretrain_on_aid()
    finetune(X_tr, y_tr, X_val, y_val)

    #+-------------+
    #| strategia 2 |
    #+-------------+
    train_uc(X_tr, y_tr, X_val, y_val)

    #+-----------------+
    #| model selection |
    #+-----------------+
    select_model(X_val, y_val)
    evaluate_test_set(X_te, y_te)