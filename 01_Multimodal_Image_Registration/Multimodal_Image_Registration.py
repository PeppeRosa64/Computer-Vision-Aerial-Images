import cv2
import numpy as np
import os
from scipy import optimize
import matplotlib.pyplot as plt


#----------------------------------------------------
#funzione per creare lista delle immagini per dataset
#----------------------------------------------------
def get_dataset(path):
    dataset = []
    #ciclo sulle sottocartelle della lista
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)   #si ottiene il path della sottocartella corrente
        if os.path.isdir(folder_path):
            im_R_path = None
            im_M_path = None
            for file in os.listdir(folder_path):
                filename, _ = os.path.splitext(file)   #separazione del filename dall'estensione per distinguere tra refenence e moving
                if filename.endswith('R'):
                    im_R_path = os.path.join(folder_path, file)
                else:
                    im_M_path = os.path.join(folder_path, file)
            
            #caricamento delle immagini nel dataset
            if im_R_path and im_M_path:
                im_R = cv2.imread(im_R_path)
                im_M = cv2.imread(im_M_path)

                dataset.append({"pair": folder,
                            "im_R": im_R,
                            "im_M": im_M})

    print("Dataset caricato da", path)
    return dataset


#---------------------------
#funzione Mutua Informazione
#---------------------------
def calculate_MI(im_R, im_T, bins):
    #costruzione istogramma 2D
    histo, _, _ = np.histogram2d(im_R, im_T, bins, range=[[0, 256], [0, 256]])

    #normalizzazione
    p_RT = histo / np.sum(histo)

    #calcolo delle probabilità marginali
    p_R = np.sum(p_RT, axis=1)   #per avere la probibilità sulle ascisse si somma lungo le colonne
    p_T = np.sum(p_RT, axis=0)   #per avere la probibilità sulle ordinate si somma lungo le righe

    #calcolo entropie
    h_p_R = -np.sum(p_R[p_R > 0] * np.log(p_R[p_R > 0]))   #filtro [p_R > 0] per scartare valori nulli ed evitare errori di calcolo (log(0) restituirebbe -inf)
    h_p_T = -np.sum(p_T[p_T > 0] * np.log(p_T[p_T > 0]))

    h_pRT = -np.sum(p_RT[p_RT > 0] * np.log(p_RT[p_RT > 0]))

    #calcolo mutua informazione
    mi = h_p_R + h_p_T - h_pRT
    return mi


#----------------------------------------------------------
#funzione da massimizzare per stimare il modello geometrico
#----------------------------------------------------------
def get_MI_trans(params, im_R, im_M, bins):
    #estrazione dei parametri
    tx, ty, theta_deg = params
    theta = np.radians(theta_deg)
    
    #calcolo della matrice di trasformazione
    cos = np.cos(theta)
    sin = np.sin(theta)

    T = np.array([
                 [cos, -sin, tx],
                 [sin, cos, ty]
                 ])
    
    #applicazione della trasformazione
    h, w = im_M.shape[:2]   #altezza, larghezza
    im_T = cv2.warpAffine(im_M, T, (w, h), flags=cv2.INTER_LINEAR)

    #creazione maschera e applicazione della trasformazione
    mask = np.ones((h, w), dtype=np.uint8)
    mask_T = cv2.warpAffine(mask, T, (w, h), flags=cv2.INTER_LINEAR)

    #estrazione dei pixel validi (restituisce le immagini in vettori monodimensionali)
    valid_R = im_R[mask_T == 1]
    valid_T = im_T[mask_T == 1]

    #safety check in caso l'immagine trasformata sia eccessivamente fuori dai bordi
    if len(valid_R)<50:
        return 0.0

    #calcolo della mutua informazione
    mi = calculate_MI(valid_R, valid_T, bins)

    #ritorna dell'opposto della mi per il minimize (trovare il massimo MI significa trovare il minimo -MI)
    return -mi


#----------------------------------------
#funzione di stima del modello geometrico
#----------------------------------------
def estimate_model(im_R, im_M, bins, method, return_history=False):
    #conversione delle immagini in scala dei grigi
    im_R_gray = cv2.cvtColor(im_R, cv2.COLOR_BGR2GRAY)
    im_M_gray = cv2.cvtColor(im_M, cv2.COLOR_BGR2GRAY)

    #applicazione del blur per ridurre il rumore
    im_R_blur = cv2.GaussianBlur(im_R_gray, (11, 11), 0)
    im_M_blur = cv2.GaussianBlur(im_M_gray, (11, 11), 0)

    #array iniziale [tx, ty, theta_deg]
    x0 = [0.0, 0.0, 0.0]

    #configurazione opzioni di minimizzazione per Nelder-Mead e BFGS
    opts = {}
    
    if method == 'Nelder-Mead':
        #costruzione del simplesso iniziale per fare passi di 2.0 pixel/gradi
        step = 2.0
        init_simplex = np.array([
            [0.0, 0.0, 0.0],    #vertice 1: origine
            [step, 0.0, 0.0],   #vertice 2: spostamento su x di 2 pixel
            [0.0, step, 0.0],   #vertice 3: spostamento su y di 2 pixel
            [0.0, 0.0, step]    #vertice 4: rotazione di 2 gradi
        ])
        opts['initial_simplex'] = init_simplex
        
    elif method == 'BFGS':
        #configurazione del valore di epsilon a 2
        opts['eps'] = 2.0
    
    mi_history = []
    def callback(xk):
        val = -get_MI_trans(xk, im_R_blur, im_M_blur, bins)
        mi_history.append(val)

    #stima dei parametri passando le opzioni corrette
    if return_history:
        #utile successivamente per plottare l'andamento della MI
        optimize.minimize(get_MI_trans, x0, args=(im_R_blur, im_M_blur, bins), method=method, options=opts, callback=callback)
        return mi_history
    else:
        #comportamento normale per la Grid Search e il test
        result = optimize.minimize(get_MI_trans, x0, args=(im_R_blur, im_M_blur, bins), method=method, options=opts)
        return result.x


#------------------------------------------------------
#funzione di calcolo dell'errore del modello geometrico
#------------------------------------------------------
def error_model(params, gt_params):
    #errore di traslazione (distanza euclidea)
    error_traslation = np.linalg.norm(params[:2] - gt_params[:2])

    #errore di rotazione (differenza in valore assoluto)
    error_rotation = np.abs(params[2] - gt_params[2])

    return error_traslation, error_rotation


#-------------------------------------------------
#funzione di calcolo dell'errore medio del dataset
#-------------------------------------------------
def mean_error(dataset, dataset_name, bins, method, gt, cache):
    #calcolo dell'errore per ogni coppia del dataset
    errors_tra = []
    errors_rot = []

    for elem in dataset:
        pair, im_R, im_M = elem.values()

        row = gt[(gt['Pair']==pair) & (gt['Dataset']==dataset_name)]   #riga specifica di quella coppia in specifico dataset
        gt_params = np.array([row['Tx'][0], row['Ty'][0], row['AngleRad'][0]])   #lettura parametri del GT (lo 0 estrae il valore dall'array)

        #stima dei parametri
        params = estimate_model(im_R, im_M, bins, method)
        params[2] = np.radians(params[2])   #conversione dell'angolo trovato in radianti per allinearlo alla scala del GT

        #scrittura dei parametri nella cache
        key = (dataset_name, pair, method, bins)
        cache[key] = params

        #stampa dei parametri calcolati e di quelli del GT
        print(f"[{pair}] TROVATI: tx={params[0]:.2f}, ty={params[1]:.2f}, theta={params[2]:.4f}")
        print(f"[{pair}] REALI(GT): tx={gt_params[0]:.2f}, ty={gt_params[1]:.2f}, theta={gt_params[2]:.4f}")
        print("-" * 50)

        #calcolo dell'errore
        errors = error_model(params, gt_params)

        errors_tra.append(errors[0])
        errors_rot.append(errors[1])
        
    error = {"Mean trans": float(np.mean(errors_tra)),
             "Std trans": float(np.std(errors_tra)),
             "Mean rot": float(np.mean(errors_rot)),
             "Std rot": float(np.std(errors_rot))}
    return error


#-----------------------
#funzione di Grid Search
#-----------------------
def grid_search(val_set, gt, cache):
    n_bins = [64, 128, 256]
    methods = ["Powell", "Nelder-Mead", "BFGS"]

    best_for_methods = {}   #{metodo: bins}, migliore configurazione per ogni metodo
    best_method = []   #[metodo, bins], migliore configurazione in assoluto
    best_mean_t = float('inf')

    print("\nINIZIO GRID SEARCH SUL VALIDATION SET\n")

    for method in methods:
        print("\t\t    ", method.upper())
        print("*" * 50)

        best_mean_t_method = float('inf')

        for bins in n_bins:
            print(f"BINS --> {bins}")
            error = mean_error(val_set, 'val', bins, method, gt, cache)

            print(f"Traslazione --> {error['Mean trans']:.2f} ± {error['Std trans']:.2f} px")
            print(f"Rotazione --> {error['Mean rot']:.4f} ± {error['Std rot']:.4f} rad")
            
            mean_t = error["Mean trans"]

            if mean_t < best_mean_t_method:
                best_mean_t_method = mean_t
                best_for_methods[method] = bins

            print("\n\n")
        
        if best_mean_t_method < best_mean_t:
            best_mean_t = best_mean_t_method
            best_method = [method, best_for_methods[method]]

    print("Migliori configurazioni per metodo")
    for method in best_for_methods:
        print(f"{method} --> {best_for_methods[method]} bin")
        
    print(f"\nMIGLIORE CONFIGURAZIONE --> {best_method[0]} - {best_method[1]} bin")
    return best_method


#-----------------------------------------------------------
#funzione di trasformazione e visualizzazione delle immagini
#-----------------------------------------------------------
def apply_trans(dataset, dataset_name, bins, method, cache):
    print(f"\n\nVisualizzazione trasformazioni sul {dataset_name} set")
    print(f"\nMETHOD --> {method}")
    print(f"BINS --> {bins} bin\n")

    for elem in dataset:
        pair, im_R, im_M = elem.values()

        print(f"COPPIA --> {pair}")

        #lettura parametri dalla cache
        chiave = (dataset_name, pair, method, bins)
        tx, ty, theta = cache[chiave]

        #creazione matrice di trasformazione
        cos = np.cos(theta)
        sin = np.sin(theta)
        T = np.array([
                     [cos, -sin, tx],
                     [sin, cos, ty]
                     ])
        
        h, w = im_M.shape[:2]

        #applicazione trasformazione
        im_T = cv2.warpAffine(im_M, T, (w, h), flags=cv2.INTER_LINEAR)

        #calcolo degli overlay (prima e dopo la trosformazione) e della differenza
        overlay_before = cv2.addWeighted(im_R, 0.5, im_M, 0.5, 0)
        overlay_after = cv2.addWeighted(im_R, 0.5, im_T, 0.5, 0)
        diff = cv2.absdiff(im_R, im_T)

        #visualizzazione immagini
        cv2.imshow(f"{pair} Reference", im_R)
        cv2.imshow(f"{pair} Moving", im_M)
        cv2.imshow(f"{pair} Allineata ({method})", im_T)
        cv2.imshow(f"{pair} Differenza", diff)
        cv2.imshow(f"{pair} overlay PRIMA", overlay_before)
        cv2.imshow(f"{pair} overlay DOPO", overlay_after)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #salvataggio immagini allineata e differenza
        save_path = f"RISULTATI/{dataset_name}/{pair}"
        os.makedirs(save_path, exist_ok=True)

        ret1 = cv2.imwrite(f"{save_path}/allineata.png", im_T)
        ret2= cv2.imwrite(f"{save_path}/differenza.png", diff)

        if ret1 and ret2:
            print(f"Immagine allineata e differenza salvate con successo in: {save_path}")


#------------------------------------------
#funzione per plottare l'andamento della MI
#------------------------------------------
def plot_MI(dataset, bins, index):
    print("\n\nGenerazione grafici andamento MI")

    methods = ['Powell', 'Nelder-Mead']

    for i in index:
        elem = dataset[i]
        pair = elem['pair']

        plt.figure(figsize=(8, 5))
        colori = {'Powell': 'blue', 'Nelder-Mead': 'red'}

        for method in methods: 
            print(f"Calcolo history per {method} su {pair}...")

            #calcolo andamento MI
            history = estimate_model(elem['im_R'], elem['im_M'], bins, method, return_history=True)
            
            plt.plot(history, marker='o', linestyle='-', color=colori.get(method, 'green'), label=f"{method}", markersize=4)
        
        plt.title(f"Andamento Mutua Informazione\nCoppia {pair} ({bins} bins)", fontsize=13)
        plt.xlabel("Iterazioni", fontsize=11)
        plt.ylabel("Mutua Informazione", fontsize=11)
        plt.legend(loc="upper right", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        #salvataggio plot
        save_path = "RISULTATI/plots"
        os.makedirs(save_path, exist_ok=True)

        plt.savefig(f"{save_path}/andamento MI {pair}.png")
        print(f"Grafico salvato con successo in: {save_path}")

        #visualizzazione plot
        plt.show()


#---------------------------------------------------------------------------------------
#funzione per richiedere il path di un'immagine e verificare che quest'ultima sia valida
#---------------------------------------------------------------------------------------
def get_valide_img(type_img):
    while True:
        path = input(f"Inserisci path dell'immagine {type_img} --> ")

        #verifica che il path inserito sia un file
        if not os.path.isfile(path):
            print("ERRORE - Nessun file trovato")
            continue
        
        #verifica che il file sia un'immagine
        img = cv2.imread(path)
        if img is None:
            print("ERRORE - il file non è un'immagine valida")
            continue

        return img


#----------------------------------------------------------
#funzione per applicare la pipeline milgiore a due immagini
#----------------------------------------------------------
def apply_custom_trans(method, bins):
    print("\n\nPROVA DELLA PIPELINE\n")

    im_R = get_valide_img("REFERENCE")
    im_M = get_valide_img("MOVING")

    #calcolo dei parametri della matrice di trasformazione
    tx, ty, theta_deg = estimate_model(im_R, im_M, bins, method)
    theta = np.radians(theta_deg)

    print(f"PARAMETRI --> tx={tx:.2f}, ty={ty:.2f}, theta={theta:.4f}")

    #calcolo matrice di trasformazione
    cos = np.cos(theta)
    sin = np.sin(theta)

    T=np.array([[cos, -sin, tx],
                [sin, cos, ty]
                ])
    
    #applicazione trasformazione
    h, w = im_M.shape[:2]
    im_T = cv2.warpAffine(im_M, T, (w, h), flags=cv2.INTER_LINEAR)

    #calcolo degli overlay (prima e dopo la trosformazione) e della differenza
    overlay_before = cv2.addWeighted(im_R, 0.5, im_M, 0.5, 0)
    overlay_after = cv2.addWeighted(im_R, 0.5, im_T, 0.5, 0)
    diff = cv2.absdiff(im_R, im_T)

    #visualizzazione immagini
    cv2.imshow("Reference", im_R)
    cv2.imshow("Moving", im_M)
    cv2.imshow(f"Allineata ({method})", im_T)
    cv2.imshow("Differenza", diff)
    cv2.imshow("Overlay PRIMA", overlay_before)
    cv2.imshow("Overlay DOPO", overlay_after)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    #lettura del dataset
    val_set = get_dataset("DATASET/val")
    test_set = get_dataset("DATASET/test")

    #lettura del ground truth
    gt = np.genfromtxt("DATASET/GT.csv", dtype=None, delimiter=';', names=True, encoding='utf-8')

    #+------------+
    #| VALIDATION |
    #+------------+
    cache_params = {}   #{(dataset_name, pair, method, bins): [tx, ty, theta_rad]}
    method, bins = grid_search(val_set, gt, cache_params)   #calcolo migliore configurazione
    apply_trans(val_set, "val", bins, method, cache_params)   #applicazione e salvataggio delle trasformazioni
    plot_MI(val_set, bins, [3, 7])   #visualizzazione e salvataggio andamento MI sulle coppie C4 e C8


    #+------+
    #| TEST |
    #+------+
    print("\n\nINIZIO FASE DI TEST\n")
    error_test = mean_error(test_set, "test", bins, method, gt, cache_params)   #calcolo errore sul TEST set
    print("\nErrori sul test set")
    print(f"Traslazione --> {error_test['Mean trans']:.2f} ± {error_test['Std trans']:.2f} px")
    print(f"Rotazione --> {error_test['Mean rot']:.4f} ± {error_test['Std rot']:.4f} rad")
    apply_trans(test_set, "test", bins, method, cache_params)   #applicazione e salvataggio delle trasformazioni

    #+----------+
    #| VERIFICA |
    #+----------+
    apply_custom_trans(method, bins)