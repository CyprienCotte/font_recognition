import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scipy.stats import mode

# ==========================================
# 1. CONFIGURATION ET PARAMÈTRES
# ==========================================
FONTS_DIR = 'fonts'        # Dossier contenant vos 24 polices de base
FONTS_TEST_DIR = 'fonts_test' # Dossier contenant les 10 polices à tester
IMG_SIZE = 150
FONT_SIZE = 120
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
THRESHOLD = 0.2            # Seuil de confiance optimal identifié
MIN_LETTERS = 20           # Nombre de lettres minimum pour valider une police

# ==========================================
# 2. GÉNÉRATION DES DONNÉES
# ==========================================
def add_noise(image_array, noise_level=0.15):
    """Ajoute un bruit aléatoire pour simuler la réalité."""
    state = np.random.RandomState(42) # Pour la reproductibilité
    noise = state.rand(*image_array.shape)
    noisy_image = image_array.copy()
    noisy_image[noise < noise_level/2] = 0
    noisy_image[noise > 1 - noise_level/2] = 255
    return noisy_image

def generate_font_dataset(folder_path):
    """Transforme les fichiers .ttf/.otf en vecteurs de pixels."""
    X, y, font_names = [], [], []
    font_files = [f for f in os.listdir(folder_path) if f.endswith(('.ttf', '.otf'))]
    
    for idx, font_file in enumerate(font_files):
        font_path = os.path.join(folder_path, font_file)
        font_names.append(font_file)
        try:
            font = ImageFont.truetype(font_path, FONT_SIZE)
            for char in ALPHABET:
                # Image propre
                img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
                draw = ImageDraw.Draw(img)
                bbox = draw.textbbox((0, 0), char, font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((IMG_SIZE-w)/2 - bbox[0], (IMG_SIZE-h)/2 - bbox[1]), char, font=font, fill=0)
                
                # Conversion et Normalisation
                arr_clean = np.array(img).flatten() / 255.0
                arr_noisy = add_noise(np.array(img)).flatten() / 255.0
                
                X.append(arr_clean); y.append(idx)
                X.append(arr_noisy); y.append(idx)
        except Exception as e:
            print(f"Erreur sur {font_file}: {e}")
            
    return np.array(X), np.array(y), font_names

# ==========================================
# 3. LOGIQUE D'INFÉRENCE ET PRODUCTION
# ==========================================
def run_font_inference(X_data, font_list, pipeline, db_labels):
    """Applique le modèle et identifie la ressemblance."""
    n_img = 52 # 26 lettres * 2
    results = []
    
    for i in range(len(X_data) // n_img):
        batch = X_data[i*n_img : (i+1)*n_img]
        probs = pipeline.predict_proba(batch)
        preds = pipeline.predict(batch)
        
        max_probs = np.max(probs, axis=1)
        valid_letters = np.sum(max_probs > THRESHOLD) / 2
        
        is_authorized = valid_letters >= MIN_LETTERS
        
        if is_authorized:
            # On cherche la police la plus fréquente parmi les prédictions
            resemblance_id = pd.Series(preds).mode()[0]
            resemblance_name = db_labels[resemblance_id]
        else:
            resemblance_name = "Inconnue"
            
        results.append({
            "Fichier": font_list[i],
            "Score (/26)": round(valid_letters, 1),
            "Verdict": "AUTORISÉ (1)" if is_authorized else "REFUSÉ (0)",
            "Ressemble à": resemblance_name
        })
    return pd.DataFrame(results)

# ==========================================
# 4. PIPELINE PRINCIPAL (MAIN)
# ==========================================
if __name__ == "__main__":

    np.random.seed(42) # Pour la reproductibilité
    print("--- Phase 1 : Préparation de la Base de Données ---")
    X, y, db_labels = generate_font_dataset(FONTS_DIR)
    
    # Séparation Train/Test (Isoler 4 polices pour validation de l'inconnu)
    unique_ids = np.unique(y)
    unknown_ids = np.random.choice(unique_ids, 4, replace=False)
    
    mask_unk = np.isin(y, unknown_ids)
    X_train_full, y_train_full = X[~mask_unk], y[~mask_unk]
    X_unk_test, y_unk_test = X[mask_unk], y[mask_unk]
    unk_names = [db_labels[i] for i in unknown_ids]
    
    # Split Train/Val pour les polices connues
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

    print(f"Entraînement sur {len(np.unique(y_train))} polices...")
    
    # Construction du Pipeline
    # # 
    # pipeline = Pipeline([
    #     ('svd', TruncatedSVD(n_components=459, random_state=42)),
    #     ('rf', RandomForestClassifier(n_estimators=250, max_depth=20, random_state=42))
    # ])
    
    # pipeline.fit(X_train, y_train)
    # print(f"Accuracy de base (connu) : {pipeline.score(X_val, y_val):.2%}")

    from sklearn.model_selection import GridSearchCV

    print("\n--- Phase 2 : Optimisation des Hyperparamètres (GridSearch) ---")

    # 1. Définition du Pipeline de base
    base_pipeline = Pipeline([
        ('svd', TruncatedSVD(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 2. Définition de la grille de recherche
    # Note : on utilise le préfixe 'nom_du_step__' pour cibler les paramètres
    param_grid = {
        'svd__n_components': [100, 250, 450],
        'rf__n_estimators': [100, 250],
        'rf__max_depth': [10, 20, 30]
    }

    # 3. Initialisation de la recherche
    # cv=3 : on divise les données en 3 pour la validation croisée
    # n_jobs=-1 : utilise tous les processeurs de votre ordinateur pour aller plus vite
    grid_search = GridSearchCV(base_pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)

    print("Recherche des meilleurs paramètres en cours (cela peut prendre quelques minutes)...")
    grid_search.fit(X_train, y_train)

    # 4. Extraction du meilleur modèle
    pipeline = grid_search.best_estimator_

    print("\nRésultats de l'optimisation :")
    print(f"Meilleurs paramètres : {grid_search.best_params_}")
    print(f"Meilleur score de validation croisée : {grid_search.best_score_:.2%}")

    # On met à jour l'accuracy sur le set de validation final
    print(f"Accuracy finale sur set de validation : {pipeline.score(X_val, y_val):.2%}")

 

    print("\n--- Phase 2 : Test sur les 10 nouvelles polices (Production) ---")
    if os.path.exists(FONTS_TEST_DIR):
        X_prod, y_prod, labels_prod = generate_font_dataset(FONTS_TEST_DIR)
        final_report = run_font_inference(X_prod, labels_prod, pipeline, db_labels)
        print(final_report.to_string(index=False))
    else:
        print(f"Erreur : Le dossier {FONTS_TEST_DIR} n'existe pas.")

import joblib


# Sauvegarder le pipeline ET la liste des noms de polices
data_to_save = {
    'pipeline': pipeline,
    'db_labels': db_labels
}
joblib.dump(data_to_save, 'mon_detecteur_de_polices.pkl')
print("\n[INFO] Modèle sauvegardé sous 'mon_detecteur_de_polices.pkl'")

# Pour le réutiliser plus tard sans ré-entraîner :
# model_charge = joblib.load('mon_detecteur_de_polices.pkl')

