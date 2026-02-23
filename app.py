import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import mode

# Configuration de la page
st.set_page_config(page_title="FontGuard AI", page_icon="🔍")

# --- CHARGEMENT DU MODÈLE ET DES ÉTIQUETTES ---
@st.cache_resource 
def load_model_data():
    if os.path.exists('mon_detecteur_de_polices.pkl'):
        # On charge le dictionnaire (Pipeline + Labels)
        data = joblib.load('mon_detecteur_de_polices.pkl')
        if isinstance(data, dict):
            return data['pipeline'], data['db_labels']
        else:
            # Cas de secours si tu avais sauvegardé uniquement le pipeline avant
            return data, None
    return None, None

pipeline, db_labels = load_model_data()

# --- CONSTANTES ---
IMG_SIZE = 150
FONT_SIZE = 120
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
THRESHOLD = 0.2
MIN_LETTERS = 20

# --- INTERFACE ---
st.title("🔍 FontGuard AI")
st.subheader("Système d'authentification et de reconnaissance de polices")



if pipeline is None:
    st.error("⚠️ Modèle non trouvé ! Assure-toi d'avoir généré 'mon_detecteur_de_polices.pkl' avec ton script d'entraînement.")
else:
    st.write("Dépose un fichier de police pour analyser sa signature morphologique.")

    uploaded_file = st.file_uploader("Choisir une police", type=['ttf', 'otf'])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            font_path = tmp_file.name

        try:
            # 1. Génération des images
            X_to_predict = []
            display_images = []
            font = ImageFont.truetype(font_path, FONT_SIZE)

            for char in ALPHABET:
                img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
                draw = ImageDraw.Draw(img)
                bbox = draw.textbbox((0, 0), char, font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((IMG_SIZE-w)/2 - bbox[0], (IMG_SIZE-h)/2 - bbox[1]), char, font=font, fill=0)
                
                X_to_predict.append(np.array(img).flatten() / 255.0)
                display_images.append(img)

            X_to_predict = np.array(X_to_predict)

            # 2. Inférence
            probs = pipeline.predict_proba(X_to_predict)
            preds = pipeline.predict(X_to_predict)
            max_probs = np.max(probs, axis=1)
            
            score = np.sum(max_probs > THRESHOLD)
            is_authorized = score >= MIN_LETTERS

            # 3. Affichage du Verdict
            st.divider()
            col1, col2 = st.columns([1, 2])

            with col1:
                if is_authorized:
                    st.success("✅ ACCÈS AUTORISÉ")
                    # Identification de la ressemblance
                    resemblance_id = pd.Series(preds).mode()[0]
                    font_name = db_labels[resemblance_id] if db_labels is not None else f"ID {resemblance_id}"
                    
                    st.write(f"**Identifié comme :** {font_name}")
                    st.metric("Confiance", f"{score}/26")
                else:
                    st.error("❌ ACCÈS REFUSÉ")
                    st.write("**Origine :** Inconnue ou non certifiée")
                    st.metric("Confiance", f"{score}/26")

            with col2:
                st.write("**Aperçu des caractères extraits :**")
                # Affichage des premières lettres pour preuve visuelle
                st.image(display_images[:7], width=70)

            st.divider()
            
        except Exception as e:
            st.error(f"Erreur d'analyse : {e}")
        finally:
            if os.path.exists(font_path):
                os.unlink(font_path)

st.sidebar.markdown("### Détails Techniques")
st.sidebar.write(f"**Seuil de confiance :** {THRESHOLD}")
st.sidebar.write(f"**Minimum requis :** {MIN_LETTERS} lettres")
st.sidebar.info("Le modèle compare la structure SVD de la police chargée avec les signatures certifiées de la base de données.")