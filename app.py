import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Chargement du pipeline, sélecteur et modèle ---
with open("final_preprocessing_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

with open("final_feature_selector.pkl", "rb") as f:
    selector = pickle.load(f)

with open("final_model_reduced.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📦 Prédiction des Retards de Livraison")
st.markdown("Entrez les informations de commande pour prédire un risque de retard.")

# --- Interface utilisateur pour saisir les données ---
with st.form("form"):
    st.header("Informations Commande")

    order_status = st.selectbox("Statut de commande", ["validated", "not validated", "delivered"])
    order_line_status = st.selectbox("Statut ligne", ["partially delivered", "fully delivered"])
    quantity = st.number_input("Quantité commandée", min_value=1, step=1)
    weight = st.number_input("Poids du produit (kg)", min_value=0.0)
    height = st.number_input("Hauteur produit (cm)", min_value=0.0)
    width = st.number_input("Largeur produit (cm)", min_value=0.0)
    length = st.number_input("Longueur produit (cm)", min_value=0.0)
    category = st.text_input("Catégorie produit")
    container = st.text_input("Type de contenant")
    seller_region = st.text_input("Région vendeur")
    customer_region = st.text_input("Région client")
    weather_level = st.selectbox("Niveau intempérie", ["Non renseigné", "Faible", "Moyenne", "Forte"])

    date_validated = st.date_input("Date de validation")
    submit = st.form_submit_button("Prédire le retard")

if submit:
    # Construction du DataFrame utilisateur
    input_data = pd.DataFrame([{
        "Order Status": order_status,
        "Order Line Status": order_line_status,
        "Order Quantity": quantity,
        "ProductWeight": weight,
        "ProductHeight": height,
        "ProductWidth": width,
        "ProductLength": length,
        "ProductCategory": category,
        "ProductContainerType": container,
        "SellerRegion": seller_region,
        "CustomerRegion": customer_region,
        "Niveau_intempérie": weather_level,
        "Order Validated Date": pd.to_datetime(date_validated)
    }])

    # Feature engineering sur date
    input_data["OrderMonth"] = input_data["Order Validated Date"].dt.month
    input_data["OrderWeekday"] = input_data["Order Validated Date"].dt.weekday
    input_data["OrderYear"] = input_data["Order Validated Date"].dt.year

    # Suppression de colonnes inutiles
    input_data.drop(columns=["Order Validated Date"], inplace=True)

    # Passage dans le pipeline + sélecteur
    transformed = pipeline.transform(input_data)
    reduced = selector.transform(transformed)

    # Prédiction
    prediction = model.predict(reduced)[0]
    prob = model.predict_proba(reduced)[0][1]

    # Affichage du résultat
    st.subheader("Résultat de la prédiction :")
    if prediction == 1:
        st.error(f"🚨 Livraison en retard probable (confiance : {prob:.2%})")
    else:
        st.success(f"✅ Livraison à l'heure probable (confiance : {1 - prob:.2%})")
