
import streamlit as st
import numpy as np
import pandas as pd
import joblib, json, os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from fpdf import FPDF
import lime
import lime.lime_tabular
import base64, io

# ─── Config page ────────────────────────────────────────────
st.set_page_config(
    page_title="Centre Neurologique – Prédiction AVC",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS personnalisé ───────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.6rem;
        font-weight: 800;
        padding: 10px 0;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
    }
    .risk-high {
        background: rgba(239,68,68,0.15);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .risk-low {
        background: rgba(34,197,94,0.15);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #6d28d9, #1d4ed8);
        transform: translateY(-1px);
    }
    [data-testid="stSidebar"] {
        background: rgba(15,12,41,0.9);
        border-right: 1px solid rgba(167,139,250,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ─── Chargement modèle ──────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("model/tabnet_mlp_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

# ─── Auth ────────────────────────────────────────────────────
def load_users():
    with open("users.json", "r") as f:
        return json.load(f)

def check_auth(username, password):
    users = load_users()
    if username in users and users[username]["password"] == password:
        return users[username]
    return None

# ─── Stockage prédictions ────────────────────────────────────
LOG_FILE = "predictions_log.json"

def save_prediction(user, data, risk_score, risk_label):
    log = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            log = json.load(f)
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "data": data,
        "risk_score": float(risk_score),
        "risk_label": risk_label
    }
    log.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

def load_predictions():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []

# ─── Page d'accueil ──────────────────────────────────────────
def page_accueil():
    st.markdown('''
    <div class="main-title">
        🧠 Bienvenue chers patients au Centre Neurologique<br>
        de Prédiction du Risque d'AVC<br>
        <span style="font-size:1.1rem">Dauris – Bouchra – Amine</span>
    </div>
    <div class="subtitle">
        Système intelligent de détection précoce basé sur l'intelligence artificielle hybride TabNet-MLP
    </div>
    ''', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('''<div class="card">
        <h3 style="color:#a78bfa">🤖 Modèle Hybride</h3>
        <p style="color:#cbd5e1">TabNet + MLP — Ensemble learning pour une précision maximale dans la détection du risque AVC</p>
        </div>''', unsafe_allow_html=True)
    with col2:
        st.markdown('''<div class="card">
        <h3 style="color:#60a5fa">🔍 Explicabilité LIME</h3>
        <p style="color:#cbd5e1">Comprendre les facteurs déterminants de chaque prédiction grâce à l'explication locale LIME</p>
        </div>''', unsafe_allow_html=True)
    with col3:
        st.markdown('''<div class="card">
        <h3 style="color:#34d399">📄 Rapport PDF</h3>
        <p style="color:#cbd5e1">Générez et téléchargez un rapport médical complet avec résultats et recommandations</p>
        </div>''', unsafe_allow_html=True)

# ─── Page connexion ──────────────────────────────────────────
def page_connexion():
    st.markdown("<h2 style='color:#a78bfa;text-align:center'>🔐 Authentification</h2>",
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            username = st.text_input("👤 Nom d'utilisateur")
            password = st.text_input("🔑 Mot de passe", type="password")
            submit   = st.form_submit_button("Se connecter")
            st.markdown("</div>", unsafe_allow_html=True)
            if submit:
                user_data = check_auth(username, password)
                if user_data:
                    st.session_state["authenticated"] = True
                    st.session_state["username"]       = username
                    st.session_state["user_data"]      = user_data
                    st.success(f"✅ Bienvenue {user_data['prenom']} {user_data['nom']} !")
                    st.rerun()
                else:
                    st.error("❌ Identifiants incorrects")

# ─── Page prédiction ────────────────────────────────────────
def page_prediction():
    model, scaler = load_model()
    st.markdown("<h2 style='color:#60a5fa'>🔮 Prédiction du risque d'AVC</h2>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        nom    = st.text_input("Nom du patient")
        prenom = st.text_input("Prénom du patient")
        age    = st.slider("Âge", 18, 100, 50)
        gender = st.selectbox("Genre", ["Masculin", "Féminin"])
        hypertension  = st.selectbox("Hypertension", ["Non", "Oui"])
        heart_disease = st.selectbox("Maladie cardiaque", ["Non", "Oui"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        ever_married      = st.selectbox("Marié(e)", ["Non", "Oui"])
        work_type         = st.selectbox("Type de travail",
                           ["Privé", "Auto-entrepreneur", "Fonction publique",
                            "Enfant", "Sans emploi"])
        residence         = st.selectbox("Type de résidence", ["Urbain", "Rural"])
        avg_glucose       = st.number_input("Glycémie moyenne (mg/dL)", 50.0, 300.0, 100.0)
        bmi               = st.number_input("IMC (kg/m²)", 10.0, 60.0, 25.0)
        smoking_status    = st.selectbox("Statut tabagique",
                           ["Ancien fumeur", "Non-fumeur", "Fumeur", "Inconnu"])
        st.markdown("</div>", unsafe_allow_html=True)

    # Encodage
    enc = {
        "gender": 1 if gender == "Masculin" else 0,
        "age": age,
        "hypertension":  1 if hypertension == "Oui" else 0,
        "heart_disease": 1 if heart_disease == "Oui" else 0,
        "ever_married":  1 if ever_married == "Oui" else 0,
        "work_type":     ["Privé","Auto-entrepreneur","Fonction publique",
                          "Enfant","Sans emploi"].index(work_type),
        "Residence_type": 1 if residence == "Urbain" else 0,
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "smoking_status": ["Ancien fumeur","Non-fumeur","Fumeur","Inconnu"].index(smoking_status)
    }

    X_input = np.array([[enc[k] for k in [
        "gender","age","hypertension","heart_disease","ever_married",
        "work_type","Residence_type","avg_glucose_level","bmi","smoking_status"
    ]]], dtype=np.float32)
    X_scaled = scaler.transform(X_input)

    if st.button("🔮 Prédire le risque d'AVC", use_container_width=True):
        proba     = model.predict_proba(X_scaled)[0][1]
        risk_pct  = proba * 100
        risk_lbl  = "ÉLEVÉ" if proba >= 0.5 else "FAIBLE"

        # Sauvegarde
        save_prediction(
            st.session_state["username"],
            {"nom": nom, "prenom": prenom, **enc},
            proba, risk_lbl
        )

        # Résultat
        st.markdown("<hr>", unsafe_allow_html=True)
        if risk_lbl == "ÉLEVÉ":
            st.markdown(f'''<div class="risk-high">
            <h2 style="color:#ef4444">⚠️ RISQUE {risk_lbl}</h2>
            <h1 style="color:#fca5a5;font-size:3rem">{risk_pct:.1f}%</h1>
            <p style="color:#fca5a5">Consultation médicale urgente recommandée</p>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div class="risk-low">
            <h2 style="color:#22c55e">✅ RISQUE {risk_lbl}</h2>
            <h1 style="color:#86efac;font-size:3rem">{risk_pct:.1f}%</h1>
            <p style="color:#86efac">Maintenir un mode de vie sain</p>
            </div>''', unsafe_allow_html=True)

        # Jauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Score de Risque AVC (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ef4444" if proba >= 0.5 else "#22c55e"},
                "steps": [
                    {"range": [0, 30],  "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef9c3"},
                    {"range": [60, 100],"color": "#fee2e2"}
                ],
                "threshold": {"line": {"color": "red","width": 3},
                               "thickness": 0.75, "value": 50}
            }
        ))
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

        # LIME
        st.markdown("<h3 style='color:#a78bfa'>🔍 Explication LIME</h3>",
                    unsafe_allow_html=True)
        feature_names = [
            "Genre","Âge","Hypertension","Maladie cardiaque","Marié(e)",
            "Type travail","Résidence","Glycémie","IMC","Tabagisme"
        ]
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_scaled,
            feature_names=feature_names,
            class_names=["Faible risque","Risque élevé"],
            mode="classification"
        )
        explanation = explainer.explain_instance(
            X_scaled[0],
            model.predict_proba,
            num_features=10
        )
        fig_lime, ax = plt.subplots(figsize=(8, 4))
        ax.set_facecolor("#0f0c29")
        fig_lime.patch.set_facecolor("#0f0c29")
        vals  = [v for _, v in explanation.as_list()]
        lbls  = [f for f, _ in explanation.as_list()]
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in vals]
        ax.barh(lbls, vals, color=colors)
        ax.set_xlabel("Impact sur la prédiction", color="white")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#475569")
        ax.spines["left"].set_color("#475569")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig_lime)
        plt.close()

        # Rapport PDF
        st.markdown("<h3 style='color:#34d399'>📄 Rapport PDF</h3>",
                    unsafe_allow_html=True)
        if st.button("📥 Télécharger le rapport PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(102, 0, 153)
            pdf.cell(0, 12, "CENTRE NEUROLOGIQUE - RAPPORT AVC", ln=True, align="C")
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "Dauris - Bouchra - Amine", ln=True, align="C")
            pdf.ln(4)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 7, f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
            pdf.cell(0, 7, f"Patient: {prenom} {nom}", ln=True)
            pdf.cell(0, 7, f"Age: {age} ans  |  Genre: {gender}", ln=True)
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_fill_color(102, 0, 153)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 8, "RESULTATS", ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 7, f"Score de risque AVC: {risk_pct:.1f}%", ln=True)
            pdf.cell(0, 7, f"Niveau de risque: {risk_lbl}", ln=True)
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_fill_color(102, 0, 153)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 8, "PARAMETRES CLINIQUES", ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
            for k, v in enc.items():
                pdf.cell(0, 6, f"  {k}: {v}", ln=True)
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_fill_color(102, 0, 153)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 8, "RECOMMANDATION", ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
            reco = ("Consultation médicale urgente recommandée. "
                    "Surveillance tensionnelle et glycémique stricte."
                    if risk_lbl == "ÉLEVÉ" else
                    "Maintenir un mode de vie sain. "
                    "Contrôle médical annuel conseillé.")
            pdf.multi_cell(0, 6, reco)
            pdf.ln(8)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 6, "Ce rapport est généré automatiquement par le système IA TabNet-MLP.", ln=True)

            buf = io.BytesIO()
            pdf.output(buf)
            b64 = base64.b64encode(buf.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="rapport_avc_{nom}_{prenom}.pdf" style="color:#34d399;font-size:1.1rem">📥 Cliquer ici pour télécharger</a>'
            st.markdown(href, unsafe_allow_html=True)

# ─── Dashboard clinique ─────────────────────────────────────
def page_dashboard():
    st.markdown("<h2 style='color:#34d399'>📊 Dashboard Clinique</h2>",
                unsafe_allow_html=True)
    preds = load_predictions()
    if not preds:
        st.info("Aucune prédiction enregistrée pour l'instant.")
        return

    df = pd.DataFrame(preds)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    col1, col2, col3, col4 = st.columns(4)
    total   = len(df)
    n_high  = (df["risk_label"] == "ÉLEVÉ").sum()
    n_low   = (df["risk_label"] == "FAIBLE").sum()
    avg_sc  = df["risk_score"].mean() * 100

    col1.metric("📋 Total analyses",  total)
    col2.metric("⚠️ Risque élevé",   n_high, delta=f"{n_high/total*100:.1f}%")
    col3.metric("✅ Risque faible",   n_low,  delta=f"{n_low/total*100:.1f}%")
    col4.metric("📈 Score moyen",     f"{avg_sc:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(
            values=[n_high, n_low],
            names=["Risque élevé", "Risque faible"],
            color_discrete_sequence=["#ef4444", "#22c55e"],
            title="Répartition des risques"
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               font_color="white")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_hist = px.histogram(
            df, x="risk_score", nbins=20,
            title="Distribution des scores de risque",
            color_discrete_sequence=["#a78bfa"]
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                font_color="white")
        st.plotly_chart(fig_hist, use_container_width=True)

    fig_time = px.scatter(
        df, x="timestamp", y="risk_score",
        color="risk_label",
        color_discrete_map={"ÉLEVÉ": "#ef4444", "FAIBLE": "#22c55e"},
        title="Évolution temporelle des prédictions"
    )
    fig_time.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                            font_color="white")
    st.plotly_chart(fig_time, use_container_width=True)
    st.dataframe(df[["timestamp","user","risk_score","risk_label"]].tail(20),
                 use_container_width=True)

# ─── Navigation principale ───────────────────────────────────
def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color:#a78bfa;text-align:center'>🧠 AVC Predict</h2>",
                    unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b;text-align:center;font-size:0.8rem'>Dauris – Bouchra – Amine</p>",
                    unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        if st.session_state["authenticated"]:
            ud = st.session_state["user_data"]
            st.markdown(f"<p style='color:#94a3b8'>👤 {ud['prenom']} {ud['nom']}</p>",
                        unsafe_allow_html=True)
            menu = st.radio("Navigation", [
                "🏠 Accueil",
                "🔮 Prédiction",
                "📊 Dashboard",
            ])
            st.markdown("<hr>", unsafe_allow_html=True)
            if st.button("🚪 Déconnexion"):
                st.session_state.clear()
                st.rerun()
        else:
            menu = st.radio("Navigation", ["🏠 Accueil", "🔐 Connexion"])

    # Routing
    if menu == "🏠 Accueil":
        page_accueil()
    elif menu == "🔐 Connexion":
        page_connexion()
    elif menu == "🔮 Prédiction":
        if st.session_state["authenticated"]:
            page_prediction()
        else:
            st.warning("🔐 Veuillez vous connecter d'abord.")
    elif menu == "📊 Dashboard":
        if st.session_state["authenticated"]:
            page_dashboard()
        else:
            st.warning("🔐 Veuillez vous connecter d'abord.")

if __name__ == "__main__":
    main()
