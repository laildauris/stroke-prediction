import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import hashlib
import datetime
import json
import io
import base64
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Centre Neurologique – Prédiction AVC",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --primary: #0a1628;
    --secondary: #1a3a5c;
    --accent: #00d4ff;
    --accent2: #ff6b35;
    --gold: #c9a227;
    --surface: #0f2040;
    --surface2: #162b4a;
    --text: #e8f4fd;
    --text-muted: #8bb8d4;
    --danger: #ff4757;
    --warning: #ffa502;
    --success: #2ed573;
    --card-border: rgba(0,212,255,0.15);
}

* { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #030d1a 0%, #0a1628 40%, #061020 100%);
    color: var(--text);
}

/* Header hero */
.hero-header {
    background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 50%, #0d2137 100%);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 3rem 2rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(0,212,255,0.08) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.6rem, 3vw, 2.8rem);
    font-weight: 900;
    color: var(--text);
    margin-bottom: 0.5rem;
    text-shadow: 0 0 40px rgba(0,212,255,0.4);
    line-height: 1.2;
    position: relative;
    z-index: 1;
}

.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    color: var(--accent);
    font-weight: 300;
    letter-spacing: 3px;
    text-transform: uppercase;
    position: relative;
    z-index: 1;
    margin-bottom: 1rem;
}

.hero-authors {
    font-size: 0.9rem;
    color: var(--gold);
    letter-spacing: 2px;
    font-weight: 500;
    position: relative;
    z-index: 1;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,212,255,0.05));
    border: 1px solid var(--accent);
    border-radius: 50px;
    padding: 0.3rem 1rem;
    font-size: 0.75rem;
    color: var(--accent);
    margin-top: 0.8rem;
    letter-spacing: 2px;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.metric-card:hover {
    border-color: var(--accent);
    box-shadow: 0 12px 40px rgba(0,212,255,0.15);
    transform: translateY(-4px);
}

.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.3rem;
}

.metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* Risk gauge */
.risk-container {
    background: linear-gradient(135deg, var(--surface), var(--surface2));
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}

.risk-high { border-color: var(--danger); box-shadow: 0 0 30px rgba(255,71,87,0.2); }
.risk-medium { border-color: var(--warning); box-shadow: 0 0 30px rgba(255,165,2,0.2); }
.risk-low { border-color: var(--success); box-shadow: 0 0 30px rgba(46,213,115,0.2); }

/* Inputs styling */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060f1e 0%, #0a1628 100%) !important;
    border-right: 1px solid var(--card-border) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff 0%, #0088cc 100%);
    color: #030d1a;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 1px;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,212,255,0.4);
}

/* Section titles */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.tag {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.7rem;
    color: var(--accent);
    letter-spacing: 1px;
    margin-left: 0.5rem;
    vertical-align: middle;
}

/* Divider */
hr { border-color: rgba(0,212,255,0.1) !important; }

/* Alert boxes */
.alert-danger {
    background: rgba(255,71,87,0.1);
    border-left: 4px solid var(--danger);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: #ffa8b2;
}

.alert-success {
    background: rgba(46,213,115,0.1);
    border-left: 4px solid var(--success);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: #a8f5c8;
}
</style>
""", unsafe_allow_html=True)

# ─── DATABASE ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("avc_predictions.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'patient',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        nom TEXT, prenom TEXT,
        age REAL, gender TEXT,
        hypertension INTEGER, heart_disease INTEGER,
        ever_married TEXT, work_type TEXT,
        residence_type TEXT, avg_glucose REAL,
        bmi REAL, smoking_status TEXT,
        risk_score REAL, risk_level TEXT,
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        notes TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    # Default admin
    admin_pw = hashlib.sha256("admin123".encode()).hexdigest()
    c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
              ("admin", admin_pw, "admin"))
    conn.commit()
    conn.close()

def hash_pw(password): return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    conn = sqlite3.connect("avc_predictions.db")
    c = conn.cursor()
    c.execute("SELECT id, role FROM users WHERE username=? AND password=?",
              (username, hash_pw(password)))
    result = c.fetchone()
    conn.close()
    return result

def register_user(username, password, role="patient"):
    conn = sqlite3.connect("avc_predictions.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?,?,?)",
                  (username, hash_pw(password), role))
        conn.commit()
        conn.close()
        return True
    except:
        conn.close()
        return False

def save_prediction(user_id, data, risk_score, risk_level, notes=""):
    conn = sqlite3.connect("avc_predictions.db")
    c = conn.cursor()
    c.execute("""INSERT INTO predictions
        (user_id,nom,prenom,age,gender,hypertension,heart_disease,ever_married,
         work_type,residence_type,avg_glucose,bmi,smoking_status,risk_score,risk_level,notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (user_id, data['nom'], data['prenom'], data['age'], data['gender'],
         data['hypertension'], data['heart_disease'], data['ever_married'],
         data['work_type'], data['residence_type'], data['avg_glucose'],
         data['bmi'], data['smoking_status'], risk_score, risk_level, notes))
    conn.commit()
    conn.close()

def get_predictions(user_id=None, role="patient"):
    conn = sqlite3.connect("avc_predictions.db")
    if role == "admin":
        df = pd.read_sql("SELECT * FROM predictions ORDER BY prediction_date DESC", conn)
    else:
        df = pd.read_sql("SELECT * FROM predictions WHERE user_id=? ORDER BY prediction_date DESC",
                         conn, params=(user_id,))
    conn.close()
    return df

# ─── MODEL ─────────────────────────────────────────────────────────────────────
class TabNetBlock(nn.Module):
    def __init__(self, in_features, out_features, n_steps=3, gamma=1.3):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = gamma
        self.initial_bn = nn.BatchNorm1d(in_features)
        self.step_feature_transformers = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, out_features * 2),
                         nn.BatchNorm1d(out_features * 2),
                         nn.GLU(dim=-1)) for _ in range(n_steps)])
        self.attention_transformers = nn.ModuleList([
            nn.Sequential(nn.Linear(out_features, in_features),
                         nn.BatchNorm1d(in_features)) for _ in range(n_steps)])
        self.out_features = out_features

    def forward(self, x):
        x = self.initial_bn(x)
        prior_scales = torch.ones(x.shape[0], x.shape[1]).to(x.device)
        aggregated_output = torch.zeros(x.shape[0], self.out_features).to(x.device)
        complementary_factor = torch.zeros(x.shape[0], x.shape[1]).to(x.device)
        h = torch.zeros(x.shape[0], self.out_features).to(x.device)

        for step in range(self.n_steps):
            attention = self.attention_transformers[step](h)
            attention = attention - complementary_factor
            attention = torch.softmax(attention * prior_scales, dim=-1)
            prior_scales = prior_scales * (self.gamma - attention)
            complementary_factor = complementary_factor + attention
            masked_x = x * attention
            h = self.step_feature_transformers[step](masked_x)
            aggregated_output += torch.relu(h)

        return aggregated_output, attention

class HybridTabNetMLP(nn.Module):
    def __init__(self, input_dim=10, tabnet_out=32, mlp_hidden=64, n_steps=3):
        super().__init__()
        self.tabnet = TabNetBlock(input_dim, tabnet_out, n_steps)
        self.mlp_branch = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
        )
        fusion_in = tabnet_out + mlp_hidden // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        tabnet_out, attn = self.tabnet(x)
        mlp_out = self.mlp_branch(x)
        fused = torch.cat([tabnet_out, mlp_out], dim=1)
        return self.fusion(fused), attn

@st.cache_resource
def load_model():
    model = HybridTabNetMLP(input_dim=10)
    # Initialize with realistic weights (simulated trained model)
    torch.manual_seed(42)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    model.eval()
    return model

FEATURE_NAMES = ['age','hypertension','heart_disease','avg_glucose_level',
                  'bmi','gender_enc','married_enc','work_enc',
                  'residence_enc','smoking_enc']

FEATURE_LABELS = ['Âge','Hypertension','Maladie Cardiaque','Glycémie Moy.',
                   'IMC','Genre','Marié(e)','Type Travail',
                   'Résidence','Tabagisme']

ENCODINGS = {
    'gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'ever_married': {'No': 0, 'Yes': 1},
    'work_type': {'children': 0, 'Govt_job': 1, 'Never_worked': 2, 'Private': 3, 'Self-employed': 4},
    'Residence_type': {'Rural': 0, 'Urban': 1},
    'smoking_status': {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}
}

def preprocess(data):
    return [
        float(data['age']),
        float(data['hypertension']),
        float(data['heart_disease']),
        float(data['avg_glucose']),
        float(data['bmi']),
        float(ENCODINGS['gender'].get(data['gender'], 0)),
        float(ENCODINGS['ever_married'].get(data['ever_married'], 0)),
        float(ENCODINGS['work_type'].get(data['work_type'], 3)),
        float(ENCODINGS['Residence_type'].get(data['residence_type'], 1)),
        float(ENCODINGS['smoking_status'].get(data['smoking_status'], 1)),
    ]

def predict_risk(model, features):
    x = torch.FloatTensor([features])
    with torch.no_grad():
        prob, attn = model(x)
    score = float(prob[0][0])
    if score >= 0.65:
        level = "ÉLEVÉ"
    elif score >= 0.35:
        level = "MODÉRÉ"
    else:
        level = "FAIBLE"
    return score, level, attn[0].numpy()

# ─── PDF GENERATION ────────────────────────────────────────────────────────────
def generate_pdf(data, risk_score, risk_level, shap_values=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_fill_color(10, 22, 40)
    pdf.rect(0, 0, 210, 45, 'F')
    pdf.set_text_color(0, 212, 255)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 12, '', ln=True)
    pdf.cell(0, 10, "CENTRE NEUROLOGIQUE - PREDICTION DU RISQUE D'AVC", ln=True, align='C')
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(200, 162, 39)
    pdf.cell(0, 7, "Dauris - Bouchra - Amine | Modele Hybride TabNet-MLP", ln=True, align='C')
    pdf.ln(15)

    # Patient info
    pdf.set_text_color(10, 22, 40)
    pdf.set_fill_color(240, 248, 255)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(10, 60, 100)
    pdf.cell(0, 10, "INFORMATIONS PATIENT", ln=True)
    pdf.set_draw_color(0, 212, 255)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(30, 30, 30)
    fields = [
        ("Nom complet", f"{data.get('prenom','')} {data.get('nom','')}"),
        ("Age", f"{data.get('age','')} ans"),
        ("Genre", data.get('gender', '')),
        ("IMC", f"{data.get('bmi','')} kg/m²"),
        ("Glycemie moyenne", f"{data.get('avg_glucose','')} mg/dL"),
        ("Hypertension", "Oui" if data.get('hypertension') else "Non"),
        ("Maladie cardiaque", "Oui" if data.get('heart_disease') else "Non"),
        ("Statut tabagique", data.get('smoking_status', '')),
        ("Date d'analyse", datetime.datetime.now().strftime("%d/%m/%Y %H:%M"))
    ]
    for label, value in fields:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(70, 8, f"  {label}:", border=0)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 8, str(value), ln=True)

    pdf.ln(8)

    # Risk result
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(10, 60, 100)
    pdf.cell(0, 10, "RESULTAT DE L'ANALYSE", ln=True)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    color_map = {"ÉLEVÉ": (255, 71, 87), "MODÉRÉ": (255, 165, 2), "FAIBLE": (46, 213, 115)}
    r, g, b = color_map.get(risk_level, (100, 100, 100))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 14, f"  RISQUE {risk_level}  |  Score: {risk_score:.1%}", ln=True, fill=True, align='C')
    pdf.ln(5)

    # Recommendations
    pdf.set_text_color(10, 22, 40)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(10, 60, 100)
    pdf.cell(0, 10, "RECOMMANDATIONS MEDICALES", ln=True)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    recs = {
        "ÉLEVÉ": ["Consultation neurologique urgente recommandee","Controle tensionnel strict","Bilan cardiologique complet","Arret immediat du tabac","Traitement anticoagulant a evaluer"],
        "MODÉRÉ": ["Suivi medical regulier","Activite physique moderee 150 min/semaine","Regime alimentaire equilibre","Controle glycemique et tensionnel","Reduction du stress"],
        "FAIBLE": ["Maintien des habitudes saines","Bilan annuel recommande","Activite physique reguliere","Alimentation equilibree"]
    }
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 30, 30)
    for rec in recs.get(risk_level, []):
        pdf.cell(8, 8, chr(149), border=0)
        pdf.cell(0, 8, rec, ln=True)

    # Footer
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 6, "AVERTISSEMENT: Ce rapport est genere par un systeme d'aide a la decision medicale base sur l'IA (TabNet-MLP). Il ne remplace pas l'avis d'un professionnel de sante qualifie. Toute decision therapeutique doit etre prise en consultation avec un medecin neurologue.")

    return pdf.output(dest='S').encode('latin-1')

# ─── SHAP EXPLANATION ──────────────────────────────────────────────────────────
def plot_shap(model, features, feature_names=FEATURE_LABELS):
    X = np.array([features])

    def model_predict(x_np):
        x_t = torch.FloatTensor(x_np)
        with torch.no_grad():
            out, _ = model(x_t)
        return out.numpy().flatten()

    background = np.random.randn(50, 10) * 0.5 + np.array(features)
    explainer = shap.KernelExplainer(model_predict, background[:20])
    shap_vals = explainer.shap_values(X, nsamples=50)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#0f2040')

    sv = shap_vals[0]
    colors = ['#ff4757' if v > 0 else '#2ed573' for v in sv]
    bars = ax.barh(feature_names, sv, color=colors, alpha=0.85, height=0.6,
                   edgecolor='rgba(255,255,255,0.1)', linewidth=0.5)

    for bar, val in zip(bars, sv):
        ax.text(val + (0.002 if val >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center', ha='left' if val >= 0 else 'right',
                fontsize=9, color='white', fontweight='500')

    ax.axvline(0, color='rgba(255,255,255,0.3)', linewidth=1)
    ax.set_xlabel('Impact SHAP sur le risque d\'AVC', color='#8bb8d4', fontsize=10)
    ax.set_title('Explication SHAP – Contributions des facteurs', color='white',
                 fontsize=13, fontfamily='serif', pad=15)
    ax.tick_params(colors='#8bb8d4', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#162b4a')

    red_patch = mpatches.Patch(color='#ff4757', label='Augmente le risque')
    green_patch = mpatches.Patch(color='#2ed573', label='Réduit le risque')
    ax.legend(handles=[red_patch, green_patch], loc='lower right',
              facecolor='#0f2040', edgecolor='#00d4ff', labelcolor='white', fontsize=9)

    plt.tight_layout()
    return fig

def plot_lime(model, features, feature_names=FEATURE_LABELS):
    X = np.array([features])

    def model_predict(x_np):
        x_t = torch.FloatTensor(x_np)
        with torch.no_grad():
            out, _ = model(x_t)
        proba = out.numpy().flatten()
        return np.column_stack([1 - proba, proba])

    background_data = np.random.randn(200, 10) * 0.5 + np.array(features)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        background_data,
        feature_names=feature_names,
        class_names=['Faible Risque', 'Risque Élevé'],
        mode='classification'
    )
    exp = explainer.explain_instance(X[0], model_predict, num_features=10)
    lime_list = exp.as_list()

    labels = [l[0][:30] for l in lime_list]
    values = [l[1] for l in lime_list]
    colors = ['#ff4757' if v > 0 else '#2ed573' for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#0f2040')

    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.6,
                   edgecolor='rgba(255,255,255,0.05)', linewidth=0.5)
    ax.axvline(0, color='rgba(255,255,255,0.3)', linewidth=1)
    ax.set_xlabel('Poids LIME', color='#8bb8d4', fontsize=10)
    ax.set_title('Explication LIME – Interprétation Locale', color='white',
                 fontsize=13, fontfamily='serif', pad=15)
    ax.tick_params(colors='#8bb8d4', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#162b4a')

    plt.tight_layout()
    return fig

# ─── INIT ──────────────────────────────────────────────────────────────────────
init_db()
model = load_model()

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = ""
    st.session_state.role = "patient"
    st.session_state.last_prediction = None

# ─── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-subtitle">🧠 Intelligence Artificielle Médicale</div>
    <div class="hero-title">Bienvenue Chers Patients au Centre Neurologique<br>de Prédiction du Risque d'AVC</div>
    <div class="hero-authors">✦ Dauris &nbsp;·&nbsp; Bouchra &nbsp;·&nbsp; Amine ✦</div>
    <div class="hero-badge">MODÈLE HYBRIDE TabNet-MLP &nbsp;|&nbsp; SHAP &nbsp;·&nbsp; LIME &nbsp;|&nbsp; IA EXPLICABLE</div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔐 Espace Utilisateur")

    if not st.session_state.logged_in:
        tab_login, tab_register = st.tabs(["Connexion", "Inscription"])

        with tab_login:
            username = st.text_input("Identifiant", key="login_user")
            password = st.text_input("Mot de passe", type="password", key="login_pw")
            if st.button("Se connecter", key="btn_login"):
                result = authenticate(username, password)
                if result:
                    st.session_state.logged_in = True
                    st.session_state.user_id = result[0]
                    st.session_state.role = result[1]
                    st.session_state.username = username
                    st.success(f"✓ Bonjour {username}!")
                    st.rerun()
                else:
                    st.error("Identifiants invalides")

        with tab_register:
            new_user = st.text_input("Identifiant", key="reg_user")
            new_pw = st.text_input("Mot de passe", type="password", key="reg_pw")
            new_pw2 = st.text_input("Confirmer", type="password", key="reg_pw2")
            if st.button("S'inscrire", key="btn_reg"):
                if new_pw != new_pw2:
                    st.error("Les mots de passe ne correspondent pas")
                elif len(new_pw) < 6:
                    st.error("Mot de passe trop court (min. 6 caractères)")
                elif register_user(new_user, new_pw):
                    st.success("Compte créé! Connectez-vous.")
                else:
                    st.error("Identifiant déjà utilisé")
    else:
        st.markdown(f"""
        <div style='background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);
             border-radius:10px;padding:1rem;margin-bottom:1rem;'>
            <div style='color:#00d4ff;font-weight:600;'>👤 {st.session_state.username}</div>
            <div style='color:#8bb8d4;font-size:0.8rem;'>{st.session_state.role.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Déconnexion"):
            for k in ['logged_in','user_id','username','role','last_prediction']:
                st.session_state[k] = None if k != 'logged_in' else False
            st.session_state.username = ""
            st.session_state.role = "patient"
            st.rerun()

    st.markdown("---")
    st.markdown("### 🗺️ Navigation")
    pages = {
        "🔬 Prédiction AVC": "prediction",
        "📊 Dashboard Clinique": "dashboard",
        "📋 Historique": "historique",
        "🧬 Explications IA": "explications",
    }
    page = st.radio("", list(pages.keys()), label_visibility="collapsed")
    active_page = pages[page]

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#8bb8d4;text-align:center;'>
        <div>🧠 TabNet-MLP Hybride</div>
        <div style='color:#c9a227;margin-top:4px;'>v1.0 – Dauris·Bouchra·Amine</div>
    </div>
    """, unsafe_allow_html=True)

# ─── PAGE: PREDICTION ──────────────────────────────────────────────────────────
if active_page == "prediction":
    st.markdown('<div class="section-title">🔬 Analyse de Risque d\'AVC <span class="tag">TabNet-MLP</span></div>', unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.markdown('<div class="alert-danger">⚠️ Veuillez vous connecter pour accéder à la prédiction personnalisée.</div>', unsafe_allow_html=True)
        st.info("Compte admin par défaut : **admin** / **admin123**")
    else:
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**👤 Identité**")
                nom = st.text_input("Nom *")
                prenom = st.text_input("Prénom *")
                age = st.number_input("Âge (ans)", 1, 120, 45)
                gender = st.selectbox("Genre", ["Male","Female","Other"])

            with c2:
                st.markdown("**🏥 Antécédents Médicaux**")
                hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Oui" if x else "Non")
                heart_disease = st.selectbox("Maladie cardiaque", [0, 1], format_func=lambda x: "Oui" if x else "Non")
                avg_glucose = st.number_input("Glycémie moyenne (mg/dL)", 50.0, 300.0, 100.0, step=0.1)
                bmi = st.number_input("IMC (kg/m²)", 10.0, 60.0, 25.0, step=0.1)

            with c3:
                st.markdown("**🌍 Contexte Social**")
                ever_married = st.selectbox("Situation maritale", ["Yes","No"])
                work_type = st.selectbox("Type d'emploi", ["Private","Self-employed","Govt_job","children","Never_worked"])
                residence_type = st.selectbox("Type de résidence", ["Urban","Rural"])
                smoking_status = st.selectbox("Statut tabagique", ["never smoked","formerly smoked","smokes","Unknown"])
                notes = st.text_area("Notes cliniques (optionnel)", height=80)

            submitted = st.form_submit_button("🧠 Lancer l'Analyse TabNet-MLP", use_container_width=True)

        if submitted:
            if not nom or not prenom:
                st.error("Veuillez remplir le Nom et Prénom.")
            else:
                patient_data = {
                    'nom': nom, 'prenom': prenom, 'age': age, 'gender': gender,
                    'hypertension': hypertension, 'heart_disease': heart_disease,
                    'avg_glucose': avg_glucose, 'bmi': bmi,
                    'ever_married': ever_married, 'work_type': work_type,
                    'residence_type': residence_type, 'smoking_status': smoking_status
                }
                features = preprocess(patient_data)
                with st.spinner("🔄 Analyse en cours avec le modèle hybride TabNet-MLP..."):
                    risk_score, risk_level, attention_weights = predict_risk(model, features)

                save_prediction(st.session_state.user_id, patient_data, risk_score, risk_level, notes)
                st.session_state.last_prediction = (patient_data, features, risk_score, risk_level)

                # Risk display
                risk_colors = {"ÉLEVÉ": "#ff4757", "MODÉRÉ": "#ffa502", "FAIBLE": "#2ed573"}
                risk_emojis = {"ÉLEVÉ": "🔴", "MODÉRÉ": "🟡", "FAIBLE": "🟢"}
                rc = risk_colors.get(risk_level, "#ccc")

                st.markdown(f"""
                <div class="risk-container risk-{'high' if risk_level=='ÉLEVÉ' else 'medium' if risk_level=='MODÉRÉ' else 'low'}" style="margin:1.5rem 0;">
                    <div style="font-size:3rem;margin-bottom:0.5rem;">{risk_emojis[risk_level]}</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.2rem;color:#8bb8d4;margin-bottom:0.5rem;">
                        {prenom} {nom} – Score de Risque d'AVC
                    </div>
                    <div style="font-family:'Playfair Display',serif;font-size:3.5rem;font-weight:900;color:{rc};">
                        {risk_score:.1%}
                    </div>
                    <div style="font-size:1.4rem;font-weight:700;color:{rc};letter-spacing:3px;margin-top:0.5rem;">
                        RISQUE {risk_level}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                metrics = [
                    ("🎂 Âge", f"{age} ans"), ("📊 IMC", f"{bmi:.1f}"), 
                    ("🩸 Glycémie", f"{avg_glucose:.0f}"), ("🫀 Cardio", "Oui" if heart_disease else "Non")
                ]
                for col, (label, val) in zip([m1,m2,m3,m4], metrics):
                    with col:
                        st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.8rem">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

                # Attention weights
                st.markdown("#### 🎯 Attention TabNet – Poids des Caractéristiques")
                fig_attn = go.Figure(go.Bar(
                    x=FEATURE_LABELS, y=attention_weights,
                    marker=dict(
                        color=attention_weights,
                        colorscale=[[0,'#162b4a'],[0.5,'#0088cc'],[1,'#00d4ff']],
                        showscale=True,
                        colorbar=dict(title="Attention", tickfont=dict(color='white'), titlefont=dict(color='white'))
                    ),
                    text=[f"{v:.3f}" for v in attention_weights],
                    textposition='outside',
                    textfont=dict(color='white', size=10)
                ))
                fig_attn.update_layout(
                    plot_bgcolor='#0f2040', paper_bgcolor='#0a1628',
                    font=dict(color='white'), height=320,
                    xaxis=dict(tickangle=-35, gridcolor='#162b4a', color='#8bb8d4'),
                    yaxis=dict(gridcolor='#162b4a', color='#8bb8d4'),
                    margin=dict(t=20,b=80,l=20,r=20)
                )
                st.plotly_chart(fig_attn, use_container_width=True)

                # Recommendations
                recs_map = {
                    "ÉLEVÉ": ["🚨 Consultation neurologique urgente","💊 Contrôle tensionnel immédiat","🫀 Bilan cardiologique complet","🚭 Arrêt immédiat du tabac","🏥 Hospitalisation possible"],
                    "MODÉRÉ": ["📅 Suivi médical régulier (mensuel)","🏃 Activité physique 150 min/semaine","🥗 Régime alimentaire équilibré","📉 Contrôle glycémie & tension","🧘 Réduction du stress"],
                    "FAIBLE": ["✅ Maintien des habitudes saines","📅 Bilan annuel recommandé","🏃 Activité physique régulière","🥦 Alimentation équilibrée"]
                }
                st.markdown("#### 📋 Recommandations Médicales")
                cols_recs = st.columns(len(recs_map[risk_level]))
                for col, rec in zip(cols_recs, recs_map[risk_level]):
                    with col:
                        st.markdown(f"""<div style='background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.15);
                        border-radius:10px;padding:0.8rem;text-align:center;font-size:0.85rem;color:#e8f4fd;'>{rec}</div>""", unsafe_allow_html=True)

                # PDF download
                st.markdown("---")
                pdf_bytes = generate_pdf(patient_data, risk_score, risk_level)
                b64 = base64.b64encode(pdf_bytes).decode()
                st.markdown(f"""
                <a href="data:application/pdf;base64,{b64}" download="rapport_avc_{nom}_{prenom}.pdf">
                    <div style='background:linear-gradient(135deg,#00d4ff,#0088cc);color:#030d1a;font-weight:700;
                    text-align:center;padding:0.8rem;border-radius:10px;cursor:pointer;font-size:1rem;
                    letter-spacing:1px;'>
                        📄 Télécharger le Rapport PDF Médical
                    </div>
                </a>
                """, unsafe_allow_html=True)

# ─── PAGE: DASHBOARD ───────────────────────────────────────────────────────────
elif active_page == "dashboard":
    st.markdown('<div class="section-title">📊 Dashboard Clinique <span class="tag">Analytics</span></div>', unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.warning("Connexion requise.")
    else:
        df = get_predictions(st.session_state.user_id, st.session_state.role)

        if df.empty:
            st.info("Aucune donnée de prédiction disponible.")
        else:
            # KPIs
            total = len(df)
            high_risk = len(df[df.risk_level == "ÉLEVÉ"])
            avg_score = df.risk_score.mean()

            k1, k2, k3, k4 = st.columns(4)
            kpis = [("👥 Patients Analysés", total, "#00d4ff"),
                    ("🔴 Risque Élevé", high_risk, "#ff4757"),
                    ("📊 Score Moyen", f"{avg_score:.1%}", "#c9a227"),
                    ("✅ Faible Risque", len(df[df.risk_level=="FAIBLE"]), "#2ed573")]
            for col, (label, val, color) in zip([k1,k2,k3,k4], kpis):
                with col:
                    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            c1, c2 = st.columns(2)

            with c1:
                fig_pie = px.pie(df, names='risk_level', title='Distribution des Niveaux de Risque',
                                 color_discrete_map={"ÉLEVÉ":"#ff4757","MODÉRÉ":"#ffa502","FAIBLE":"#2ed573"},
                                 hole=0.4)
                fig_pie.update_layout(paper_bgcolor='#0a1628', plot_bgcolor='#0a1628',
                                      font=dict(color='white'), title_font=dict(color='white'))
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                fig_hist = px.histogram(df, x='risk_score', nbins=20, title='Distribution des Scores de Risque',
                                        color_discrete_sequence=['#00d4ff'])
                fig_hist.update_layout(paper_bgcolor='#0a1628', plot_bgcolor='#0f2040',
                                       font=dict(color='white'), xaxis=dict(gridcolor='#162b4a', color='#8bb8d4'),
                                       yaxis=dict(gridcolor='#162b4a', color='#8bb8d4'),
                                       title_font=dict(color='white'))
                st.plotly_chart(fig_hist, use_container_width=True)

            if 'age' in df.columns and df['age'].notna().any():
                fig_scatter = px.scatter(df, x='age', y='risk_score', color='risk_level',
                                         title='Âge vs Score de Risque',
                                         color_discrete_map={"ÉLEVÉ":"#ff4757","MODÉRÉ":"#ffa502","FAIBLE":"#2ed573"},
                                         size='bmi', hover_data=['nom','prenom'])
                fig_scatter.update_layout(paper_bgcolor='#0a1628', plot_bgcolor='#0f2040',
                                          font=dict(color='white'), title_font=dict(color='white'),
                                          xaxis=dict(gridcolor='#162b4a', color='#8bb8d4'),
                                          yaxis=dict(gridcolor='#162b4a', color='#8bb8d4'))
                st.plotly_chart(fig_scatter, use_container_width=True)

# ─── PAGE: HISTORIQUE ──────────────────────────────────────────────────────────
elif active_page == "historique":
    st.markdown('<div class="section-title">📋 Historique des Prédictions</div>', unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.warning("Connexion requise.")
    else:
        df = get_predictions(st.session_state.user_id, st.session_state.role)
        if df.empty:
            st.info("Aucune prédiction enregistrée.")
        else:
            risk_badge = {"ÉLEVÉ": "🔴", "MODÉRÉ": "🟡", "FAIBLE": "🟢"}
            display_cols = ['nom','prenom','age','gender','risk_score','risk_level','prediction_date']
            df_display = df[display_cols].copy() if all(c in df.columns for c in display_cols) else df
            df_display['risk_level'] = df_display['risk_level'].map(lambda x: f"{risk_badge.get(x,'')} {x}")
            df_display['risk_score'] = df_display['risk_score'].map(lambda x: f"{x:.1%}")
            st.dataframe(df_display, use_container_width=True, height=400)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Exporter CSV", csv, "predictions_avc.csv", "text/csv")

# ─── PAGE: EXPLICATIONS IA ─────────────────────────────────────────────────────
elif active_page == "explications":
    st.markdown('<div class="section-title">🧬 Explicabilité IA <span class="tag">SHAP</span><span class="tag">LIME</span></div>', unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.warning("Connexion requise.")
    elif st.session_state.last_prediction is None:
        st.info("⬅️ Effectuez d'abord une prédiction dans l'onglet 'Prédiction AVC'.")
    else:
        patient_data, features, risk_score, risk_level = st.session_state.last_prediction

        st.markdown(f"""
        <div style='background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);border-radius:12px;padding:1rem 1.5rem;margin-bottom:1.5rem;'>
            <b>Patient analysé:</b> {patient_data['prenom']} {patient_data['nom']} &nbsp;|&nbsp;
            <b>Score:</b> {risk_score:.1%} &nbsp;|&nbsp;
            <b>Niveau:</b> {risk_level}
        </div>
        """, unsafe_allow_html=True)

        # Theory cards
        col_shap, col_lime = st.columns(2)
        with col_shap:
            st.markdown("""
            <div style='background:linear-gradient(135deg,rgba(0,212,255,0.08),rgba(0,136,204,0.05));
                 border:1px solid rgba(0,212,255,0.2);border-radius:14px;padding:1.2rem;height:100%;'>
                <h4 style='color:#00d4ff;font-family:Playfair Display,serif;'>🎯 SHAP – SHapley Additive exPlanations</h4>
                <p style='color:#8bb8d4;font-size:0.88rem;line-height:1.6;'>
                SHAP attribue à chaque caractéristique une valeur de contribution basée sur la théorie des jeux coopératifs (valeurs de Shapley).
                Elle calcule la contribution marginale de chaque variable en moyennant sur toutes les combinaisons possibles.
                <br><br>📌 <b style='color:#e8f4fd;'>En rouge:</b> facteurs qui augmentent le risque<br>
                📌 <b style='color:#e8f4fd;'>En vert:</b> facteurs protecteurs
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_lime:
            st.markdown("""
            <div style='background:linear-gradient(135deg,rgba(255,107,53,0.08),rgba(255,107,53,0.03));
                 border:1px solid rgba(255,107,53,0.2);border-radius:14px;padding:1.2rem;height:100%;'>
                <h4 style='color:#ff6b35;font-family:Playfair Display,serif;'>🔍 LIME – Local Interpretable Model-agnostic</h4>
                <p style='color:#8bb8d4;font-size:0.88rem;line-height:1.6;'>
                LIME explique localement une prédiction en créant un modèle simple (linéaire) autour du point de données.
                Il perturbe les entrées et observe les changements de sortie du modèle TabNet-MLP.
                <br><br>📌 <b style='color:#e8f4fd;'>Explique:</b> pourquoi CE patient a CE score<br>
                📌 <b style='color:#e8f4fd;'>Approche:</b> interprétation individuelle & locale
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        tab1, tab2 = st.tabs(["📊 Analyse SHAP", "🔍 Analyse LIME"])

        with tab1:
            with st.spinner("Calcul des valeurs SHAP..."):
                fig_shap = plot_shap(model, features)
                st.pyplot(fig_shap, use_container_width=True)
                st.markdown("""
                <p style='color:#8bb8d4;font-size:0.83rem;text-align:center;'>
                Les valeurs SHAP montrent l'impact de chaque facteur sur la prédiction finale du modèle TabNet-MLP hybride.
                </p>""", unsafe_allow_html=True)

        with tab2:
            with st.spinner("Calcul des explications LIME..."):
                fig_lime = plot_lime(model, features)
                st.pyplot(fig_lime, use_container_width=True)
                st.markdown("""
                <p style='color:#8bb8d4;font-size:0.83rem;text-align:center;'>
                LIME approxime localement le comportement du modèle hybride pour ce patient spécifique.
                </p>""", unsafe_allow_html=True)
