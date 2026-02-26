# ════════════════════════════════════════════════════════════════════════════
#  ChurnShield AI — Premium Streamlit Dashboard
#  Microsoft Elevate Internship Project
# ════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ChurnShield AI", page_icon="🛡️", layout="wide", initial_sidebar_state="auto")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Grotesk:wght@300;400;500;600&display=swap');
:root {
    --bg:     #050B18;
    --card:   #0D1B2E;
    --glass:  rgba(13,27,46,0.7);
    --border: rgba(99,179,237,0.12);
    --glow:   rgba(139,92,246,0.4);
    --indigo: #6366F1; --violet: #8B5CF6; --cyan: #22D3EE;
    --green:  #10B981; --rose:   #F43F5E; --amber: #FBBF24;
    --txt:    #E2E8F0; --muted:  #64748B; --bright:#F8FAFC;
}
html,body,[class*="css"],.stApp{font-family:'Space Grotesk',sans-serif!important;background-color:var(--bg)!important;color:var(--txt)!important;}
.stApp{background:radial-gradient(ellipse at 10% 20%,rgba(99,102,241,.08) 0%,transparent 50%),radial-gradient(ellipse at 90% 80%,rgba(139,92,246,.08) 0%,transparent 50%),var(--bg)!important;background-attachment:fixed!important;}

/* ── HIDE ONLY FOOTER AND DECORATION — keep header/hamburger visible ── */
#MainMenu,footer,[data-testid="stDecoration"]{visibility:hidden!important;display:none!important;}

/* ── HEADER — dark themed, always visible for hamburger ── */
[data-testid="stHeader"]{
    background:linear-gradient(90deg,#080F1E,#0D1528)!important;
    border-bottom:1px solid rgba(99,102,241,.25)!important;
    backdrop-filter:blur(12px)!important;
}

/* ── HAMBURGER BUTTON — make it glow so mobile users can find it ── */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"]{
    background:rgba(99,102,241,.25)!important;
    border:1px solid rgba(99,102,241,.6)!important;
    border-radius:10px!important;
    padding:4px!important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg{
    color:#A5B4FC!important;
    fill:#A5B4FC!important;
    width:20px!important;
    height:20px!important;
}
/* Also style the expand arrow inside sidebar */
[data-testid="stSidebarCollapseButton"] button{
    background:rgba(99,102,241,.2)!important;
    border-radius:8px!important;
    color:#A5B4FC!important;
}

/* ── LAYOUT ── */
.block-container{padding:1.5rem 2.5rem 3rem!important;max-width:1400px!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#080F1E,#0D1528)!important;border-right:1px solid rgba(99,102,241,.2)!important;}

/* ── MOBILE RESPONSIVE ── */
@media(max-width:768px){
    .block-container{padding:0.8rem 0.8rem 2rem!important;}
    .hero{padding:1.2rem 1rem!important;border-radius:16px!important;}
    .hero-title{font-size:1.6rem!important;line-height:1.2!important;}
    .hero-sub{font-size:.85rem!important;}
    .hero-badge{font-size:.65rem!important;}
    .kpi{padding:.9rem 1rem!important;border-radius:14px!important;}
    .kpi-val{font-size:1.5rem!important;}
    .kpi-lbl{font-size:.65rem!important;}
    .ig{grid-template-columns:1fr!important;}
    .ccard{padding:1rem .9rem .8rem!important;}
}

/* HERO */
.hero{background:linear-gradient(135deg,rgba(99,102,241,.15) 0%,rgba(13,27,46,.95) 40%,rgba(139,92,246,.1) 100%);border:1px solid rgba(99,102,241,.3);border-radius:24px;padding:3rem 3.5rem;margin-bottom:2rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-80px;right:-80px;width:320px;height:320px;background:radial-gradient(circle,rgba(139,92,246,.2) 0%,transparent 65%);border-radius:50%;}
.hero::after{content:'';position:absolute;bottom:-60px;left:20%;width:250px;height:250px;background:radial-gradient(circle,rgba(34,211,238,.12) 0%,transparent 65%);border-radius:50%;}
.hero-title{font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;background:linear-gradient(135deg,#A5B4FC 0%,#C4B5FD 40%,#67E8F9 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 .75rem;line-height:1.1;letter-spacing:-.02em;}
.hero-sub{color:var(--muted);font-size:1.05rem;font-weight:300;max-width:580px;line-height:1.6;}
.hero-badge{display:inline-flex;align-items:center;background:linear-gradient(135deg,rgba(99,102,241,.2),rgba(139,92,246,.2));border:1px solid rgba(139,92,246,.4);color:#C4B5FD;font-size:.72rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;padding:.35rem 1rem;border-radius:999px;margin-bottom:1.2rem;}

/* KPI CARDS */
.kpi{background:var(--glass);backdrop-filter:blur(16px);border:1px solid var(--border);border-radius:18px;padding:1.5rem 1.8rem;position:relative;overflow:hidden;transition:transform .25s,border-color .25s,box-shadow .25s;}
.kpi:hover{transform:translateY(-4px);border-color:var(--glow);box-shadow:0 12px 40px rgba(139,92,246,.15);}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:18px 18px 0 0;}
.kpi-i::before{background:linear-gradient(90deg,#6366F1,#8B5CF6);}
.kpi-r::before{background:linear-gradient(90deg,#F43F5E,#FBBF24);}
.kpi-g::before{background:linear-gradient(90deg,#10B981,#22D3EE);}
.kpi-c::before{background:linear-gradient(90deg,#22D3EE,#6366F1);}
.kpi-lbl{font-size:.7rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:.6rem;}
.kpi-val{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:var(--bright);line-height:1;margin-bottom:.4rem;}
.kpi-sub{font-size:.78rem;color:var(--muted);}

/* CHART CARDS */
.ccard{background:var(--glass);backdrop-filter:blur(16px);border:1px solid var(--border);border-radius:18px;padding:1.5rem 1.8rem 1rem;margin-bottom:1.2rem;transition:border-color .25s;}
.ccard:hover{border-color:rgba(99,102,241,.3);}
.chead{font-family:'Syne',sans-serif;font-size:.78rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:.8rem;}

/* INSIGHTS */
.ig{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1rem;}
.ip{background:rgba(13,27,46,.8);border-radius:14px;padding:1.2rem 1.4rem;border:1px solid var(--border);transition:transform .2s;}
.ip:hover{transform:translateY(-2px);}
.in{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;margin-bottom:.3rem;}
.it{font-size:.82rem;color:var(--muted);line-height:1.5;}

/* RESULT */
.res{border-radius:20px;padding:2.5rem 2rem;text-align:center;margin:1.5rem 0;}
.res-c{background:linear-gradient(135deg,rgba(244,63,94,.12),rgba(244,63,94,.04));border:1px solid rgba(244,63,94,.35);}
.res-s{background:linear-gradient(135deg,rgba(16,185,129,.12),rgba(16,185,129,.04));border:1px solid rgba(16,185,129,.35);}
.res-icon{font-size:3.5rem;margin-bottom:.6rem;}
.res-title{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;margin-bottom:.5rem;}
.res-desc{font-size:1rem;color:var(--muted);line-height:1.5;}

/* ── NAVIGATION RADIO BUTTONS ── make text bright white, active = glowing ── */
[data-testid="stSidebar"] .stRadio > div {gap:.3rem!important;}
[data-testid="stSidebar"] .stRadio label {
    display:flex!important;align-items:center!important;
    padding:.65rem 1rem!important;border-radius:12px!important;
    border:1px solid transparent!important;
    font-size:.9rem!important;font-weight:500!important;
    color:#94A3B8!important;                /* default: muted grey */
    cursor:pointer!important;
    transition:all .2s ease!important;
    margin:0!important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background:rgba(99,102,241,.12)!important;
    border-color:rgba(99,102,241,.3)!important;
    color:#E0E7FF!important;               /* bright on hover */
}
/* Active/selected nav item */
[data-testid="stSidebar"] .stRadio label[data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] input[type="radio"]:checked + div label,
[data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ label,
[data-testid="stSidebar"] .stRadio label:has(> div > p) {
    color:#FFFFFF!important;
}
/* Force all radio label text to be visible white */
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stRadio label span {
    color:#CBD5E1!important;
    font-weight:500!important;
}
/* Hide the actual radio circle dot */
[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {display:none!important;}
[data-testid="stSidebar"] input[type="radio"] {display:none!important;}

/* ── SEPARATION LINES ── glowing dividers ── */
.glowline {
    height:1px;
    background:linear-gradient(90deg, transparent, rgba(99,102,241,.5), rgba(139,92,246,.5), transparent);
    margin:1.2rem 0;
    border:none;
}
.sec{
    font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;
    color:var(--bright);margin:1.8rem 0 1rem;
    display:flex;align-items:center;gap:.8rem;
}
.sec::after{
    content:'';flex:1;height:1px;
    background:linear-gradient(90deg,rgba(99,102,241,.4),rgba(139,92,246,.2),transparent);
}

/* ── HERO TITLE — more vivid gradient ── */
.hero-title{
    font-family:'Syne',sans-serif!important;
    font-size:3.2rem!important;
    font-weight:800!important;
    background:linear-gradient(135deg,#FFFFFF 0%,#C4B5FD 35%,#67E8F9 70%,#A5F3FC 100%)!important;
    -webkit-background-clip:text!important;
    -webkit-text-fill-color:transparent!important;
    background-clip:text!important;
    margin:0 0 .75rem!important;
    line-height:1.08!important;
    letter-spacing:-.03em!important;
    text-shadow:none!important;
}

/* ── CARD OUTLINES — visible glowing borders ── */
.kpi {
    border:1px solid rgba(99,102,241,.35)!important;
    box-shadow:0 0 0 0 transparent,inset 0 1px 0 rgba(255,255,255,.06)!important;
}
.kpi:hover {
    border-color:rgba(139,92,246,.7)!important;
    box-shadow:0 8px 32px rgba(99,102,241,.2), 0 0 0 1px rgba(139,92,246,.3)!important;
}
.ccard {
    border:1px solid rgba(99,102,241,.28)!important;
}
.ccard:hover {
    border-color:rgba(99,102,241,.55)!important;
    box-shadow:0 0 20px rgba(99,102,241,.1)!important;
}

/* ── MISC ── */
.stProgress>div>div>div>div{background:linear-gradient(90deg,#6366F1,#8B5CF6,#22D3EE)!important;border-radius:999px!important;}
.stProgress>div>div{background:rgba(99,102,241,.15)!important;border-radius:999px!important;}
.stButton>button{background:linear-gradient(135deg,#6366F1,#8B5CF6,#7C3AED)!important;color:#F8FAFC!important;border:none!important;border-radius:14px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:1rem!important;padding:.85rem 2rem!important;width:100%!important;box-shadow:0 4px 20px rgba(99,102,241,.35)!important;transition:opacity .2s,transform .15s!important;}
.stButton>button:hover{opacity:.9!important;transform:translateY(-1px)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--glass)!important;border-radius:14px!important;padding:.35rem!important;border:1px solid rgba(99,102,241,.3)!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#94A3B8!important;border-radius:10px!important;font-weight:500!important;padding:.5rem 1.2rem!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#6366F1,#8B5CF6)!important;color:white!important;box-shadow:0 4px 12px rgba(99,102,241,.4)!important;}
.rc{background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.25);border-radius:12px;padding:.9rem 1.2rem;margin-bottom:.6rem;font-size:.9rem;line-height:1.5;}
.mb{background:var(--glass);border:1px solid rgba(99,102,241,.3);border-radius:16px;padding:1.5rem;text-align:center;transition:transform .2s,border-color .2s;backdrop-filter:blur(16px);}
.mb:hover{transform:translateY(-3px);border-color:var(--glow);box-shadow:0 8px 30px rgba(99,102,241,.15);}
.mbig{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;line-height:1;margin-bottom:.4rem;}
.mname{font-weight:600;color:var(--bright);margin-bottom:.3rem;font-size:.95rem;}
.mdesc{font-size:.78rem;color:var(--muted);line-height:1.4;}
.cmb{background:rgba(13,27,46,.8);border-radius:14px;padding:1rem .8rem;text-align:center;border:1px solid rgba(99,102,241,.25);margin-top:.5rem;}
.cmn{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;line-height:1;}
.cml{font-size:.75rem;font-weight:600;color:var(--bright);margin-top:.3rem;}
.cmd{font-size:.7rem;color:var(--muted);margin-top:.2rem;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:rgba(99,102,241,.4);border-radius:10px;}
::-webkit-scrollbar-thumb:hover{background:rgba(139,92,246,.6);}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({'figure.facecolor':'#0D1B2E','axes.facecolor':'#0D1B2E','axes.edgecolor':'#1E3A5F',
    'axes.labelcolor':'#64748B','xtick.color':'#64748B','ytick.color':'#64748B',
    'text.color':'#E2E8F0','grid.color':'#1E3A5F','grid.alpha':.4})

# ── DATA & MODEL ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/telco_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.drop('customerID', axis=1, inplace=True)
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
    return df

@st.cache_data
def encode_split(df):
    df2 = df.copy()
    le = LabelEncoder()
    for col in df2.select_dtypes(include='object').columns:
        df2[col] = le.fit_transform(df2[col])
    X, y = df2.drop('Churn',axis=1), df2['Churn']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=42)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte), ytr, yte, sc, X.columns.tolist()

@st.cache_resource
def get_model(Xtr,ytr):
    m = GradientBoostingClassifier(n_estimators=100,random_state=42)
    m.fit(Xtr,ytr); return m

df = load_data()
Xtr,Xte,ytr,yte,sc,feats = encode_split(df)
model  = get_model(Xtr,ytr)
yp     = model.predict(Xte)
acc    = accuracy_score(yte,yp)
prec   = precision_score(yte,yp)
rec    = recall_score(yte,yp)
f1s    = f1_score(yte,yp)
cr     = df['Churn'].mean()*100
total  = len(df); churned=int(df['Churn'].sum()); retained=total-churned

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:1.5rem 0 1.8rem;'>
        <div style='font-size:3rem;margin-bottom:.6rem;filter:drop-shadow(0 0 12px rgba(99,102,241,.6));'>🛡️</div>
        <div style='font-family:Syne,sans-serif;font-size:1.45rem;font-weight:800;
            background:linear-gradient(135deg,#FFFFFF 0%,#C4B5FD 50%,#67E8F9 100%);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            letter-spacing:-.02em;'>ChurnShield AI</div>
        <div style='font-size:.68rem;color:#475569;margin-top:.4rem;text-transform:uppercase;
            letter-spacing:.14em;font-weight:500;'>Churn Intelligence Platform</div>
    </div>
    <div style='height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,.6),rgba(139,92,246,.4),transparent);margin-bottom:1.5rem;'></div>
    <div style='font-size:.63rem;color:#6366F1;letter-spacing:.12em;text-transform:uppercase;
        font-weight:700;margin-bottom:.6rem;padding-left:.5rem;'>Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio("Nav",["📊  Dashboard","🔮  Predict Churn","📈  Model Insights"],label_visibility="collapsed")

    st.markdown(f"""
    <div style='height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,.6),rgba(139,92,246,.4),transparent);margin:1.5rem 0;'></div>
    <div style='background:linear-gradient(135deg,rgba(13,27,46,.9),rgba(9,18,36,.95));
                border:1px solid rgba(99,102,241,.35);border-radius:14px;padding:1.2rem;
                box-shadow:0 4px 20px rgba(99,102,241,.08);'>
        <div style='font-size:.63rem;letter-spacing:.14em;text-transform:uppercase;
            color:#6366F1;font-weight:700;margin-bottom:.9rem;'>System Status</div>
        <div style='display:flex;align-items:center;gap:.5rem;margin-bottom:.8rem;'>
            <div style='width:8px;height:8px;background:#10B981;border-radius:50%;
                box-shadow:0 0 8px #10B981,0 0 16px rgba(16,185,129,.4);'></div>
            <span style='font-size:.85rem;color:#E2E8F0;font-weight:600;'>Model Online</span>
        </div>
        <div style='font-size:.8rem;line-height:2.1;'>
            <div style='display:flex;justify-content:space-between;border-bottom:1px solid rgba(99,102,241,.1);padding-bottom:.3rem;margin-bottom:.3rem;'>
                <span style='color:#64748B;'>Algorithm</span>
                <span style='color:#C4B5FD;font-weight:600;'>Gradient Boosting</span>
            </div>
            <div style='display:flex;justify-content:space-between;border-bottom:1px solid rgba(99,102,241,.1);padding-bottom:.3rem;margin-bottom:.3rem;'>
                <span style='color:#64748B;'>Accuracy</span>
                <span style='color:#67E8F9;font-weight:700;'>{acc*100:.1f}%</span>
            </div>
            <div style='display:flex;justify-content:space-between;border-bottom:1px solid rgba(99,102,241,.1);padding-bottom:.3rem;margin-bottom:.3rem;'>
                <span style='color:#64748B;'>Records</span>
                <span style='color:#E2E8F0;'>{total:,}</span>
            </div>
            <div style='display:flex;justify-content:space-between;border-bottom:1px solid rgba(99,102,241,.1);padding-bottom:.3rem;margin-bottom:.3rem;'>
                <span style='color:#64748B;'>Features</span>
                <span style='color:#E2E8F0;'>{len(feats)} variables</span>
            </div>
            <div style='display:flex;justify-content:space-between;'>
                <span style='color:#64748B;'>Split</span>
                <span style='color:#E2E8F0;'>80% / 20%</span>
            </div>
        </div>
    </div>
    <div style='height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,.4),transparent);margin:1.5rem 0;'></div>
    <div style='font-size:.68rem;color:#334155;text-align:center;line-height:1.8;'>
        Microsoft Elevate Internship<br>
        <span style='color:#475569;'>Customer Churn Prediction</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Dashboard":
    st.markdown(f"""
    <div class='hero'>
        <div class='hero-badge'>🚀 Microsoft Elevate Internship — AI & ML Project</div>
        <div class='hero-title'>Customer Churn<br>Intelligence Platform</div>
        <p class='hero-sub'>Predict which customers are about to leave — and stop them before they do.
        Powered by Gradient Boosting ML trained on {total:,} telecom customer records.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='kpi kpi-i'><div class='kpi-lbl'>Total Customers</div><div class='kpi-val'>{total:,}</div><div class='kpi-sub'>Full telecom dataset</div></div>",unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi kpi-r'><div class='kpi-lbl'>Churn Rate</div><div class='kpi-val'>{cr:.1f}%</div><div class='kpi-sub'>{churned:,} customers lost</div></div>",unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi kpi-g'><div class='kpi-lbl'>Retained</div><div class='kpi-val'>{retained:,}</div><div class='kpi-sub'>Loyal customers</div></div>",unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi kpi-c'><div class='kpi-lbl'>Model Accuracy</div><div class='kpi-val'>{acc*100:.1f}%</div><div class='kpi-sub'>Gradient Boosting</div></div>",unsafe_allow_html=True)
    st.markdown("<div style='height:1.5rem'></div>",unsafe_allow_html=True)

    col1,col2 = st.columns([1,1.4])
    with col1:
        st.markdown("<div class='ccard'><div class='chead'>CHURN VS RETAINED</div>",unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(5,4))
        w,t,at = ax.pie([retained,churned],labels=['Retained','Churned'],colors=['#6366F1','#F43F5E'],
            autopct='%1.1f%%',startangle=90,pctdistance=.78,
            wedgeprops={'linewidth':3,'edgecolor':'#0D1B2E','width':.65},
            textprops={'color':'#94A3B8','fontsize':10})
        for a in at: a.set_color('#F8FAFC'); a.set_fontweight('bold'); a.set_fontsize(11)
        ax.text(0,0,f'{cr:.0f}%\nchurn',ha='center',va='center',fontsize=13,fontweight='bold',color='#F43F5E')
        fig.patch.set_facecolor('#0D1B2E'); ax.set_facecolor('#0D1B2E')
        st.pyplot(fig,use_container_width=True); plt.close()
        st.markdown("</div>",unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='ccard'><div class='chead'>CONTRACT TYPE VS CHURN RATE</div>",unsafe_allow_html=True)
        cc = df.groupby('Contract')['Churn'].mean()*100
        fig,ax = plt.subplots(figsize=(6.5,4))
        bars = ax.bar(cc.index,cc.values,color=['#F43F5E','#6366F1','#10B981'],width=.5,edgecolor='none',zorder=3)
        for b,v in zip(bars,cc.values):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+.8,f'{v:.1f}%',ha='center',fontsize=11,fontweight='bold',color='#E2E8F0')
        ax.set_ylabel('Churn Rate (%)',color='#64748B'); ax.set_ylim(0,max(cc.values)+12)
        ax.yaxis.grid(True,zorder=0,alpha=.4); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('#0D1B2E'); ax.set_facecolor('#0D1B2E'); fig.tight_layout()
        st.pyplot(fig,use_container_width=True); plt.close()
        st.markdown("</div>",unsafe_allow_html=True)

    col3,col4 = st.columns(2)
    with col3:
        st.markdown("<div class='ccard'><div class='chead'>MONTHLY CHARGES — CHURNED VS RETAINED</div>",unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(6,3.5))
        ax.hist(df[df['Churn']==0]['MonthlyCharges'],bins=35,color='#6366F1',alpha=.75,label='Retained',edgecolor='none')
        ax.hist(df[df['Churn']==1]['MonthlyCharges'],bins=35,color='#F43F5E',alpha=.75,label='Churned',edgecolor='none')
        ax.set_xlabel('Monthly Charges ($)',color='#64748B'); ax.set_ylabel('Customers',color='#64748B')
        ax.legend(facecolor='#0D1B2E',edgecolor='#1E3A5F',labelcolor='#E2E8F0')
        ax.yaxis.grid(True,alpha=.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('#0D1B2E'); ax.set_facecolor('#0D1B2E'); fig.tight_layout()
        st.pyplot(fig,use_container_width=True); plt.close()
        st.markdown("</div>",unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='ccard'><div class='chead'>TENURE DISTRIBUTION — CHURNED VS RETAINED</div>",unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(6,3.5))
        sns.kdeplot(df[df['Churn']==0]['tenure'],ax=ax,color='#6366F1',linewidth=2.5,label='Retained',fill=True,alpha=.2)
        sns.kdeplot(df[df['Churn']==1]['tenure'],ax=ax,color='#F43F5E',linewidth=2.5,label='Churned',fill=True,alpha=.2)
        ax.set_xlabel('Tenure (Months)',color='#64748B'); ax.set_ylabel('Density',color='#64748B')
        ax.legend(facecolor='#0D1B2E',edgecolor='#1E3A5F',labelcolor='#E2E8F0')
        ax.yaxis.grid(True,alpha=.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('#0D1B2E'); ax.set_facecolor('#0D1B2E'); fig.tight_layout()
        st.pyplot(fig,use_container_width=True); plt.close()
        st.markdown("</div>",unsafe_allow_html=True)

    st.markdown("<div class='div'></div><div class='sec'>💡 Key Business Insights</div>",unsafe_allow_html=True)
    st.markdown("""
    <div class='ig'>
        <div class='ip'><div class='in' style='color:#F43F5E;'>~42%</div>
            <div class='it'><b style='color:#E2E8F0;'>Month-to-month</b> customers churn at 42% — highest risk segment</div></div>
        <div class='ip'><div class='in' style='color:#22D3EE;'>10 mo</div>
            <div class='it'>Avg tenure of churned customers vs <b style='color:#E2E8F0;'>37 months</b> for retained</div></div>
        <div class='ip'><div class='in' style='color:#FBBF24;'>$74</div>
            <div class='it'>Avg monthly charge of churned vs <b style='color:#E2E8F0;'>$61</b> for retained</div></div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT CHURN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Predict Churn":
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div style='font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;
            background:linear-gradient(135deg,#A5B4FC,#C4B5FD,#67E8F9);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Predict Churn Risk</div>
        <div style='color:#64748B;font-size:.95rem;margin-top:.4rem;'>
            Fill in customer details to get an instant AI-powered churn probability</div>
    </div>""", unsafe_allow_html=True)

    with st.form("pf"):
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**👤 Customer Profile**")
            gender=st.selectbox("Gender",["Male","Female"])
            senior=st.selectbox("Senior Citizen",["No","Yes"])
            partner=st.selectbox("Has Partner",["Yes","No"])
            dependents=st.selectbox("Has Dependents",["No","Yes"])
            tenure=st.slider("Tenure (Months)",0,72,12)
            paperless=st.selectbox("Paperless Billing",["Yes","No"])
        with c2:
            st.markdown("**📱 Services**")
            phone=st.selectbox("Phone Service",["Yes","No"])
            mlines=st.selectbox("Multiple Lines",["No","Yes","No phone service"])
            internet=st.selectbox("Internet Service",["DSL","Fiber optic","No"])
            osec=st.selectbox("Online Security",["No","Yes","No internet service"])
            obkp=st.selectbox("Online Backup",["Yes","No","No internet service"])
            dprot=st.selectbox("Device Protection",["No","Yes","No internet service"])
        with c3:
            st.markdown("**💰 Billing & Contract**")
            tsup=st.selectbox("Tech Support",["No","Yes","No internet service"])
            stv=st.selectbox("Streaming TV",["No","Yes","No internet service"])
            smov=st.selectbox("Streaming Movies",["No","Yes","No internet service"])
            contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
            payment=st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            mchg=st.number_input("Monthly Charges ($)",0.0,200.0,65.0,step=0.5)
            tchg=st.number_input("Total Charges ($)",0.0,10000.0,float(mchg*tenure),step=1.0)
        sub=st.form_submit_button("🔮  Run Churn Prediction",use_container_width=True)

    if sub:
        raw = pd.DataFrame([{'gender':gender,'SeniorCitizen':1 if senior=="Yes" else 0,
            'Partner':partner,'Dependents':dependents,'tenure':tenure,
            'PhoneService':phone,'MultipleLines':mlines,'InternetService':internet,
            'OnlineSecurity':osec,'OnlineBackup':obkp,'DeviceProtection':dprot,
            'TechSupport':tsup,'StreamingTV':stv,'StreamingMovies':smov,
            'Contract':contract,'PaperlessBilling':paperless,'PaymentMethod':payment,
            'MonthlyCharges':mchg,'TotalCharges':tchg}])
        df_full = load_data().copy()
        le = LabelEncoder()
        for col in df_full.select_dtypes(include='object').columns:
            le.fit(df_full[col])
            if col in raw.columns: raw[col]=le.transform(raw[col])
        rs = sc.transform(raw[feats])
        pred  = model.predict(rs)[0]
        proba = model.predict_proba(rs)[0][1]

        if pred==1:
            st.markdown(f"<div class='res res-c'><div class='res-icon'>⚠️</div><div class='res-title' style='color:#F43F5E;'>HIGH CHURN RISK</div><div class='res-desc'>This customer has a <strong style='color:#F43F5E;font-size:1.2rem;'>{proba*100:.1f}%</strong> probability of churning.<br>Immediate retention action recommended.</div></div>",unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='res res-s'><div class='res-icon'>✅</div><div class='res-title' style='color:#10B981;'>LOW CHURN RISK</div><div class='res-desc'>This customer has only a <strong style='color:#10B981;font-size:1.2rem;'>{proba*100:.1f}%</strong> probability of churning.<br>Customer appears stable.</div></div>",unsafe_allow_html=True)

        st.markdown("**Churn Probability Meter**")
        st.progress(float(proba))
        st.caption(f"Risk Score: {proba*100:.1f}% — {'High Risk 🔴' if proba>.5 else 'Low Risk 🟢'}")

        st.markdown("<div class='div'></div>**📋 Retention Recommendations**",unsafe_allow_html=True)
        tips=[]
        if contract=="Month-to-month": tips.append("🔄 Offer a <b>discounted annual/2-year contract</b> — month-to-month is the #1 churn driver")
        if mchg>70: tips.append("💸 Consider a <b>loyalty pricing discount</b> — high charges increase churn risk")
        if tenure<12: tips.append("🎁 Enroll in a <b>new customer onboarding program</b> — early months are highest risk")
        if osec=="No" and internet!="No": tips.append("🔒 Offer a <b>free Online Security trial</b> to increase perceived value")
        if payment=="Electronic check": tips.append("💳 Encourage switch to <b>auto-pay</b> — reduces payment friction")
        if not tips: tips.append("✨ Strong retention indicators — maintain current service quality")
        for t in tips: st.markdown(f"<div class='rc'>{t}</div>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Model Insights":
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div style='font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;
            background:linear-gradient(135deg,#A5B4FC,#C4B5FD,#67E8F9);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Model Performance & Insights</div>
        <div style='color:#64748B;font-size:.95rem;margin-top:.4rem;'>Under the hood — metrics, feature importance, and where the model is right or wrong</div>
    </div>""", unsafe_allow_html=True)

    t1,t2,t3 = st.tabs(["🎯  Performance Metrics","🔥  Feature Importance","🧩  Confusion Matrix"])

    with t1:
        m1,m2,m3,m4=st.columns(4)
        for col,val,name,color,desc in [
            (m1,f"{acc*100:.1f}%","Accuracy","#6366F1","Overall % of correct predictions"),
            (m2,f"{prec*100:.1f}%","Precision","#8B5CF6","Of predicted churners, how many actually churned"),
            (m3,f"{rec*100:.1f}%","Recall","#22D3EE","Of all actual churners, how many we caught"),
            (m4,f"{f1s*100:.1f}%","F1 Score","#10B981","Balanced metric — harmonic mean of P & R"),
        ]:
            with col:
                st.markdown(f"<div class='mb'><div class='mbig' style='color:{color};'>{val}</div><div class='mname'>{name}</div><div class='mdesc'>{desc}</div></div>",unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>",unsafe_allow_html=True)
        st.markdown("<div class='ccard'><div class='chead'>ALGORITHM COMPARISON</div>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(9,3.5))
        bars=ax.barh(['Logistic\nRegression','Random\nForest','Gradient\nBoosting ★'],[78.54,79.03,79.53],
            color=['#1E3A5F','#2D4F7C','#8B5CF6'],height=.4,edgecolor='none')
        for b,v in zip(bars,[78.54,79.03,79.53]):
            ax.text(v+.05,b.get_y()+b.get_height()/2,f'{v:.2f}%',va='center',fontsize=11,fontweight='bold',color='#E2E8F0')
        ax.set_xlim(75,83); ax.set_xlabel('Accuracy (%)',color='#64748B')
        ax.xaxis.grid(True,alpha=.3); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('#0D1B2E'); ax.set_facecolor('#0D1B2E'); fig.tight_layout()
        st.pyplot(fig,use_container_width=True); plt.close()
        st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.2);border-radius:14px;padding:1.2rem 1.5rem;margin-top:.5rem;'>
            <div style='font-weight:700;color:#A5B4FC;margin-bottom:.6rem;'>🧠 Why does Gradient Boosting win?</div>
            <div style='color:#64748B;font-size:.88rem;line-height:1.8;'>
                <b style='color:#E2E8F0;'>Logistic Regression</b> — draws a straight decision boundary. Simple, fast, limited for complex patterns.<br>
                <b style='color:#E2E8F0;'>Random Forest</b> — 100 independent trees vote. Powerful but trees don't learn from each other.<br>
                <b style='color:#E2E8F0;'>Gradient Boosting</b> — trees built sequentially, each fixing the previous one's mistakes. Iterative error-correction = strongest predictions.
            </div>
        </div>""", unsafe_allow_html=True)

    with t2:
        st.markdown("<div class='ccard'><div class='chead'>TOP 15 FEATURES DRIVING CHURN</div>",unsafe_allow_html=True)
        imp=pd.Series(model.feature_importances_,index=feats)
        top=imp.sort_values(ascending=True).tail(15)
        fig,ax=plt.subplots(figsize=(9,6))
        n=len(top); bc=[plt.cm.cool(i/n) for i in range(n)]
        ax.barh(top.index,top.values,color=bc,height=.65,edgecolor='none')
        for i,(v,nm) in enumerate(zip(top.values,top.index)):
            ax.text(v+.0008,i,f'{v:.4f}',va='center',fontsize=8.5,color='#94A3B8')
        ax.set_xlabel('Importance Score',color='#64748B')
        ax.xaxis.grid(True,alpha=.3); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('#0D1B2E'); ax.set_facecolor('#0D1B2E'); fig.tight_layout()
        st.pyplot(fig,use_container_width=True); plt.close()
        st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(34,211,238,.05);border:1px solid rgba(34,211,238,.15);border-radius:14px;padding:1.2rem 1.5rem;margin-top:.5rem;'>
            <div style='font-weight:700;color:#67E8F9;margin-bottom:.8rem;'>📌 What drives churn?</div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:.8rem;font-size:.85rem;'>
                <div><b style='color:#E2E8F0;'>tenure</b> <span style='color:#64748B;'>— Longer = more loyal</span></div>
                <div><b style='color:#E2E8F0;'>TotalCharges</b> <span style='color:#64748B;'>— Higher lifetime spend = more invested</span></div>
                <div><b style='color:#E2E8F0;'>MonthlyCharges</b> <span style='color:#64748B;'>— High bills drive dissatisfaction</span></div>
                <div><b style='color:#E2E8F0;'>Contract</b> <span style='color:#64748B;'>— Month-to-month = easiest to leave</span></div>
                <div><b style='color:#E2E8F0;'>InternetService</b> <span style='color:#64748B;'>— Fiber optic users churn more</span></div>
                <div><b style='color:#E2E8F0;'>PaymentMethod</b> <span style='color:#64748B;'>— Electronic check = highest churn</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with t3:
        cc1,cc2=st.columns([1,1])
        with cc1:
            st.markdown("<div class='ccard'><div class='chead'>CONFUSION MATRIX</div>",unsafe_allow_html=True)
            cm=confusion_matrix(yte,yp)
            fig,ax=plt.subplots(figsize=(5.5,4.5))
            sns.heatmap(cm,annot=True,fmt='d',cmap='Purples',
                xticklabels=['No Churn','Churn'],yticklabels=['No Churn','Churn'],
                linewidths=3,linecolor='#050B18',annot_kws={'size':22,'weight':'bold','color':'white'},ax=ax)
            ax.set_xlabel('Predicted',labelpad=12,fontsize=11,color='#64748B')
            ax.set_ylabel('Actual',labelpad=12,fontsize=11,color='#64748B')
            fig.patch.set_facecolor('#0D1B2E'); ax.set_facecolor('#0D1B2E'); fig.tight_layout()
            st.pyplot(fig,use_container_width=True); plt.close()
            st.markdown("</div>",unsafe_allow_html=True)
        with cc2:
            tn,fp,fn,tp=confusion_matrix(yte,yp).ravel()
            for val,label,color,desc in [
                (tp,"True Positives","#10B981","Predicted churn → Actually churned ✓"),
                (tn,"True Negatives","#6366F1","Predicted stay  → Actually stayed ✓"),
                (fp,"False Positives","#FBBF24","Predicted churn → Actually stayed ✗"),
                (fn,"False Negatives","#F43F5E","Predicted stay  → Actually churned ✗"),
            ]:
                st.markdown(f"<div class='cmb'><div class='cmn' style='color:{color};'>{val}</div><div class='cml'>{label}</div><div class='cmd'>{desc}</div></div>",unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(244,63,94,.06);border:1px solid rgba(244,63,94,.2);border-radius:14px;padding:1.2rem 1.5rem;margin-top:1rem;'>
            <div style='font-weight:700;color:#F43F5E;margin-bottom:.5rem;'>⚠️ Why do False Negatives matter most?</div>
            <div style='color:#64748B;font-size:.88rem;line-height:1.7;'>
                A <b style='color:#E2E8F0;'>False Negative</b> = predicted STAY but customer CHURNED — we missed them and they left. Most costly mistake.<br><br>
                A <b style='color:#E2E8F0;'>False Positive</b> = predicted CHURN but they stayed — just a wasted discount. Manageable.<br><br>
                This is why <b style='color:#F43F5E;'>Recall</b> matters more than Precision for churn prediction.
            </div>
        </div>""", unsafe_allow_html=True)