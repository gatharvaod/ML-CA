# app.py
import os, re, requests
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import plotly.express as px

# ---------- Page config ----------
st.set_page_config(page_title="News URL → Category", page_icon="🗞️", layout="wide")

# ---------- Visual theme ----------
st.markdown("""
<style>
:root{
  --bg:#0b1220; --bg2:#0e172a; --panel:rgba(255,255,255,0.06);
  --text:#e6f1ff; --muted:#94a3b8; --brand:#22d3ee; --accent:#22c55e; --warn:#f59e0b;
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1400px 600px at 10% -10%, rgba(34,211,238,.12), transparent 60%),
    radial-gradient(1200px 500px at 90% 0%, rgba(34,197,94,.10), transparent 60%),
    linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
  color:var(--text);
}
.block-container{max-width:1200px;}
.card{
  background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.04));
  border: 1px solid rgba(255,255,255,.12);
  backdrop-filter: blur(8px);
  border-radius:18px; padding:18px 20px; box-shadow: 0 10px 30px rgba(0,0,0,.35);
}
h1{font-size:44px; letter-spacing:.5px;}
.badge{
  display:inline-flex; gap:8px; align-items:center;
  padding:8px 14px; border-radius:999px;
  background:rgba(34,211,238,.15); border:1px solid rgba(34,211,238,.4); color:#dff9ff; font-weight:700;
}
.big{
  font-size:36px; font-weight:800; letter-spacing:.5px;
  padding:6px 14px; border-radius:12px;
  color:#031b11; background:linear-gradient(90deg,#22c55e,#86efac);
  display:inline-block;
}
.smallmuted{color:var(--muted); font-size:13px}
hr{border:0; height:1px; margin:12px 0 4px;
   background:linear-gradient(90deg,transparent, rgba(255,255,255,.18), transparent);}
.stButton>button{
  background:linear-gradient(90deg,#22d3ee,#22c55e);
  color:#001; border:none; font-weight:700; border-radius:12px; padding:10px 16px;
}
.stTextInput>div>div>input{border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("### 🗞️ News Article Classification")
st.caption("Paste a news link → extract text → predict category. TF-IDF + Logistic Regression.")

# ---------- NLTK bootstrap ----------
for corp, path in [("stopwords","corpora/stopwords"),
                   ("wordnet","corpora/wordnet"),
                   ("punkt","tokenizers/punkt")]:
    try: nltk.data.find(path)
    except LookupError: nltk.download(corp)

STOP = set(stopwords.words("english"))
LEMM = WordNetLemmatizer()

def preprocess(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    toks = [LEMM.lemmatize(t) for t in text.split() if t not in STOP]
    return " ".join(toks)

# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_raw_csv() -> pd.DataFrame:
    local = "data_news.csv"
    if os.path.exists(local): return pd.read_csv(local)
    url = "https://raw.githubusercontent.com/kavya-chakradhari/News-Article-Classification-NLP/main/data_news.csv"
    return pd.read_csv(url)

def guess_columns(df: pd.DataFrame):
    cols = list(df.columns)
    text_guess = next((c for c in [
        "combined_text","text","content","article","body","headline_text",
        "Title_Text","TitleText","news","News","Description","desc",
        "short_description","headline","Title","title"
    ] if c in cols), None)
    label_guess = next((c for c in [
        "label","category","target","Class","class","Category","topic","Topic","Section"
    ] if c in cols), None)
    if text_guess is None:
        title = next((c for c in ["title","Title","headline","Headline"] if c in cols), None)
        body  = next((c for c in ["text","Text","content","Content","article","Article","body","Body","news","News"] if c in cols), None)
        if title and body:
            df["combined_text"] = (df[title].fillna("").astype(str) + " " + df[body].fillna("").astype(str)).str.strip()
            text_guess = "combined_text"
    if text_guess is None:
        obj_cols = [c for c in cols if df[c].dtype == "object"]
        text_guess = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean()) if obj_cols else cols[0]
    if label_guess is None:
        cands = [(df[c].nunique(dropna=True), c) for c in cols if c != text_guess]
        label_guess = sorted([t for t in cands if 2 <= t[0] <= 100], default=cands)[0][1]
    return text_guess, label_guess

def train_model(df_xy: pd.DataFrame):
    X = df_xy["text"].astype(str).apply(preprocess)
    y = df_xy["label"].astype(str)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)),
        ("clf", LogisticRegression(max_iter=200, solver="liblinear", multi_class="ovr")),
    ])
    pipe.fit(Xtr, ytr)
    try: print(classification_report(yva, pipe.predict(Xva)))
    except Exception: pass
    joblib.dump(pipe, "model.pkl")
    return pipe

def load_or_train(df_xy: pd.DataFrame):
    if os.path.exists("model.pkl"):
        try: return joblib.load("model.pkl")
        except Exception: pass
    return train_model(df_xy)

def fetch_article_text(url: str) -> str:
    r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"}); r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for bad in soup(["script","style","noscript","header","footer","aside","nav"]): bad.extract()
    article = soup.find("article")
    ps = article.find_all("p") if article else soup.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in ps)
    if len(text.split()) < 50:
        title = soup.find("title")
        text = ((title.get_text(strip=True) if title else "") + " " + text).strip()
    return text

# ---------- Sidebar mapping ----------
df_raw = load_raw_csv()
cols = list(df_raw.columns)
text_guess, label_guess = guess_columns(df_raw)
with st.sidebar:
    st.markdown("### Dataset column mapping")
    st.write("Detected columns:", cols)
    text_col = st.selectbox("Text column", options=cols,
                            index=cols.index("short_description") if "short_description" in cols else cols.index(text_guess))
    label_col = st.selectbox("Label/Category column", options=cols,
                             index=cols.index("category") if "category" in cols else cols.index(label_guess))
df_xy = df_raw[[text_col, label_col]].dropna().copy()
df_xy.columns = ["text", "label"]

with st.expander("Preview data_news.csv", expanded=False):
    st.dataframe(df_xy.head(), use_container_width=True)

# ---------- Train / Load ----------
with st.spinner("Training or loading model…"):
    model = load_or_train(df_xy)
    classes = list(getattr(model.named_steps["clf"], "classes_", []))
    
# ---------- Input card ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
url = st.text_input("Paste a news article URL", placeholder="https://…")
go = st.button("Classify", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Predict ----------
if go and url:
    try:
        with st.spinner("Fetching article…"):
            article_text = fetch_article_text(url)

        with st.expander("Preview extracted text", expanded=False):
            st.write(article_text[:2500] + ("…" if len(article_text) > 2500 else ""))

        with st.spinner("Predicting…"):
            Xq = [preprocess(article_text)]
            pred = model.predict(Xq)[0]
            proba = model.predict_proba(Xq)[0] if hasattr(model.named_steps["clf"], "predict_proba") else None

        # Result header
        st.markdown('<div class="card" style="margin-top:20px;">', unsafe_allow_html=True)
        st.markdown("**Prediction**")
        st.markdown(f'<span class="big">{pred}</span>', unsafe_allow_html=True)

        if proba is not None and len(classes) == len(proba):
            top_p = float(max(proba))
            st.progress(min(1.0, max(0.0, top_p)))

            dfp = pd.DataFrame({"Category": classes, "Probability": proba}).sort_values("Probability", ascending=True)
            fig = px.bar(dfp, x="Probability", y="Category", orientation="h",
                         text=dfp["Probability"].map(lambda x:f"{x:.2f}"), range_x=[0, 1])
            fig.update_traces(textposition="outside")
            fig.update_layout(height=520, margin=dict(l=10,r=30,t=20,b=10),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(gridcolor="rgba(255,255,255,.15)"),
                              yaxis=dict(gridcolor="rgba(255,255,255,0)"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dfp.sort_values("Probability", ascending=False).reset_index(drop=True),
                         use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(str(e))
