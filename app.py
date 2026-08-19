from pathlib import Path
import re
import string

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "Model" / "logistic_regression_tfidf_tuned_model.joblib"
TRAIN_PATH = ROOT / "NLP" / "train.txt"

EMOTION_NAMES = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy",
}


def ensure_nltk_resources() -> None:
    resources = {
        "corpora/stopwords": "stopwords",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def clean_text(text: str, stop_words: set[str]) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = "".join(char for char in text if char.isalnum() or char.isspace())
    return " ".join(
        word for word in word_tokenize(text) if word.lower() not in stop_words
    )


@st.cache_resource(show_spinner="Loading model and rebuilding TF-IDF vocabulary...")
def load_pipeline():
    ensure_nltk_resources()
    data = __import__("pandas").read_csv(
        TRAIN_PATH, sep=";", header=None, names=["text", "emotion"]
    )
    stop_words = set(stopwords.words("english"))
    data["text"] = data["text"].apply(lambda value: clean_text(value, stop_words))
    label_map = {label: index for index, label in enumerate(data["emotion"].unique())}
    x_train, _, _, _ = train_test_split(
        data["text"],
        data["emotion"].map(label_map),
        test_size=0.20,
        random_state=42,
    )
    vectorizer = TfidfVectorizer()
    vectorizer.fit(x_train)
    model = joblib.load(MODEL_PATH)
    return model, vectorizer, stop_words


def render_confidence(label: str, value: float) -> None:
    st.markdown(
        f"""
        <div class="confidence-row">
            <div class="confidence-label"><span>{label}</span><span>{value:.1%}</span></div>
            <div class="confidence-track"><div class="confidence-fill" style="width: {value * 100:.1f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Emotion Classifier",
    page_icon=":material/psychology:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    :root { --ink: #e8edf2; --muted: #8f9aa7; --panel: #151a20; --line: #29313a; --accent: #55d6be; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0c1014; color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #10151a; border-right: 1px solid var(--line); }
    .brand { display: flex; align-items: center; gap: 11px; margin: 12px 0 58px; }
    .brand-mark { width: 34px; height: 34px; display: grid; place-items: center; border: 1px solid #3f776e; color: var(--accent); border-radius: 9px; }
    .brand-name { color: var(--ink); font-size: 15px; font-weight: 700; letter-spacing: .01em; }
    .eyebrow { color: var(--accent); text-transform: uppercase; font-size: 11px; letter-spacing: .16em; font-weight: 700; margin-bottom: 12px; }
    h1 { color: var(--ink) !important; font-size: 42px !important; line-height: 1.08 !important; letter-spacing: -.04em !important; margin-bottom: 12px !important; }
    .subhead { color: var(--muted); font-size: 15px; line-height: 1.6; max-width: 580px; margin-bottom: 34px; }
    .input-label { color: #c5cdd5; font-size: 13px; font-weight: 600; margin: 0 0 9px; }
    .result-panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 25px; margin-top: 28px; }
    .result-kicker { color: var(--muted); font-size: 11px; letter-spacing: .14em; text-transform: uppercase; }
    .result-label { color: var(--accent); font-size: 32px; font-weight: 700; margin: 7px 0 22px; }
    .confidence-row { margin: 12px 0; }
    .confidence-label { display: flex; justify-content: space-between; color: #c8d0d8; font-size: 13px; margin-bottom: 6px; }
    .confidence-track { height: 5px; background: #29313a; border-radius: 4px; overflow: hidden; }
    .confidence-fill { height: 100%; background: var(--accent); border-radius: 4px; }
    .status { border-top: 1px solid var(--line); padding-top: 18px; margin-top: 20px; color: var(--muted); font-size: 12px; line-height: 1.8; }
    .status strong { color: #d6dee5; font-weight: 600; }
    .stTextArea textarea { background: #12171c; border: 1px solid var(--line); color: var(--ink); border-radius: 9px; font-size: 15px; }
    .stTextArea textarea:focus { border-color: #3f776e; box-shadow: 0 0 0 1px #3f776e; }
    .stButton button { background: var(--accent); color: #09201b; border: 0; border-radius: 8px; font-weight: 700; min-height: 43px; }
    .stButton button:hover { background: #72e5d0; color: #09201b; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark"><span class="material-symbols-rounded">psychology</span></div><div class="brand-name">Emotion Classifier</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Model status")
    st.markdown(
        '<div class="status"><strong>Ready for inference</strong><br>TF-IDF + Logistic Regression<br>6 emotion classes<br>Model accuracy: 88.2%</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="eyebrow">Text intelligence</div>', unsafe_allow_html=True)
st.title("What emotion is behind the text?")
st.markdown(
    '<div class="subhead">Write a sentence and the trained model will estimate the dominant emotion across six categories.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="input-label">Your text</div>', unsafe_allow_html=True)
text = st.text_area(
    "Text to classify",
    placeholder="I feel calm and ready for the day...",
    height=150,
    label_visibility="collapsed",
)

if st.button("Classify emotion", icon=":material/arrow_forward:", use_container_width=True):
    if not text.strip():
        st.warning("Enter some text before classifying.", icon=":material/edit:")
    else:
        with st.spinner("Analyzing text..."):
            model, vectorizer, stop_words = load_pipeline()
            cleaned = clean_text(text, stop_words)
            features = vectorizer.transform([cleaned])
            prediction = int(model.predict(features)[0])
            probabilities = model.predict_proba(features)[0]

        st.markdown(
            f'<div class="result-panel"><div class="result-kicker">Predicted emotion</div><div class="result-label">{EMOTION_NAMES[prediction].title()}</div>',
            unsafe_allow_html=True,
        )
        for index in probabilities.argsort()[::-1]:
            render_confidence(EMOTION_NAMES[int(index)].title(), float(probabilities[index]))
        st.markdown("</div>", unsafe_allow_html=True)
