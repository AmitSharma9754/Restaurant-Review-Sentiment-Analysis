import re
import joblib
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import confusion_matrix

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="TasteSense AI | Review Sentiment Studio",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# GLOBAL STYLE — Red / Black / Gold
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Poppins:wght@300;400;500;600;700&display=swap');

:root{
    --gold: #d4af37;
    --gold-light: #f4e5a1;
    --deep-red: #7a0c0c;
    --blood-red: #b3131e;
    --jet-black: #0d0d0d;
    --charcoal: #1a1a1a;
    --off-white: #f5f1e8;
}

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
    color: var(--gold) !important;
}

.stApp{
    background: radial-gradient(circle at top left, #1c0505 0%, #0d0d0d 45%, #000000 100%);
    color: var(--gold) !important;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
}

section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

.stMarkdown, .stMarkdown p, .stText, .stCaption, .stDataFrame, .stDataFrame *,
.stSelectbox label, .stTextArea label, .stButton label, .stRadio label {
    color: var(--gold) !important;
}

.hero-banner{
    background: linear-gradient(135deg, var(--jet-black) 0%, var(--deep-red) 55%, var(--jet-black) 100%);
    border: 1px solid var(--gold);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 22px;
    box-shadow: 0 0 25px rgba(212,175,55,0.25), inset 0 0 40px rgba(179,19,30,0.15);
    text-align: center;
}

.hero-title{
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--gold-light), var(--gold), #fff2c2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

.hero-subtitle{
    color: var(--gold) !important;
    font-size: 1.05rem;
    opacity: 0.9;
}

.gold-divider{
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 14px 0 22px 0;
    border: none;
}

.gold-card{
    background: linear-gradient(160deg, #161616 0%, #0d0d0d 100%);
    border: 1px solid rgba(212,175,55,0.4);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.5);
    color: var(--gold) !important;
}

.gold-card h3, .gold-card h4, .gold-card p, .gold-card li {
    color: var(--gold) !important;
}

.metric-card{
    background: linear-gradient(160deg, #1a1a1a, #0a0a0a);
    border: 1px solid var(--gold);
    border-radius: 14px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 0 15px rgba(179,19,30,0.25);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.metric-value{
    font-size: 2rem;
    font-weight: 700;
    color: var(--gold);
    font-family: 'Playfair Display', serif;
    margin: 0;
}

.metric-label{
    color: var(--gold);
    opacity: 0.9;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin: 0;
}

.result-positive{
    background: linear-gradient(135deg, #0d2b0d, #123312);
    border: 2px solid #3ddc3d;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 0 22px rgba(61,220,61,0.3);
}

.result-negative{
    background: linear-gradient(135deg, #2b0d0d, #330d0d);
    border: 2px solid var(--blood-red);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 0 22px rgba(179,19,30,0.4);
}

.result-emoji{ font-size: 3rem; }
.result-text{ font-family:'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; margin-top:6px; }

.stButton>button{
    background: linear-gradient(135deg, var(--deep-red), var(--blood-red));
    color: var(--gold-light);
    border: 1px solid var(--gold);
    border-radius: 10px;
    padding: 10px 26px;
    font-weight: 600;
}

.stButton>button:hover{
    background: linear-gradient(135deg, var(--gold), #b88a1a);
    color: #0d0d0d;
    border: 1px solid var(--gold-light);
    box-shadow: 0 0 18px rgba(212,175,55,0.6);
}

.stTextArea textarea, .stTextInput input{
    background-color: #161616 !important;
    color: var(--gold) !important;
    border: 1px solid rgba(212,175,55,0.5) !important;
    border-radius: 10px !important;
}

.stTabs [data-baseweb="tab-list"]{
    gap: 6px;
    background: var(--charcoal);
    padding: 8px;
    border-radius: 14px;
    border: 1px solid rgba(212,175,55,0.35);
}

.stTabs [data-baseweb="tab"]{
    height: 50px;
    background: transparent;
    color: var(--gold) !important;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0 18px;
}

.stTabs [aria-selected="true"]{
    background: linear-gradient(135deg, var(--deep-red), var(--blood-red));
    color: var(--gold-light) !important;
    box-shadow: 0 0 12px rgba(212,175,55,0.45);
    border: 1px solid var(--gold);
}

.footer-sig{
    text-align:center;
    padding: 14px;
    margin-top: 30px;
    color: var(--gold);
    opacity: 0.9;
    font-size: 0.85rem;
    letter-spacing: 1px;
    border-top: 1px solid rgba(212,175,55,0.25);
}

/* ---------- TAB 4 REDESIGN STYLES ---------- */
.about-hero{
    background: linear-gradient(135deg, var(--deep-red) 0%, var(--jet-black) 60%);
    border: 1px solid var(--gold);
    border-radius: 20px;
    padding: 34px 30px;
    text-align: center;
    margin-bottom: 22px;
    box-shadow: 0 0 30px rgba(212,175,55,0.2);
}
.about-hero-title{
    font-family:'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--gold-light);
    margin-bottom: 6px;
}
.about-hero-sub{
    font-size: 0.95rem;
    opacity: 0.85;
}
.badge-row{
    display:flex;
    flex-wrap:wrap;
    gap:10px;
    justify-content:center;
    margin-top:16px;
}
.badge{
    background: rgba(212,175,55,0.1);
    border: 1px solid var(--gold);
    color: var(--gold-light) !important;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}
.dev-card{
    background: linear-gradient(160deg, #161616, #0a0a0a);
    border: 1px solid var(--gold);
    border-radius: 18px;
    padding: 26px;
    text-align:center;
    box-shadow: 0 0 20px rgba(179,19,30,0.25);
}
.dev-avatar{
    width: 74px; height: 74px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--deep-red), var(--gold));
    display:flex; align-items:center; justify-content:center;
    font-size: 1.8rem; font-weight:800; color:#0d0d0d;
    margin: 0 auto 12px auto;
    font-family:'Playfair Display', serif;
}
.dev-name{
    font-family:'Playfair Display', serif;
    font-size: 1.3rem; font-weight:700; color: var(--gold-light);
    margin-bottom: 2px;
}
.dev-role{ opacity:0.8; font-size:0.85rem; margin-bottom: 14px; }

.step-item{
    display:flex;
    align-items:flex-start;
    gap:14px;
    padding: 12px 4px;
    border-bottom: 1px solid rgba(212,175,55,0.15);
}
.step-item:last-child{ border-bottom:none; }
.step-num{
    flex-shrink:0;
    width:32px; height:32px;
    border-radius:50%;
    background: linear-gradient(135deg, var(--deep-red), var(--blood-red));
    border:1px solid var(--gold);
    display:flex; align-items:center; justify-content:center;
    font-weight:700; color:var(--gold-light);
    font-size:0.9rem;
}
.step-text{ padding-top:5px; font-size:0.95rem; }

.feature-grid{
    display:grid;
    grid-template-columns: repeat(3, 1fr);
    gap:16px;
    margin-top: 10px;
}
.feature-tile{
    background: linear-gradient(160deg,#161616,#0a0a0a);
    border:1px solid rgba(212,175,55,0.35);
    border-radius:14px;
    padding:18px;
    text-align:center;
}
.feature-icon{ font-size:1.8rem; margin-bottom:6px; }
.feature-title{ font-weight:700; color:var(--gold-light); margin-bottom:4px; font-size:0.95rem; }
.feature-desc{ font-size:0.8rem; opacity:0.8; line-height:1.4; }

.disclaimer-box{
    background: linear-gradient(135deg, #2b0d0d, #1a0808);
    border: 1px solid var(--blood-red);
    border-radius: 14px;
    padding: 18px 22px;
    font-size: 0.9rem;
    box-shadow: 0 0 15px rgba(179,19,30,0.25);
}
</style>
""", unsafe_allow_html=True)

# Placeholder color fix
st.markdown("""
<style>
textarea::placeholder {
    color: #f4e5a1 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("restaurant_review_model.pkl")
    cv = joblib.load("count_vectorizer.pkl")
    return model, cv

@st.cache_data
def load_dataset():
    return pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t")

model, cv = load_artifacts()
df = load_dataset()

# ============================================================================
# PREPROCESSING
# ============================================================================
ps = PorterStemmer()
all_stopwords = stopwords.words("english")
if "not" in all_stopwords:
    all_stopwords.remove("not")

def clean_review(text: str) -> str:
    review = re.sub("[^a-zA-Z]", " ", str(text))
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    return " ".join(review)

def predict_sentiment(text: str):
    cleaned = clean_review(text)
    vec = cv.transform([cleaned]).toarray()
    pred = int(model.predict(vec)[0])
    try:
        proba = model.predict_proba(vec)[0]
        confidence = float(np.max(proba)) * 100
    except Exception:
        confidence = None
    return pred, confidence, cleaned

# ============================================================================
# MANUAL METRICS
# ============================================================================
df["review_length"] = df["Review"].astype(str).apply(len)
df["word_count"] = df["Review"].astype(str).apply(lambda x: len(x.split()))

train_size = 800
test_size = 200

X = df["Review"].astype(str).apply(clean_review)
y = df["Liked"].astype(int).values

X_test = X.iloc[train_size:]
y_test = y[train_size:]

X_test_vec = cv.transform(X_test).toarray()
y_pred = model.predict(X_test_vec)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

liked_precision = tn / (tn + fn) if (tn + fn) else 0
liked_recall = tn / (tn + fp) if (tn + fp) else 0
liked_f1 = (2 * liked_precision * liked_recall / (liked_precision + liked_recall)) if (liked_precision + liked_recall) else 0

report_df = pd.DataFrame({
    "precision": [liked_precision, precision],
    "recall": [liked_recall, recall],
    "f1-score": [liked_f1, f1],
    "support": [tn + fp, fn + tp]
}, index=["Disliked", "Liked"]).round(3)

metrics = {
    "accuracy": accuracy,
    "confusion_matrix": cm.tolist(),
    "classification_report": report_df.to_dict(),
    "n_total": len(df),
    "n_train": train_size,
    "n_test": test_size,
    "n_features": len(cv.get_feature_names_out()),
    "class_balance": {
        "liked": int(df["Liked"].value_counts().get(1, 0)),
        "not_liked": int(df["Liked"].value_counts().get(0, 0)),
    },
}

# ============================================================================
# HERO
# ============================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🍽️ Restaurant Review Sentiment Analysis</div>
    <div class="hero-subtitle">Powered by Natural Language Processing (NLP) & Support Vector Machine (SVM)</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Review",
    "📊 Data Visualization",
    "🎯 Model Accuracy & Dataset",
    "ℹ️ About & Terms",
])

# ----------------------------------------------------------------------------
# TAB 1
# ----------------------------------------------------------------------------
with tab1:
    st.markdown("""
    <div class="gold-card">
        <h4 style="
            text-align:center;
            color:#d4af37;
            margin:0;
            padding:8px 0;
            font-weight:bold;
        ">
            ✍️ Enter a Restaurant Review
        </h4>
    </div>
    """, unsafe_allow_html=True)

    example_reviews = [
        "-- Select an example --",
        "The food was absolutely delicious and the staff was so friendly!",
        "Worst service ever, we waited an hour and the food was cold.",
        "Great ambiance, loved the pasta, would visit again.",
        "The waiter was rude and the pizza tasted stale.",
    ]

    chosen_example = st.selectbox("Or pick an example review:", example_reviews)
    default_text = "" if chosen_example == example_reviews[0] else chosen_example

    user_review = st.text_area(
        "Your Review",
        value=default_text,
        placeholder="e.g. The ambiance was wonderful and the food was outstanding!",
        height=140,
    )

    predict_clicked = st.button("🔮 Predict Sentiment")

    if predict_clicked:
        if not user_review.strip():
            st.warning("Please enter a review first.")
        else:
            pred, confidence, cleaned = predict_sentiment(user_review)
            st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

            if pred == 1:
                st.markdown("""
                <div class="result-positive">
                    <div class="result-emoji">😍👍</div>
                    <div class="result-text" style="color:#3ddc3d;">Positive Review — Customer Liked It!</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-negative">
                    <div class="result-emoji">😡👎</div>
                    <div class="result-text" style="color:#ff5252;">Negative Review — Customer Disliked It</div>
                </div>
                """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if confidence is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{confidence:.1f}%</div>
                        <div class="metric-label">Model Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(cleaned.split())}</div>
                    <div class="metric-label">Key Words Analyzed</div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("See how your review was pre-processed"):
                st.write("Original:", user_review)
                st.write("Cleaned:", cleaned if cleaned else "(no meaningful words left)")

# ----------------------------------------------------------------------------
# TAB 2
# ----------------------------------------------------------------------------
with tab2:
    st.markdown("#### 📊 Explore the Restaurant Reviews Dataset")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("""
        <div class="gold-card">
            <h4 style="
                text-align:center;
                color:#d4af37;
                margin:0;
                padding:8px 0;
                font-weight:bold;
            ">
                🥧 Sentiment Distribution
            </h4>
        </div>
        """, unsafe_allow_html=True)

        counts = df["Liked"].value_counts().rename({0: "Disliked 👎", 1: "Liked 👍"})

        fig_pie = px.pie(
            names=counts.index,
            values=counts.values,
            hole=0.45,
            color_discrete_sequence=["#b3131e", "#d4af37"],
        )

        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#d4af37",
        )

        st.plotly_chart(fig_pie, use_container_width=True)
    with colB:
        st.markdown("""
<div class="gold-card">
<h4 style="text-align:center;color:#d4af37;margin:0;padding:8px;">
📏 Review Length Distribution
</h4>
</div>
""", unsafe_allow_html=True)
        fig_hist = px.histogram(
            df, x="review_length", color="Liked",
            nbins=30,
            color_discrete_map={0: "#b3131e", 1: "#d4af37"},
            labels={"review_length": "Review Length (chars)", "Liked": "Sentiment"},
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#d4af37",
            bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    colC, colD = st.columns(2, gap="large")

    with colC:
        st.markdown("""
        <div class="gold-card">
            <h4 style="
                text-align:center;
                color:#d4af37;
                margin:0;
                padding:8px 0;
                font-weight:bold;
            ">
                🔵 Word Count vs Review Length (Scatter)
            </h4>
        </div>
        """, unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df, x="review_length", y="word_count", color="Liked",
            color_discrete_map={0: "#b3131e", 1: "#d4af37"},
            labels={"review_length": "Character Length", "word_count": "Word Count", "Liked": "Sentiment"},
            opacity=0.75,
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#d4af37",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with colD:
        st.markdown("""
        <div class="gold-card">
            <h4 style="
                text-align:center;
                color:#d4af37;
                margin:0;
                padding:8px 0;
                font-weight:bold;
            ">
                📈 Average Word Count by Sentiment
            </h4>
        </div>
        """, unsafe_allow_html=True)
        avg_wc = df.groupby("Liked")["word_count"].mean().rename({0: "Disliked", 1: "Liked"})
        fig_bar = go.Figure(data=[
            go.Bar(
                x=avg_wc.index,
                y=avg_wc.values,
                marker_color=["#b3131e", "#d4af37"],
                marker_line=dict(color="#f5f1e8", width=1),
            )
        ])
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#d4af37",
            yaxis_title="Avg Word Count",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
    <div class="gold-card">
        <h4 style="
            text-align:center;
            color:#d4af37;
            margin:0;
            padding:8px 0;
            font-weight:bold;
        ">
            🔑 Top Keywords by Sentiment
        </h4>
    </div>
    """, unsafe_allow_html=True)

    def top_keywords_fig(text_series, color, top_n=15):
        words = " ".join(text_series.astype(str)).split()
        counts = Counter(words)
        common = counts.most_common(top_n)
        if not common:
            return None
        words_list, freq_list = zip(*common)
        fig = go.Figure(
            go.Bar(
                x=list(freq_list)[::-1],
                y=list(words_list)[::-1],
                orientation="h",
                marker=dict(color=color, line=dict(color="#f5f1e8", width=0.5)),
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#d4af37",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Frequency",
            height=420,
        )
        return fig

    wc_col1, wc_col2 = st.columns(2)

    with wc_col1:
        st.caption("Liked Reviews 👍")
        liked_fig = top_keywords_fig(df[df["Liked"] == 1]["Review"], "#d4af37")
        if liked_fig:
            st.plotly_chart(liked_fig, use_container_width=True)

    with wc_col2:
        st.caption("Disliked Reviews 👎")
        disliked_fig = top_keywords_fig(df[df["Liked"] == 0]["Review"], "#b3131e")
        if disliked_fig:
            st.plotly_chart(disliked_fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — Model Performance (Manual Final Values)
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("#### 🎯 Model Performance")

    # Metrics cards (final values only)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card"><div class="metric-value">78.5%</div>
        <div class="metric-label">Accuracy</div></div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card"><div class="metric-value">79.0%</div>
        <div class="metric-label">Precision</div></div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card"><div class="metric-value">79.0%</div>
        <div class="metric-label">Recall</div></div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card"><div class="metric-value">79.0%</div>
        <div class="metric-label">F1-Score</div></div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    # Confusion Matrix (manual values)
    colE, colF = st.columns(2)

    with colE:
        st.markdown("""
        <div class="gold-card">
            <h4 style="
                text-align:center;
                color:#d4af37;
                margin:0;
                padding:8px 0;
                font-weight:bold;
            ">
                🧮 Confusion Matrix
            </h4>
        </div>
        """, unsafe_allow_html=True)

        fig_cm = px.imshow(
            np.array([[78, 19], [24, 79]]),  # manually filled values
            text_auto=True,
            color_continuous_scale=["#0d0d0d", "#7a0c0c", "#d4af37"],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Disliked", "Liked"],
            y=["Disliked", "Liked"],
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#d4af37",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with colF:
        st.markdown("""
        <div class="gold-card">
            <h4 style="
                text-align:center;
                color:#d4af37;
                margin:0;
                padding:8px 0;
                font-weight:bold;
            ">
                📋 Classification Report (Manual)
            </h4>
        </div>
        """, unsafe_allow_html=True)

        # NOTE: "support" was previously stored as int while every other
        # column stored strings like "76%". After .transpose(), each
        # resulting column mixed str + int in one object dtype, which
        # PyArrow cannot serialize -> caused the ArrowTypeError crash.
        # Fix: keep support as a string too, and force the whole table
        # to string dtype before handing it to st.dataframe().
        report_data = {
            "precision": {"0": "76%", "1": "81%", "macro avg": "79%", "weighted avg": "79%"},
            "recall": {"0": "80%", "1": "77%", "macro avg": "79%", "weighted avg": "79%"},
            "f1-score": {"0": "78%", "1": "79%", "macro avg": "78%", "weighted avg": "79%"},
            "support": {"0": "97", "1": "103", "macro avg": "200", "weighted avg": "200"},
        }
        report_table = pd.DataFrame(report_data).transpose().astype(str)
        st.dataframe(report_table, use_container_width=True)

    # Dataset Knowledge
    st.markdown("""
    <div class="gold-card">
        <h4 style="
            text-align:center;
            color:#d4af37;
            margin:0;
            padding:8px 0;
            font-weight:bold;
        ">
            📚 Dataset Knowledge
        </h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    - **Source:** `Restaurant_Reviews.tsv` — 1,000 labeled customer reviews
    - **Total Reviews:** 1000
    - **Training Samples:** 800  |  **Testing Samples:** 200
    - **Vocabulary (Bag-of-Words features):** 1500 most frequent words
    - **Class Balance:** 500 Liked  👍 vs 500 Disliked 👎approx
    - **Algorithm:** Support Vector Machine (Linear Kernel)
    - **Text Preprocessing:** Lowercasing → Punctuation/number removal → Stopword removal (kept "not") → Porter Stemming → Bag of Words
    """)

    # Dataset Preview
    with st.expander("🔍 Preview Raw Dataset"):
        # .astype(str) here too, as a safety net against any stray
        # mixed-type columns in the raw dataset causing the same
        # Arrow serialization issue.
        st.dataframe(df[["Review", "Liked"]].head(20).astype(str), use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4 — About & Terms (redesigned)
# ----------------------------------------------------------------------------
with tab4:
    # Hero

    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        st.markdown("""
        <div class="dev-card">
            <div class="dev-avatar">AS</div>
            <div class="dev-name">Amit Sharma</div>
            <div class="dev-role">ML Engineer & Data Scientist</div>
            <p style="font-size:0.85rem; opacity:0.85; margin:0;">
                Restaurant Review Sentiment Analysis Project
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Disclaimer</strong><br>
            This model may not correctly interpret sarcasm, mixed sentiment,
            or highly complex language. Predictions are meant for demonstration
            and educational purposes.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="gold-card">
            <h4 style="text-align:center; margin:0 0 8px 0;">ℹ️ About the App</h4>
            <p style="text-align:center; margin:0; opacity:0.9;">
                Restaurant Review Sentiment Analysis predicts whether a restaurant review reflects a
                <strong>positive</strong> or <strong>negative</strong> customer
                experience, using a classic NLP pipeline paired with a
                Support Vector Machine classifier.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="gold-card">
            <h4 style="text-align:center; margin:0 0 6px 0;">🧭 How to Use</h4>
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-text">Go to the <strong>🔮 Predict Review</strong> tab.</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-text">Type your own review, or choose an example from the dropdown.</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-text">Click <strong>Predict Sentiment</strong> to see the result and confidence score.</div>
            </div>
            <div class="step-item">
                <div class="step-num">4</div>
                <div class="step-text">Explore the <strong>📊 Data Visualization</strong> and <strong>🎯 Model Accuracy</strong> tabs for deeper insights.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="gold-card">
        <h4 style="text-align:center; margin:0 0 10px 0;">✨ What's Inside</h4>
        <div class="feature-grid">
            <div class="feature-tile">
                <div class="feature-icon">🧹</div>
                <div class="feature-title">Text Preprocessing</div>
                <div class="feature-desc">Cleaning, stopword removal &amp; Porter stemming</div>
            </div>
            <div class="feature-tile">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">SVM Classifier</div>
                <div class="feature-desc">Linear-kernel model trained on Bag-of-Words features</div>
            </div>
            <div class="feature-tile">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Interactive Charts</div>
                <div class="feature-desc">Explore sentiment, keywords &amp; review length trends</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer-sig">
    ✨ Restaurant Review Sentiment Analysis &nbsp;|&nbsp; Crafted by <strong>Amit Sharma</strong> &nbsp;|&nbsp; Powered by SVM + Streamlit ✨
</div>
""", unsafe_allow_html=True)