import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from nlp_utils import prediction_frame, clean_text, STAR_LABELS


APP_TITLE = "OpinionLens AI"
APP_SUBTITLE = "Extreme vs Moderate Opinion Identifier for Customer & Market Insights"
MODEL_PATH = Path("models/opinion_intensity_pipeline.joblib")
METRICS_PATH = Path("models/metrics.json")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #172033;
    --muted: #64748B;
    --brand: #6864F7;
    --teal: #14B8A6;
    --rose: #E05269;
    --amber: #F6A85A;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background:
      radial-gradient(circle at 6% 5%, rgba(104, 100, 247, 0.20), transparent 30%),
      radial-gradient(circle at 90% 8%, rgba(20, 184, 166, 0.18), transparent 28%),
      radial-gradient(circle at 78% 88%, rgba(246, 168, 90, 0.18), transparent 30%),
      linear-gradient(135deg, #F4F0FF 0%, #EFFAFF 38%, #FFF7EA 74%, #F7F5FF 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1320px;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 2.25rem 2.2rem 1.85rem 2.2rem;
    border: 1px solid rgba(255, 255, 255, 0.72);
    border-radius: 34px;
    background:
      linear-gradient(135deg, rgba(255, 255, 255, 0.90), rgba(255, 255, 255, 0.68)),
      linear-gradient(120deg, rgba(104, 100, 247, 0.14), rgba(20, 184, 166, 0.11));
    box-shadow: 0 28px 85px rgba(53, 65, 112, 0.14);
    backdrop-filter: blur(18px);
    margin-bottom: 1.25rem;
}

.hero::before {
    content: "";
    position: absolute;
    width: 360px;
    height: 360px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(104, 100, 247, 0.20), transparent 64%);
    right: -95px;
    top: -130px;
}

.hero::after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(20, 184, 166, 0.17), transparent 66%);
    left: -110px;
    bottom: -150px;
}

.hero > * {
    position: relative;
    z-index: 1;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(104, 100, 247, 0.12);
    color: #3730A3;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    margin-bottom: 1rem;
    border: 1px solid rgba(104, 100, 247, 0.16);
}

.hero h1 {
    font-size: clamp(2.25rem, 5vw, 4.7rem);
    line-height: 0.95;
    letter-spacing: -0.07em;
    margin: 0;
    color: var(--ink);
}

.hero p {
    margin-top: 1rem;
    color: #475569;
    font-size: 1.05rem;
    max-width: 880px;
    line-height: 1.72;
}

.metric-card {
    border-radius: 24px;
    padding: 1.15rem 1.25rem;
    background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(250,247,255,0.78));
    border: 1px solid rgba(255,255,255,0.74);
    box-shadow: 0 18px 48px rgba(53, 65, 112, 0.10);
    min-height: 125px;
}

.metric-label {
    color: #69758A;
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .06em;
}

.metric-value {
    color: var(--ink);
    font-size: 1.78rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin-top: 0.25rem;
}

.insight-box {
    padding: 1.1rem 1.2rem;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(104,100,247,0.12), rgba(20,184,166,0.11), rgba(246,168,90,0.10));
    border: 1px solid rgba(104, 100, 247, 0.16);
    color: var(--ink);
    margin: .75rem 0;
    box-shadow: 0 18px 52px rgba(53, 65, 112, 0.07);
}

.result-extreme {
    border-left: 7px solid var(--rose);
    padding: 1rem 1.1rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #FFF1F3, #FFF7EA);
    border-top: 1px solid #FFD3DA;
    border-right: 1px solid #FFD3DA;
    border-bottom: 1px solid #FFD3DA;
    box-shadow: 0 18px 50px rgba(224,82,105,0.10);
}

.result-moderate {
    border-left: 7px solid var(--brand);
    padding: 1rem 1.1rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #F2F0FF, #ECFEFF);
    border-top: 1px solid #DCD8FF;
    border-right: 1px solid #DCD8FF;
    border-bottom: 1px solid #DCD8FF;
    box-shadow: 0 18px 50px rgba(104,100,247,0.10);
}

.small-muted {
    color: var(--muted);
    font-size: 0.88rem;
}

.stButton>button {
    border-radius: 15px;
    font-weight: 800;
    border: 1px solid rgba(104,100,247,0.22);
    background: linear-gradient(135deg, #6864F7, #14B8A6);
    color: white;
    box-shadow: 0 16px 34px rgba(104,100,247,0.22);
    transition: transform .18s ease, box-shadow .18s ease;
}

.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 20px 44px rgba(104,100,247,0.28);
    color: white;
    border-color: rgba(104,100,247,0.28);
}

[data-testid="stSidebar"] {
    background:
      radial-gradient(circle at top left, rgba(104,100,247,0.18), transparent 42%),
      radial-gradient(circle at bottom right, rgba(20,184,166,0.15), transparent 40%),
      linear-gradient(180deg, #F8F5FF 0%, #EFFAFF 100%);
    border-right: 1px solid rgba(104,100,247,0.12);
}

[data-testid="stSidebar"] h2 {
    letter-spacing: -0.04em;
    color: var(--ink);
}

[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    color: #69758A;
    letter-spacing: .06em;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    margin: 0.42rem 0;
    padding: 0.74rem 0.82rem;
    border-radius: 17px;
    background: rgba(255,255,255,0.66);
    border: 1px solid rgba(104,100,247,0.13);
    box-shadow: 0 12px 30px rgba(53,65,112,0.07);
    transition: all .18s ease;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.92);
    transform: translateY(-1px);
    border-color: rgba(104,100,247,0.28);
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-weight: 800;
    color: var(--ink);
}

[data-testid="stMetricValue"] {
    font-weight: 800;
    color: var(--ink);
}

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

div[data-testid="stFileUploader"] section {
    border-radius: 22px;
    border: 1.5px dashed rgba(104,100,247,0.36);
    background: rgba(255,255,255,0.62);
}

h1, h2, h3 {
    color: var(--ink);
}

p, li, .stMarkdown {
    color: #334155;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data(show_spinner=False)
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def metric_card(label, value, help_text=None):
    help_html = f"<div class='small-muted'>{help_text}</div>" if help_text else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero():
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-badge">🧭 NLP in Business · Opinion Intelligence</div>
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}. This application classifies opinions into <b>Extreme</b> and <b>Moderate</b> categories, detects opinion direction, assigns confidence, and converts raw feedback into business action priorities.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_model_missing():
    st.error("Model artifact not found: `models/opinion_intensity_pipeline.joblib`")
    st.info(
        "Run the training cell first. After the model is created inside the `models/` folder, refresh this app."
    )


def find_text_columns(df):
    candidates = []
    for c in df.columns:
        if df[c].dtype == "object":
            avg_len = df[c].dropna().astype(str).str.len().mean()
            if pd.notna(avg_len) and avg_len > 15:
                candidates.append(c)
    return candidates or list(df.select_dtypes(include=["object"]).columns)


def get_keywords(pipeline, text, top_n=8):
    try:
        tfidf = pipeline.named_steps["tfidf"]
        X = tfidf.transform([text])
        scores = X.toarray().ravel()
        if scores.sum() == 0:
            return []
        top_idx = scores.argsort()[::-1][:top_n]
        names = np.array(tfidf.get_feature_names_out())
        return [names[i] for i in top_idx if scores[i] > 0]
    except Exception:
        return []


def plot_wordcloud(texts, title):
    combined = " ".join([clean_text(x) for x in texts if isinstance(x, str)])
    if not combined.strip():
        st.info("Not enough text for word cloud.")
        return
    wc = WordCloud(width=1200, height=480, background_color="white", collocations=False, max_words=90).generate(combined)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    st.caption(title)


def home_page(metrics):
    hero()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Dataset", "Yelp Full", "Large public review dataset")
    with c2:
        rows = metrics.get("training_rows", "Train first")
        metric_card("Training Rows", f"{rows:,}" if isinstance(rows, int) else rows)
    with c3:
        acc = metrics.get("binary_extreme_moderate_accuracy", "—")
        metric_card("Binary Accuracy", f"{acc:.2%}" if isinstance(acc, float) else acc)
    with c4:
        f1 = metrics.get("binary_extreme_moderate_macro_f1", "—")
        metric_card("Macro F1", f"{f1:.2%}" if isinstance(f1, float) else f1)

    st.markdown("### What the project does")
    st.markdown(
        """
        This project solves a practical NLP problem: brands receive thousands of reviews, comments, survey responses, and social media opinions. 
        Not all opinions require the same urgency. The app separates **extreme opinions** from **moderate opinions**, then explains whether the opinion is strongly negative, mildly negative, neutral/mixed, mildly positive, or strongly positive.
        """
    )

    st.markdown("### Project workflow")
    workflow = pd.DataFrame({
        "Stage": [
            "1. Data collection",
            "2. Preprocessing",
            "3. Feature extraction",
            "4. Model training",
            "5. Prediction",
            "6. Business insight",
        ],
        "Implementation": [
            "Yelp Review Full dataset with star-labeled review text",
            "Lowercasing, URL/user masking, punctuation cleanup, whitespace normalization",
            "TF-IDF with unigrams and bigrams",
            "Logistic-style SGD classifier trained on five opinion intensity levels",
            "Extreme vs Moderate category, confidence, star-level intensity, direction",
            "Priority tagging and recommended business action for CX/marketing teams",
        ],
    })
    st.dataframe(workflow, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="insight-box">
            <b>Business interpretation:</b> Extreme negative feedback is useful for complaint escalation and churn-risk monitoring. 
            Extreme positive feedback is useful for testimonials, referrals, and brand advocacy. Moderate opinions help identify improvement patterns without overreacting.
        </div>
        """,
        unsafe_allow_html=True,
    )


def single_analyzer_page(pipeline):
    st.markdown("## 🔎 Single Opinion Analyzer")
    st.caption("Paste one customer review, survey comment, product opinion, or social media-style comment.")

    examples = {
        "Extreme Negative": "This was the worst service I have ever experienced. I waited for hours, nobody helped me, and I will never buy from this brand again.",
        "Moderate Negative": "The product is okay, but delivery was late and support could have communicated better.",
        "Neutral / Mixed": "The app has useful features, but some parts are confusing. Overall, it is average.",
        "Moderate Positive": "The service was good and the staff was helpful. I would probably recommend it.",
        "Extreme Positive": "Absolutely amazing experience! The product exceeded every expectation and I will definitely recommend it to everyone.",
    }

    choice = st.selectbox("Try a sample opinion", ["Write my own"] + list(examples.keys()))
    default_text = "" if choice == "Write my own" else examples[choice]

    text = st.text_area(
        "Opinion text",
        value=default_text,
        height=190,
        placeholder="Example: The support team solved my issue quickly, but the app still feels slightly slow...",
    )

    analyze = st.button("Analyze Opinion", type="primary", use_container_width=True)

    if analyze:
        if not text.strip():
            st.warning("Please enter an opinion text.")
            return

        pred = prediction_frame(pipeline, [text]).iloc[0]
        box_class = "result-extreme" if pred["category"] == "Extreme" else "result-moderate"

        st.markdown(
            f"""
            <div class="{box_class}">
                <h3 style="margin:0;">{pred['category']} Opinion</h3>
                <p style="margin:.35rem 0 0 0;"><b>Intensity:</b> {pred['intensity_type']} · <b>Confidence:</b> {pred['confidence']:.1%}</p>
                <p style="margin:.35rem 0 0 0;"><b>Recommended action:</b> {pred['recommended_business_action']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Extreme Probability", f"{pred['extreme_probability']:.1%}")
        c2.metric("Moderate Probability", f"{pred['moderate_probability']:.1%}")
        c3.metric("Priority", pred["priority"])

        prob_df = pd.DataFrame({
            "Category": ["Extreme", "Moderate"],
            "Probability": [pred["extreme_probability"], pred["moderate_probability"]],
        })
        fig = px.bar(
            prob_df,
            x="Category",
            y="Probability",
            text_auto=".1%",
            title="Extreme vs Moderate Probability",
            range_y=[0, 1],
            color="Category",
            color_discrete_map={"Extreme": "#E05269", "Moderate": "#6864F7"},
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.55)")
        st.plotly_chart(fig, use_container_width=True)

        keywords = get_keywords(pipeline, text)
        if keywords:
            st.markdown("### Important terms detected")
            st.write(" · ".join([f"`{k}`" for k in keywords]))

        with st.expander("View model output table"):
            st.dataframe(pd.DataFrame([pred]), use_container_width=True, hide_index=True)


def batch_dashboard_page(pipeline):
    st.markdown("## 📊 Batch Opinion Dashboard")
    st.caption("Upload a CSV file with customer reviews, survey comments, product feedback, or social media opinions.")

    st.download_button(
        "Download sample CSV format",
        data=Path("data/sample_reviews.csv").read_text(encoding="utf-8") if Path("data/sample_reviews.csv").exists() else "review_text\nThis product is excellent\nThe service was okay\n",
        file_name="sample_reviews.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV to generate charts, batch classifications, and downloadable outputs.")
        return

    df = pd.read_csv(uploaded)
    st.markdown("### Uploaded data preview")
    st.dataframe(df.head(10), use_container_width=True)

    text_columns = find_text_columns(df)
    if not text_columns:
        st.error("No text column found in the uploaded file.")
        return

    selected_col = st.selectbox("Select the text column to analyze", text_columns)

    row_limit = int(min(5000, len(df)))
    if row_limit <= 0:
        st.error("No rows available to analyze.")
        return
    elif row_limit < 10:
        max_rows = row_limit
        st.caption(f"Small file detected: analyzing all {row_limit} rows.")
    else:
        max_rows = st.slider(
            "Rows to analyze",
            min_value=10,
            max_value=row_limit,
            value=int(min(1000, row_limit)),
            step=10,
        )

    if st.button("Run Batch Analysis", type="primary", use_container_width=True):
        work_df = df.head(max_rows).copy()
        texts = work_df[selected_col].fillna("").astype(str).tolist()

        with st.spinner("Classifying opinions..."):
            preds = prediction_frame(pipeline, texts)

        result = pd.concat([work_df.reset_index(drop=True), preds.drop(columns=["text"]).reset_index(drop=True)], axis=1)

        st.success(f"Analyzed {len(result):,} opinions.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Opinions", f"{len(result):,}")
        c2.metric("Extreme Opinions", f"{(result['category'] == 'Extreme').sum():,}")
        c3.metric("Moderate Opinions", f"{(result['category'] == 'Moderate').sum():,}")
        c4.metric("Critical Cases", f"{(result['priority'] == 'Critical').sum():,}")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            cat_counts = result["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig = px.pie(
                cat_counts,
                names="category",
                values="count",
                hole=0.55,
                title="Extreme vs Moderate Share",
                color="category",
                color_discrete_map={"Extreme": "#E05269", "Moderate": "#6864F7"},
            )
            fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            detail_counts = result["intensity_type"].value_counts().reset_index()
            detail_counts.columns = ["intensity_type", "count"]
            detail_order = ["Extreme Negative", "Moderate Negative", "Neutral / Mixed", "Moderate Positive", "Extreme Positive"]
            fig = px.bar(
                detail_counts,
                x="intensity_type",
                y="count",
                text_auto=True,
                title="Detailed Opinion Intensity",
                category_orders={"intensity_type": detail_order},
                color="intensity_type",
                color_discrete_sequence=["#E05269", "#F6A85A", "#8CA3AF", "#14B8A6", "#6864F7"],
            )
            fig.update_layout(height=420, xaxis_title="", yaxis_title="Count", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.55)")
            st.plotly_chart(fig, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            fig = px.histogram(result, x="confidence", nbins=20, title="Model Confidence Distribution", color_discrete_sequence=["#14B8A6"])
            fig.update_layout(height=390, xaxis_title="Confidence", yaxis_title="Number of opinions", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.55)")
            st.plotly_chart(fig, use_container_width=True)

        with chart_col4:
            priority_counts = result["priority"].value_counts().reset_index()
            priority_counts.columns = ["priority", "count"]
            fig = px.bar(
                priority_counts,
                x="priority",
                y="count",
                text_auto=True,
                title="Business Action Priority",
                color="priority",
                color_discrete_map={"Critical": "#E05269", "High": "#F6A85A", "Normal": "#6864F7"},
            )
            fig.update_layout(height=390, xaxis_title="", yaxis_title="Count", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.55)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Extreme opinion word cloud")
        extreme_texts = result.loc[result["category"] == "Extreme", selected_col].dropna().astype(str).tolist()
        plot_wordcloud(extreme_texts, "Most frequent terms inside opinions classified as Extreme.")

        st.markdown("### Highest-priority opinions")
        priority_df = result.sort_values(["priority", "confidence"], ascending=[True, False])
        st.dataframe(priority_df.head(30), use_container_width=True)

        st.markdown("### Full classified output")
        st.dataframe(result, use_container_width=True)

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download classified CSV",
            data=csv,
            file_name="classified_opinions.csv",
            mime="text/csv",
            use_container_width=True,
        )


def model_dataset_page(metrics):
    st.markdown("## 🧠 Model, Dataset & Methodology")
    st.markdown("### Dataset")
    st.write(
        "The project is designed around the Yelp Review Full dataset. Review star ratings are converted into opinion intensity labels: "
        "1 star = Extreme Negative, 2 stars = Moderate Negative, 3 stars = Neutral/Mixed, 4 stars = Moderate Positive, 5 stars = Extreme Positive."
    )

    mapping = pd.DataFrame({
        "Original Label": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
        "New NLP Label": [
            "Extreme Negative",
            "Moderate Negative",
            "Neutral / Mixed",
            "Moderate Positive",
            "Extreme Positive",
        ],
        "Project Category": ["Extreme", "Moderate", "Moderate", "Moderate", "Extreme"],
        "Business Meaning": [
            "Complaint escalation, possible churn, reputation risk",
            "Service/product improvement signal",
            "Mixed or average satisfaction",
            "Positive but not highly emotional",
            "Brand advocacy, referral/testimonial opportunity",
        ],
    })
    st.dataframe(mapping, use_container_width=True, hide_index=True)

    st.markdown("### Methodology")
    method = pd.DataFrame({
        "Component": ["Preprocessing", "Feature Extraction", "ML Model", "Evaluation", "Business Output"],
        "Description": [
            "Normalize raw opinion text, remove noise, mask URLs/users, clean punctuation",
            "TF-IDF vectors with unigram and bigram features",
            "SGDClassifier with logistic-loss probability output",
            "Accuracy, macro F1, precision, recall, confusion matrix",
            "Extreme/moderate classification, confidence, action priority, dashboard insights",
        ],
    })
    st.dataframe(method, use_container_width=True, hide_index=True)

    st.markdown("### Trained model metrics")
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training Rows", f"{metrics.get('training_rows', 0):,}")
        c2.metric("Testing Rows", f"{metrics.get('testing_rows', 0):,}")
        c3.metric("Binary Accuracy", f"{metrics.get('binary_extreme_moderate_accuracy', 0):.2%}")
        c4.metric("Binary Macro F1", f"{metrics.get('binary_extreme_moderate_macro_f1', 0):.2%}")

        if "binary_class_metrics" in metrics:
            st.dataframe(pd.DataFrame(metrics["binary_class_metrics"]).T, use_container_width=True)

        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            labels = metrics.get("confusion_matrix_labels", [STAR_LABELS[i] for i in range(5)])
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            fig = px.imshow(cm_df, text_auto=True, aspect="auto", title="Five-Level Confusion Matrix", color_continuous_scale="Purples")
            fig.update_layout(height=560, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("View full metrics JSON"):
            st.json(metrics)
    else:
        st.warning("Metrics file not found. Train the model to generate `models/metrics.json`.")


def main():
    pipeline = load_pipeline()
    metrics = load_metrics()

    st.sidebar.markdown("## 🧭 OpinionLens AI")
    st.sidebar.markdown(
        "<div style='font-size:0.92rem;color:#64748B;line-height:1.55;margin-top:-0.3rem;margin-bottom:1rem;'>Opinion intensity dashboard for customer and market insights.</div>",
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Single Opinion Analyzer", "Batch CSV Dashboard", "Model & Dataset"],
    )

    st.sidebar.markdown("---")
    if pipeline is not None:
        st.sidebar.markdown(
            "<div style='padding:0.85rem;border-radius:16px;background:rgba(20,184,166,0.12);border:1px solid rgba(20,184,166,0.20);font-weight:800;color:#0F766E;'>✓ Model ready</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            "<div style='padding:0.85rem;border-radius:16px;background:rgba(249,115,136,0.12);border:1px solid rgba(249,115,136,0.20);font-weight:800;color:#BE123C;'>Model not loaded</div>",
            unsafe_allow_html=True,
        )

    if page == "Home":
        home_page(metrics)
        if pipeline is None:
            show_model_missing()
    elif page == "Single Opinion Analyzer":
        if pipeline is None:
            show_model_missing()
        else:
            single_analyzer_page(pipeline)
    elif page == "Batch CSV Dashboard":
        if pipeline is None:
            show_model_missing()
        else:
            batch_dashboard_page(pipeline)
    elif page == "Model & Dataset":
        model_dataset_page(metrics)


if __name__ == "__main__":
    main()
