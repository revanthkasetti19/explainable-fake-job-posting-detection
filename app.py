from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fraud_detection_model.joblib"
METRICS_PATH = ROOT / "metrics.json"
CM_PATH = ROOT / "confusion_matrix.png"
REPORT_PATH = ROOT / "classification_report.txt"

st.set_page_config(page_title="JobGuard | Explainable Job Screening", page_icon="🛡️", layout="wide")

st.title("🛡️ JobGuard")
st.subheader("Explainable Fake Job Posting Detection")
st.write("Screen a job advertisement for patterns associated with fraudulent postings. This is a decision-support tool, not proof that a job is fraudulent.")

if not MODEL_PATH.exists():
    st.warning("The trained model has not been created yet.")
    st.markdown("**First:** download the dataset, save it as `fake_job_postings.csv` in this project folder, then run `python train_model.py`.")
    st.code("pip install -r requirements.txt\npython train_model.py\nstreamlit run app.py", language="bash")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.sidebar:
    st.header("About this project")
    st.write("**Method:** CRISP-DM")
    st.write("**Model:** TF-IDF + Logistic Regression")
    st.write("**Explainability:** influential words from the trained linear model")
    st.caption("Do not enter personal or confidential information.")

st.markdown("### Enter job posting details")
col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Job title", placeholder="e.g. Junior Data Analyst")
    company_profile = st.text_area("Company profile", height=120, placeholder="What does the company do?")
    requirements = st.text_area("Requirements", height=150, placeholder="Skills, qualifications, experience...")
with col2:
    location = st.text_input("Location", placeholder="e.g. Hyderabad, India")
    employment_type = st.selectbox("Employment type", ["", "Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"])
    description = st.text_area("Job description", height=180, placeholder="Describe the role and responsibilities...")
    benefits = st.text_area("Benefits (optional)", height=90, placeholder="Salary, benefits, training, etc.")

if st.button("Analyze posting", type="primary", use_container_width=True):
    combined_text = " ".join([
        title, company_profile, description, requirements, benefits,
        employment_type, location
    ]).strip()

    if len(combined_text.split()) < 5:
        st.error("Please enter more job information so the model has enough text to analyze.")
    else:
        probabilities = model.predict_proba([combined_text])[0]
        classes = list(model.named_steps["classifier"].classes_)
        fraud_index = classes.index(1) if 1 in classes else int(probabilities.argmax())
        fraud_probability = float(probabilities[fraud_index])
        prediction = int(model.predict([combined_text])[0])

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            if prediction == 1:
                st.error("Result: Potentially suspicious posting")
            else:
                st.success("Result: More consistent with legitimate postings")
            st.metric("Estimated fraud probability", f"{fraud_probability:.1%}")
        with right:
            st.progress(min(max(fraud_probability, 0.0), 1.0))
            st.caption("This probability is a model estimate, not a verified real-world fraud probability.")

        # Explain a linear model by calculating each token's signed contribution.
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]
        vector = vectorizer.transform([combined_text])
        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]
        present_indices = vector.nonzero()[1]
        contributions = []
        for idx in present_indices:
            value = float(vector[0, idx] * coefficients[idx])
            contributions.append((feature_names[idx], value))
        contributions.sort(key=lambda item: item[1], reverse=True)
        positive = [(word, value) for word, value in contributions if value > 0][:8]
        negative = [(word, value) for word, value in sorted(contributions, key=lambda item: item[1]) if value < 0][:8]

        st.markdown("### Explanation")
        st.write("These are influential words/phrases in the linear model. Positive contributions push toward the fraudulent class; negative contributions push toward the legitimate class. They are associations learned from the dataset, not proof.")
        exp1, exp2 = st.columns(2)
        with exp1:
            st.markdown("**Words pushing toward potential fraud**")
            if positive:
                st.dataframe(pd.DataFrame(positive, columns=["Word / phrase", "Contribution"]), use_container_width=True, hide_index=True)
            else:
                st.info("No strong positive word contributions were found.")
        with exp2:
            st.markdown("**Words pushing toward legitimate**")
            if negative:
                st.dataframe(pd.DataFrame(negative, columns=["Word / phrase", "Contribution"]), use_container_width=True, hide_index=True)
            else:
                st.info("No strong negative word contributions were found.")

st.markdown("---")
st.markdown("### Model evaluation")
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    b.metric("Fraud precision", f"{metrics.get('precision_fraud', 0):.1%}")
    c.metric("Fraud recall", f"{metrics.get('recall_fraud', 0):.1%}")
    d.metric("Fraud F1-score", f"{metrics.get('f1_fraud', 0):.1%}")
    st.caption(metrics.get("note", "Metrics depend on the dataset and test split."))
else:
    st.info("Run train_model.py to create evaluation metrics.")

if CM_PATH.exists():
    st.image(str(CM_PATH), caption="Confusion matrix on the held-out test set", use_container_width=True)

if REPORT_PATH.exists():
    with st.expander("View classification report"):
        st.code(REPORT_PATH.read_text(encoding="utf-8"))
