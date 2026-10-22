# Explainable Fake Job Posting Detection — Project Report Template

> Replace every bracketed placeholder with results from your own implementation. Do not invent model scores or claim tests you did not perform.

## Abstract
This project develops a machine learning prototype to classify job advertisements as legitimate or potentially fraudulent. The workflow follows CRISP-DM, including business understanding, data understanding, data preparation, modeling, evaluation, and deployment. Text fields are converted to TF-IDF features and classified using Logistic Regression. The interface presents a prediction estimate and influential words to help users review a posting. The system is a decision-support prototype and cannot independently verify fraud.

## 1. Problem Statement
Online job seekers may encounter misleading or fraudulent job advertisements. The objective is to build a model that learns patterns from a labeled dataset and flags postings that merit additional review.

## 2. Objectives
- Prepare and explore a labeled job-posting dataset.
- Train a baseline text-classification model.
- Compare model performance using suitable metrics.
- Explain influential words or phrases for individual predictions.
- Deploy the trained pipeline in a simple web interface.
- Document limitations, bias, and ethical use.

## 3. Dataset
- Dataset name: Real or Fake Fake Job Posting Prediction
- Source URL: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
- Total rows used: [fill after cleaning]
- Number of legitimate postings: [fill]
- Number of fraudulent postings: [fill]
- Features used: [list actual columns]
- Target definition: `fraudulent` (0 = legitimate, 1 = fraudulent), after confirming the dataset encoding.

## 4. CRISP-DM
### 4.1 Business Understanding
[Describe intended users and why a suspicious-posting flag may help.]

### 4.2 Data Understanding
[Include shape, data types, missing-value summary, class distribution, and charts.]

### 4.3 Data Preparation
[Describe text fields combined, missing values, text normalization, split strategy, and leakage prevention.]

### 4.4 Modeling
[Describe TF-IDF and Logistic Regression. If comparing models, document settings.]

### 4.5 Evaluation
Report the results measured by your own run.

| Metric | Result |
|---|---:|
| Accuracy | [fill] |
| Fraud precision | [fill] |
| Fraud recall | [fill] |
| Fraud F1-score | [fill] |

Discuss false positives and false negatives using the confusion matrix.

### 4.6 Deployment
[Explain how the Streamlit interface loads the saved model and displays predictions and word contributions.]

## 5. Explainability
The classifier's linear coefficients and the input's TF-IDF values are used to calculate signed word contributions. Positive contributions push the score toward the fraudulent class and negative contributions push it toward the legitimate class. These are model associations, not proof of fraud or causal explanations.

## 6. Results and Discussion
[Insert your measured confusion matrix, classification report, model comparison table, and example outputs. Discuss performance on the held-out test set.]

## 7. Limitations and Ethics
- The labels may be noisy or outdated.
- Dataset patterns may not generalize to current job markets.
- A suspicious classification is not proof of fraud.
- False positives may unfairly damage trust in legitimate employers.
- Do not enter sensitive personal data.
- The system should support human review, not automatically make employment decisions.

## 8. Conclusion and Future Scope
[Summarize actual results. Potential extensions: compare Naive Bayes and Linear SVM, threshold tuning, drift monitoring, and careful explanation validation.]

## References
1. Dataset: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
2. Scikit-learn documentation: https://scikit-learn.org/stable/
3. Streamlit documentation: https://docs.streamlit.io/
