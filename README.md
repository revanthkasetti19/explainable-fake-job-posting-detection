# JobGuard — Explainable Fake Job Posting Detection

A beginner-friendly final-year B.Tech Machine Learning project that classifies job postings as legitimate or potentially fraudulent using TF-IDF and Logistic Regression. The Streamlit app displays model estimates and influential words from the trained linear model.

## Dataset

Download the **Real or Fake Fake Job Posting Prediction** dataset from Kaggle:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Place the CSV containing the `fraudulent` column in this folder and name it:

`fake_job_postings.csv`

The target is expected to use `0` for legitimate and `1` for fraudulent. Check the dataset's license and terms before redistributing it.

## Requirements

- Python 3.10 or newer recommended
- pip
- Internet access to download the dataset once

## Run on Windows (VS Code terminal)

```bash
python -m venv .venv
```

Activate the environment:

**PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**Git Bash**
```bash
source .venv/Scripts/activate
```

Install packages:

```bash
python -m pip install -r requirements.txt
```

Train the model and create evaluation artifacts:

```bash
python train_model.py
```

Run the app:

```bash
streamlit run app.py
```

Streamlit will show a local URL, usually `http://localhost:8501`.

## Project modules

1. Dataset loading and validation
2. Data preparation and text combination
3. TF-IDF feature extraction
4. Logistic Regression training
5. Evaluation with accuracy, precision, recall, F1-score, classification report, and confusion matrix
6. Interactive prediction UI
7. Explainability using signed contributions from the linear model

## CRISP-DM mapping

1. **Business understanding:** identify the need to flag suspicious job advertisements for review.
2. **Data understanding:** inspect the labels, missing values, class balance, and text fields.
3. **Data preparation:** combine selected fields, handle missing values, and transform text using TF-IDF.
4. **Modeling:** train a Logistic Regression classifier with class weights.
5. **Evaluation:** use a stratified holdout test set and report class-specific metrics.
6. **Deployment:** serve predictions through Streamlit.

## Important limitations

- This is an academic prototype, not a fraud verification service.
- A model output is not proof that an employer or job is fraudulent.
- Dataset bias, label quality, and changes in scam behavior can reduce performance on new postings.
- The probability shown by the model is not necessarily calibrated to the true real-world chance of fraud.
- Do not use this model as the sole basis for rejecting an employer or making a financial or employment decision.
- The word explanation shows learned model associations, not causal evidence.
- Do not evaluate the project on training data only. Use held-out test data and discuss false positives and false negatives.

## Suggested extensions

- Compare Logistic Regression with Multinomial Naive Bayes and Linear SVM.
- Tune thresholds based on the cost of missed fraud versus false alarms.
- Add SHAP/LIME explanations if you can support and validate the added complexity.
- Add model versioning and a data drift discussion.
- Add manual checks for suspicious contact requests, upfront fees, and unverifiable employer details, clearly labeled as heuristic rules rather than ML conclusions.

## Suggested final report chapters

1. Abstract
2. Introduction and problem statement
3. Literature review
4. Dataset description
5. CRISP-DM methodology
6. System design and implementation
7. Experimental results
8. Limitations and ethical considerations
9. Conclusion and future scope
10. References
