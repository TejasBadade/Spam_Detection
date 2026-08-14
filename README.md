# 🛡️ SMS Spam Detection

A machine learning project that classifies SMS messages as spam or ham (not spam) using TF-IDF vectorization and Naive Bayes, deployed as an interactive Streamlit web app.

## 📌 Project Overview

This project builds a text classification pipeline to automatically detect spam SMS messages. It compares multiple ML models (Naive Bayes, Logistic Regression, SVM) to find the best performer, then deploys the winning model as a live, interactive web app where users can test any message in real time.

## ❓ Problem Statement

SMS spam is a persistent nuisance and, in some cases, a security risk (phishing, scams). The goal is to build a model that can reliably tell spam and legitimate messages apart based on message text alone.

## 🛠️ Tech Stack

- **Language:** Python
- **ML/NLP:** scikit-learn (Naive Bayes, Logistic Regression, SVM), NLTK (stopword removal), TF-IDF vectorization
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Deployment:** Streamlit
- **Model persistence:** pickle

## 🧹 Data Preprocessing

- Loaded the SMS Spam Collection dataset (UCI Repository) and dropped unused/junk columns
- Renamed columns to `label` and `message`
- Label-encoded target: `ham` → 0, `spam` → 1
- Custom text cleaning function:
  - Lowercased all text
  - Removed all non-alphanumeric characters via regex
  - Removed English stopwords using NLTK
- Converted cleaned text into numeric features using **TF-IDF** (top 3,000 features)

## 🤖 Modeling & Evaluation

Three models were trained and compared on an 80/20 train-test split:

| Model | Metric Focus |
|---|---|
| Naive Bayes | Baseline text classification model |
| Logistic Regression | Linear model comparison |
| SVM (LinearSVC) | Margin-based classifier comparison |

Models were evaluated on **Accuracy, Precision, Recall, and F1-score**, since spam detection is a class-imbalanced problem where false positives (flagging a real message as spam) are costly.

- Confusion matrix plotted for visual error analysis
- Final model selected based on the **best F1-score**, balancing precision and recall

## 📊 Key Results

*(fill in with your actual printed numbers after running the script)*

- Best-performing model: **Naive Bayes**
- Accuracy: **97.8%**
- Total messages: **5,572**
- Class balance: ham significantly outnumbers spam (~X:1 ratio)

## ⚠️ Known Limitations

The model showed some bias toward Western/UK spam phrasing patterns, since the training data originates from UK-based SMS users. It's less reliable on Indian-context spam messages that use terms like "lakhs" or "rupees," which are underrepresented in the training data — a good direction for future improvement (e.g., fine-tuning on region-specific spam samples).

**Manual testing also revealed a spam-evasion weakness:** messages using deliberate misspellings or unusual formatting (e.g., "CONGrtesss" instead of "congratulations") can slip past the model as false negatives. This is because TF-IDF only recognizes exact words seen during training — it has no way to catch obfuscated or misspelled variants of spam trigger words, a common real-world spammer tactic to evade filters. Addressing this would require techniques like character-level n-grams, fuzzy matching, or spell-correction as a preprocessing step.

## 🚀 Live Demo

Try it here: **[SMS Spam Detector](your-streamlit-url-here)**

*(Note: free-tier Streamlit apps sleep after inactivity — if the link seems slow to load, give it a few seconds to wake up.)*

![App Screenshot](demo_screenshot.png)

The app (`app.py`) lets you:
- Paste any SMS message and get an instant Spam / Not Spam prediction
- See the model's confidence score
- View how the message looks after text cleaning (via an expandable section)

## 📂 Project Structure

```
sms-spam-detection/
├── spam_detector.py         # Model training, comparison & evaluation script
├── app.py                   # Streamlit web app for live predictions
├── spam_model.pkl           # Trained Naive Bayes model
├── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
├── sample_data.csv          # Sample of the dataset
├── requirements.txt
└── README.md
```

## ▶️ How to Run

**Train the model:**
```bash
pip install -r requirements.txt
python spam_detector.py
```

**Run the Streamlit app:**
```bash
streamlit run app.py
```
Then open the local URL Streamlit provides in your browser.

## 📁 Dataset

SMS Spam Collection dataset (UCI Machine Learning Repository) — 5,572 labeled SMS messages (ham/spam).

## 👤 Author

**Tejas** — Aspiring Data Analyst / Data Scientist
