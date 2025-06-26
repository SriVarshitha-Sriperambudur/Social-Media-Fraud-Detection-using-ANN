# Social-Media-Fraud-Detection-using-ANN

This project implements a Fraud Detection System for social media platforms using a Neural Network model (MLPClassifier from scikit-learn). It helps classify whether a user account is genuine or fake based on behavioral metrics such as post count, followers, and account age.

🚀 Features:

📥 Loads social media user data from a CSV file

⚙️ Preprocesses and scales features with StandardScaler

🧠 Trains an Artificial Neural Network with hidden layers using MLPClassifier

📊 Evaluates performance using confusion matrix and classification report

🧪 Predicts new user profiles via interactive CLI

💾 Model and scaler persistence using joblib


📂 Input Features:

post_count: Number of posts published by the user

follower_count: Number of followers

following_count: Number of accounts the user follows

account_age_days: How old the account is (in days)


🧠 Model:

MLPClassifier (ANN) with hidden layers: (100, 50)

Train-Test Split with stratification (80-20)

StandardScaler used for normalization

Evaluation Metrics:

Accuracy

Confusion Matrix

Precision, Recall, F1-score (via Classification Report)
