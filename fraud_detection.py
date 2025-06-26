import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib

def load_and_prepare_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: Dataset file not found. Please ensure 'social_media_users.csv' is in the project directory.")
        return None, None, None
    X = data[['post_count', 'follower_count', 'following_count', 'account_age_days']]
    y = data['is_fake']
    return X, y, data

def train_model(X, y):
    # Use stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, 'Bot_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model Accuracy on Test Set: {accuracy:.2f}")
    # Additional metrics
    y_pred = model.predict(X_test_scaled)
    # Check unique classes in test set
    unique_classes = np.unique(y_test)
    if len(unique_classes) > 1:
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Genuine', 'Fake'], labels=[0, 1]))
    else:
        print("\nWarning: Test set contains only one class. Confusion matrix and classification report cannot be generated.")
        print(f"Predicted class: {'Fake' if unique_classes[0] == 1 else 'Genuine'}")
    return model, scaler

def predict_account(model, scaler):
    print("\nEnter user profile details for prediction:")
    try:
        post_count = float(input("Post Count: "))
        follower_count = float(input("Follower Count: "))
        following_count = float(input("Following Count: "))
        account_age_days = float(input("Account Age (Days): "))
    except ValueError:
        print("Error: Please enter numeric values.")
        return
    features = pd.DataFrame({
        'post_count': [post_count],
        'follower_count': [follower_count],
        'following_count': [following_count],
        'account_age_days': [account_age_days]
    })
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    if prediction[0] == 1:
        print("FRAUD USER DETECTED IN SOCIAL NETWORK")
    else:
        print("GENUINE USER DETECTED IN SOCIAL NETWORK")

def main():
    X, y, data = load_and_prepare_data('social_media_users.csv')
    if X is None:
        return
    print("Training ANN model...")
    model, scaler = train_model(X, y)
    while True:
        predict_account(model, scaler)
        again = input("\nDo you want to predict another account? (yes/no): ").lower()
        if again != 'yes':
            break

if __name__ == "__main__":
    main()