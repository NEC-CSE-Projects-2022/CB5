# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import pandas as pd
# from .preprocess import preprocess_data

# def train_model(filepath):
#     # Preprocess the data
#     X, y = preprocess_data(filepath,target_column='Label')
    
#     # Split the data
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     # Train the Decision Tree Classifier
#     model = DecisionTreeClassifier(random_state=42)
#     model.fit(X_train, y_train)
    
#     # Predict and evaluate
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred, output_dict=True)
#     confusion = confusion_matrix(y_test, y_pred)
    
#     # Prepare results
#     results = {
#         'accuracy': accuracy,
#         'classification_report': report,
#         'confusion_matrix': confusion.tolist(),
#     }
#     return results
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
from model.preprocess import preprocess_data

def train_model(filepath, target_column='Label'):
    X, y = preprocess_data(filepath, target_column=target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    # Save the model
    joblib.dump(model, './model/decision_tree_model.pkl')

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),
        "correctly_predicted_normal": confusion[0][0],
        "correctly_predicted_malicious": confusion[1][1]
    }
