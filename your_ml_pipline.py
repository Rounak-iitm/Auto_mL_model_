import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Global variables to store results
model_scores = []
feature_importance_df = None
best_model = None

def run_pipeline(df, target_column):
    global model_scores, feature_importance_df, best_model

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical variables
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = []
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = f1_score(y_test, preds, average='weighted')
        results.append({"Model": name, "F1 Score": score})

        if score > best_score:
            best_score = score
            best_model = model
            # Feature importance
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": importance
                }).sort_values(by="Importance", ascending=False)

    model_scores = pd.DataFrame(results)

def get_model_scores():
    return model_scores

def get_feature_importance():
    return feature_importance_df

def get_best_model():
    return best_model
