from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_and_evaluate(X, y, scaler, le_dict, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10]
    }

    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Best Parameters:", grid.best_params_)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

    # Save everything
    joblib.dump((model, scaler, le_dict, feature_names), 'rf_dropout_model.pkl')
    print("Model saved as rf_dropout_model.pkl")

    return model
