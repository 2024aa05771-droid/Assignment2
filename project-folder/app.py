import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from collections import Counter

st.title("ML Model Comparison Dashboard")

# -------------------------------
# Upload dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Drop ID column if exists
    if 'CIDs' in df.columns:
        df = df.drop(columns=['CIDs'])

    # Select target column
    target_col = st.selectbox("Select Target Column", df.columns)

    if target_col:

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Convert all to numeric
        X = X.apply(pd.to_numeric, errors='coerce')

        # Handle missing and extreme values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna(axis=1, how='all')

        if X.shape[1] == 0:
            st.error("No numeric features available after preprocessing.")
            st.stop()

        X = X.fillna(X.mean()).fillna(0)
        X = X.clip(lower=-1e6, upper=1e6)

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Remove rare classes
        counts = Counter(y)
        valid_classes = [c for c, cnt in counts.items() if cnt > 1]

        mask = np.array([label in valid_classes for label in y])
        X = X[mask]
        y = y[mask]

        if len(np.unique(y)) < 2:
            st.error("Not enough class diversity after filtering.")
            st.stop()

        # -------------------------------
        # Safe Train-Test Split
        # -------------------------------
        test_size = 0.2
        n_samples = len(y)
        n_classes = len(np.unique(y))

        if isinstance(test_size, float):
            n_test = int(np.floor(test_size * n_samples))
        else:
            n_test = int(test_size)

        stratify_param = y
        if n_test < n_classes:
            st.warning("Too many classes for test size. Using random split.")
            stratify_param = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_param
        )

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Impute again (safety)
        imputer = SimpleImputer(strategy='mean')
        X_train_scaled = imputer.fit_transform(X_train_scaled)
        X_test_scaled = imputer.transform(X_test_scaled)

        # -------------------------------
        # Evaluation Function
        # -------------------------------
        def evaluate_model(name, model, X_test_input, y_test):

            y_pred = model.predict(X_test_input)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_input)
            else:
                y_prob = None

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            mcc = matthews_corrcoef(y_test, y_pred)

            auc = 0
            if y_prob is not None:
                try:
                    if len(np.unique(y_test)) == 2:
                        auc = roc_auc_score(y_test, y_prob[:, 1])
                    else:
                        y_bin = label_binarize(y_test, classes=np.unique(y_test))
                        auc = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
                except:
                    auc = 0

            return {
                "Model": name,
                "Accuracy": round(acc, 4),
                "AUC": round(auc, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1, 4),
                "MCC": round(mcc, 4)
            }

        # -------------------------------
        # Run Models
        # -------------------------------
        if st.button("Run Models"):

            results = []

            # Logistic Regression
            lr = LogisticRegression(max_iter=5000)
            lr.fit(X_train_scaled, y_train)
            results.append(evaluate_model("Logistic Regression", lr, X_test_scaled, y_test))

            # Decision Tree
            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(X_train, y_train)
            results.append(evaluate_model("Decision Tree", dt, X_test, y_test))

            # KNN
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_scaled, y_train)
            results.append(evaluate_model("KNN", knn, X_test_scaled, y_test))

            # Naive Bayes
            nb = GaussianNB()
            nb.fit(X_train, y_train)
            results.append(evaluate_model("Naive Bayes", nb, X_test, y_test))

            # Random Forest
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)
            results.append(evaluate_model("Random Forest", rf, X_test, y_test))

            # Gradient Boosting
            gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
            gb.fit(X_train, y_train)
            results.append(evaluate_model("Gradient Boosting", gb, X_test, y_test))

            # Results table
            results_df = pd.DataFrame(results)

            st.write("## Final Model Comparison")
            sorted_df = results_df.sort_values(by="F1 Score", ascending=False)
            st.dataframe(sorted_df)

            # Best model
            best_model = sorted_df.iloc[0]
            st.success(f"Best Model: {best_model['Model']} with F1 Score = {best_model['F1 Score']}")

            

            # -------------------------------
            # Assignment Comparison Table
            # -------------------------------
            st.write("## ML Model Performance Comparison Table")

            comparison_data = {
                "ML Model Name": [
                    "Logistic Regression",
                    "Decision Tree",
                    "kNN",
                    "Naive Bayes",
                    "Random Forest (Ensemble)",
                    "XGBoost (Ensemble)"
                ],
                "Observation about model performance": [
                    "Performs well for linear relationships. Fast and interpretable but weak on complex data.",
                    "Captures non-linear patterns but prone to overfitting.",
                    "Works well when similar points are near. Sensitive to scaling.",
                    "Fast and good for high-dimensional data but assumes independence.",
                    "Robust ensemble model that reduces overfitting and improves accuracy.",
                    "Powerful boosting model with high accuracy but needs tuning."
                ]
            }

            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df)
