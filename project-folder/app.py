import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
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
uploaded_file = st.file_uploader("mixed_desc.csv", type=["csv"])

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
        X = X.fillna(X.mean())
        X = X.clip(lower=-1e6, upper=1e6)

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Remove rare classes
        counts = Counter(y)
        valid_classes = [c for c, cnt in counts.items() if cnt > 1]

        mask = [label in valid_classes for label in y]
        X = X[mask]
        y = y[mask]

        # Check class count
        if len(np.unique(y)) < 2:
            st.error("Not enough class diversity after filtering. Need at least 2 classes.")
            st.stop()

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scaling for distance-based models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # -------------------------------
        # Evaluation Function
        # -------------------------------
        def evaluate_model(name, model, X_test_input, y_test):

            y_pred = model.predict(X_test_input)

            # Some models may not support predict_proba
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
                test_classes = np.unique(y_test)
                class_indices = [list(model.classes_).index(c) for c in test_classes]
                y_prob_filtered = y_prob[:, class_indices]

                if len(test_classes) == 2:
                    auc = roc_auc_score(y_test, y_prob_filtered[:, 1])
                else:
                    y_test_bin = label_binarize(y_test, classes=test_classes)
                    auc = roc_auc_score(y_test_bin, y_prob_filtered, multi_class='ovr', average='macro')

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
            gb = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1
            )
            gb.fit(X_train, y_train)
            results.append(evaluate_model("Gradient Boosting", gb, X_test, y_test))

            # Create results dataframe
            results_df = pd.DataFrame(results)

            # Display results
            st.write("##Final Model Comparison")
            sorted_df = results_df.sort_values(by="F1 Score", ascending=False)
            st.dataframe(sorted_df)

            # Best Model Highlight
            best_model = sorted_df.iloc[0]
            st.success(f"Best Model: {best_model['Model']} with F1 Score = {best_model['F1 Score']}")
