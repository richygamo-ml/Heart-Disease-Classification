import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

st.title("Heart Disease Prediction Dashboard")

st.write("""
This application predicts the likelihood of heart disease using multiple machine learning models.
The models are trained on the Cleveland Heart Disease dataset.
""")

# Load dataset
data = pd.read_csv("Heart_disease_cleveland_new.csv")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Patient input section
st.header("Patient Information")

age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
cp = st.slider("Chest Pain Type", 0, 3, 1)
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
restecg = st.slider("Resting ECG", 0, 2, 1)
thalach = st.slider("Max Heart Rate", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
slope = st.slider("Slope of ST Segment", 0, 2, 1)
ca = st.slider("Number of Major Vessels", 0, 4, 0)
thal = st.slider("Thalassemia", 0, 3, 1)

features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

# Prediction
if st.button("Predict Heart Disease Risk"):

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Train models only once using @st.cache_resource
@st.cache_resource
def train_models(X_train, y_train):

    log_model = LogisticRegression(max_iter=1000)  # Logistic Regression
    tree_model = DecisionTreeClassifier()          # Decision Tree
    rf_model = RandomForestClassifier()            # Random Forest
    gb_model = GradientBoostingClassifier()        # Gradient Boosting
    nn_model = MLPClassifier(max_iter=1000)        # Neural Network

    log_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    nn_model.fit(X_train, y_train)

    return {
        "Logistic Regression": log_model,
        "Decision Tree": tree_model,
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "Neural Network": nn_model
    }

models = train_models(X_train, y_train)

# Model selection
model_choice = st.selectbox(
    "Select Machine Learning Model",
    list(models.keys())       # To get a view of all the keys in the dictionary (models)
)

model = models[model_choice]    # Select the model from "models" and train it on the TRAIN set

# Prediction
y_predicted = model.predict(X_test)

# Model performance
accuracy = accuracy_score(y_test, y_predicted)
st.header("Model Performance")
st.metric("Test Accuracy", f"{accuracy:.3f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_predicted)

# Visualization
fig, ax = plt.subplots()
ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted Target")
ax.set_ylabel("Actual Target")

st.pyplot(fig)

# Model comparison chart with accuracies of all models
st.subheader("Model Comparison")
accuracies = []

for name, m in models.items():    # "models" is a dictionary. Name is the string key (e.g. Logistic Reg); m is the actual model object (e.g.                                                                    LogisticRegression())
    
    acc = accuracy_score(y_test, m.predict(X_test))  # e.g. LogisticRegression().predict(X_test)
    accuracies.append(acc)

comparison_df = pd.DataFrame({
    "Model": list(models.keys()),
    "Accuracy": accuracies
})

# Visualization of accuracies
fig2, ax2 = plt.subplots()
ax2.bar(comparison_df["Model"], comparison_df["Accuracy"])
ax2.set_ylabel("Accuracy")
ax2.set_xticklabels(comparison_df["Model"], rotation=45)

st.pyplot(fig2)

# ROC curve : shows the trade-off between True Positive Rate (TPR = How many SICK patients did we catch?)
# and False Positive Rate (FPR = How many patients did we scare?); 0 = no heart disease, 1 = heart disease.

st.subheader("ROC Curve Comparison")

fig3, ax3 = plt.subplots()

for name, m in models.items():

    y_probability = m.predict_proba(X_test)[:,1]    # e.g. LogisticRegression().predict_proba(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_probability)
    roc_auc = auc(fpr, tpr)

    ax3.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

ax3.plot([0,1],[0,1],'k--')

ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curve Comparison")
ax3.legend()

st.pyplot(fig3)

# AUC (Area under the curve) measures the overall classifier performance: 0.5 = random guess; 0.7-0.8 = Fair; 0.8-0.9 = Good; > 0.9 = Excellent.
# A higher AUC means the model better distinguishes between positive and negative classes.