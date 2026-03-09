from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Train all models
log_model = LogisticRegression(max_iter=1000)  # Logistic Regression
tree_model = DecisionTreeClassifier()          # Decision Tree
rf_model = RandomForestClassifier()            # Random Forest
gb_model = GradientBoostingClassifier()        # Neural Network
nn_model = MLPClassifier(max_iter=1000)

# Map input data (X) to target output (y)
log_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)

# User chooses the model to train
model_choice = st.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "Neural Network"
    ]
)

# Use selected model
models = {
    "Logistic Regression": log_model,
    "Decision Tree": tree_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Neural Network": nn_model
}

model = models[model_choice]

# Model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
st.metric("Model Accuracy", f"{accuracy:.3f}")

# User enters patient data
prediction = model.predict(features)