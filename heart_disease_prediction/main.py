"""
Heart Disease Prediction Using Machine Learning

Step 1: Load the dataset
Step 2: Assign meaningful column names
Step 3: Check for missing values
"""

import pandas as pd


# ============================================
# STEP 1: LOAD THE DATASET
# ============================================

DATA_PATH = "data/processed.cleveland.data"

data = pd.read_csv(
    DATA_PATH,
    header=None,
    na_values="?"
)

print("Dataset loaded successfully!")


# ============================================
# STEP 2: ASSIGN COLUMN NAMES
# ============================================

data.columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]


# ============================================
# STEP 3: CHECK FOR MISSING VALUES
# ============================================

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nMissing values in each column:")
print(data.isnull().sum())
# ============================================
# STEP 4: HANDLE MISSING VALUES
# ============================================

# Remove rows containing missing values
data = data.dropna()

print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nDataset shape after cleaning:")
print(data.shape)
# ============================================
# STEP 5: SEPARATE FEATURES AND TARGET
# ============================================

X = data.drop("target", axis=1)
y = data["target"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())
# ============================================
# STEP 6: CONVERT TARGET INTO BINARY VALUES
# ============================================

data["target"] = data["target"].apply(
    lambda x: 1 if x > 0 else 0
)

X = data.drop("target", axis=1)
y = data["target"]

print("\nTarget values after conversion:")
print(y.value_counts())
# ============================================
# STEP 7: SPLIT DATA INTO TRAINING AND TESTING
# ============================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)
# ============================================
# STEP 8: CREATE THE MACHINE LEARNING MODEL
# ============================================

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

print("\nLogistic Regression model created successfully!")
# ============================================
# STEP 9: TRAIN THE MODEL
# ============================================

model.fit(X_train, y_train)

print("\nModel trained successfully!")
model.fit(X_train, y_train)

print("\nModel trained successfully!")
print("STEP 9 COMPLETE!")
# ============================================
# STEP 10: MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)
# ============================================
# STEP 11: EVALUATE MODEL ACCURACY
# ============================================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)
# ============================================
# STEP 12: CONFUSION MATRIX
# ============================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
# ============================================
# STEP 13: CLASSIFICATION REPORT
# ============================================

from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(report)
# ============================================
# STEP 14: VISUALIZE CONFUSION MATRIX
# ============================================

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
)

disp.plot()

plt.title("Heart Disease Prediction - Confusion Matrix")
plt.show()
# ============================================
# STEP 16: FEATURE SCALING AND PIPELINE
# ============================================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(max_iter=1000))
])

# Train the pipeline
pipeline_model.fit(X_train, y_train)

# Make predictions
pipeline_pred = pipeline_model.predict(X_test)

# Calculate accuracy
pipeline_accuracy = accuracy_score(y_test, pipeline_pred)

print("\nScaled Pipeline Accuracy:")
print(pipeline_accuracy)



# ============================================
# STEP 15: ROC CURVE AND AUC
# ============================================

from sklearn.metrics import roc_curve, roc_auc_score

# Get probability predictions for the positive class
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate AUC
auc = roc_auc_score(y_test, y_prob)

print('AUC Score:')
print(auc)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Heart Disease Prediction')
plt.legend()
plt.show()

# ============================================
# STEP 20: VERIFY SAVED MODEL
# ============================================

predictions_match = (pipeline_pred == loaded_predictions).all()

print("\nDo original and saved-model predictions match?")
print(predictions_match)
# ============================================
# STEP 21: CREATE A SINGLE-PATIENT PREDICTION FUNCTION
# ============================================

def predict_heart_disease(patient_data):

    prediction = loaded_model.predict(patient_data)

    if prediction[0] == 1:
        return "Heart disease present"
    else:
        return "No heart disease"


print("\nSingle-patient prediction function created successfully!")
# ============================================
# STEP 22: TEST SINGLE-PATIENT PREDICTION
# ============================================

sample_patient = X_test.iloc[[0]]

result = predict_heart_disease(sample_patient)

print("\nSample patient prediction:")
print(result)
# ============================================
# STEP 23: DISPLAY SAMPLE PATIENT DATA
# ============================================

print("\nSample patient data:")
print(sample_patient)

print("\nPrediction:")
print(result)
# ============================================
# STEP 24: PREDICTION PROBABILITY
# ============================================

probabilities = loaded_model.predict_proba(sample_patient)

print("\nPrediction probabilities:")
print("No heart disease:", probabilities[0][0])
print("Heart disease:", probabilities[0][1])
# ============================================
# STEP 25: FORMAT PREDICTION PROBABILITIES
# ============================================

no_disease_probability = probabilities[0][0] * 100
disease_probability = probabilities[0][1] * 100

print("\nPrediction probabilities:")
print(f"No heart disease: {no_disease_probability:.2f}%")
print(f"Heart disease: {disease_probability:.2f}%")