# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 13:56:31 2024

@author: hardik polamarasetti
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


path = r"C:\Users\hardi\OneDrive\Documents\CSU\DSA460\DSA460FinalProject\heart.csv"
heart_data = pd.read_csv(path)


categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Agebefore preprocessing
plt.figure(figsize=(12, 6))
sns.histplot(heart_data['age'], kde=True, color='blue')
plt.title('Distribution of Age Before Scaling')
plt.show()

#Preprocessing
transformers = [
    ('num', StandardScaler(), continuous_cols),
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
]
preprocessor = ColumnTransformer(transformers, remainder='passthrough')
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', None)
])

# splitting data
X = heart_data.drop('target', axis=1)
y = heart_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    pipeline.set_params(classifier=model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results[name] = {
        'Test Accuracy': accuracy_score(y_test, y_pred),
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred),
        '10-Fold CV Scores': cross_val_score(pipeline, X, y, cv=10)
    }

#results
for model, metrics in results.items():
    print(f"Model: {model}")
    print("Test Accuracy:", metrics['Test Accuracy'])
    print("Confusion Matrix:\n", metrics['Confusion Matrix'])
    print("Classification Report:\n", metrics['Classification Report'])
    print("10-Fold CV Scores:", metrics['10-Fold CV Scores'])
    print("Mean CV Score:", np.mean(metrics['10-Fold CV Scores']))
    print("\n" + "-"*50 + "\n")


X_preprocessed = preprocessor.transform(X)
column_names = continuous_cols + [f"{col}_{val}" for col in categorical_cols for val in heart_data[col].unique()[1:]]
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=column_names)

# Age data after preprocessing
plt.figure(figsize=(12, 6))
sns.histplot(X_preprocessed_df['age'], kde=True, color='green')
plt.title('Distribution of Age After Scaling')
plt.show()