import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np

# Suppress convergence warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Load data (assuming "news.csv" is in the working directory)
df = pd.read_csv("news.csv", dtype=str)

# Filter out only 'FAKE' and 'REAL' labels
df = df[df['label'].isin(['FAKE', 'REAL'])]

# Drop rows with NaN values in specified columns
columns_to_check = ['title', 'text', 'label']
df.dropna(subset=columns_to_check, inplace=True)

# Check class distribution
print(df['label'].value_counts())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=20, stratify=df["label"])

# Feature Engineering - TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tf_train = vectorizer.fit_transform(X_train)
tf_test = vectorizer.transform(X_test)

# Hyperparameter Tuning for SVM using RandomizedSearchCV
parameters = {'C': [0.1, 1, 10, 100], 'max_iter': [1000, 5000, 10000]}
svm_model = LinearSVC(dual=False)
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)
clf = RandomizedSearchCV(svm_model, parameters, cv=stratified_kfold, scoring='accuracy', n_iter=10, random_state=20)
clf.fit(tf_train, y_train)

# Best model from RandomizedSearch
best_svm_model = clf.best_estimator_

# Prediction and Evaluation
y_pred = best_svm_model.predict(tf_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(accuracy * 100, 2)}%")


# Save Model and Vectorizer
with open('finalized_model_svm.pkl', 'wb') as model_file:
    pickle.dump(best_svm_model, model_file)
with open('vectorizer_svm.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and Vectorizer saved successfully!")