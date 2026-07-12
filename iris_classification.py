"""
TASK 1: Iris Flower Classification
CodeAlpha Data Science Internship

Goal:
- Use measurements of Iris flowers (setosa, versicolor, virginica) as input data.
- Train a machine learning model to classify species based on measurements.
- Use Scikit-learn for dataset access and model building.
- Evaluate accuracy and performance using test data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# -----------------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))

print("First 5 rows of the dataset:")
print(df.head(), "\n")

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["species_name"].value_counts(), "\n")

print("Basic statistics:")
print(df.describe(), "\n")

# -----------------------------------------------------------
# 2. EXPLORATORY DATA VISUALIZATION
# -----------------------------------------------------------
sns.set_style("whitegrid")

# Pairplot to see feature relationships between species
pair = sns.pairplot(df, hue="species_name", vars=iris.feature_names, palette="Set2")
pair.fig.suptitle("Iris Feature Relationships by Species", y=1.02)
pair.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.close("all")

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("iris_correlation_heatmap.png", dpi=150)
plt.close()

# -----------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------------------------------------
X = df[iris.feature_names]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------
# 4. TRAIN MULTIPLE MODELS & COMPARE
# -----------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear"),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

results = {}
best_model_name = None
best_acc = 0
best_model = None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: Accuracy = {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model = model

print(f"\nBest Model: {best_model_name} with Accuracy = {best_acc:.4f}\n")

# -----------------------------------------------------------
# 5. DETAILED EVALUATION OF BEST MODEL
# -----------------------------------------------------------
best_preds = best_model.predict(X_test_scaled)

print("Classification Report (Best Model):")
print(classification_report(y_test, best_preds, target_names=iris.target_names))

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=150)
plt.close()

# -----------------------------------------------------------
# 6. MODEL COMPARISON CHART
# -----------------------------------------------------------
plt.figure(figsize=(8, 5))
bars = plt.bar(results.keys(), results.values(), color=sns.color_palette("Set2"))
plt.ylabel("Accuracy")
plt.title("Model Comparison on Iris Test Set")
plt.ylim(0.8, 1.02)
plt.xticks(rotation=20)
for bar, acc in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 0.005, f"{acc:.3f}",
              ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("iris_model_comparison.png", dpi=150)
plt.close()

# -----------------------------------------------------------
# 7. SAVE PROCESSED DATASET (for reference)
# -----------------------------------------------------------
df.to_csv("iris_dataset.csv", index=False)

print("Done. Plots saved: iris_pairplot.png, iris_correlation_heatmap.png,")
print("iris_confusion_matrix.png, iris_model_comparison.png")
