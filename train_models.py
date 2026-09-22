import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load feature dataset
csv_path = '/home/nyx/cwru_dataset/cwru_features.csv'
df = pd.read_csv(csv_path)

# Separate features (X) and target label (y)
X = df.drop(columns=['label'])
y_raw = df['label']

# Encode target strings into integer numbers (0, 1, 2, 3)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
class_names = label_encoder.classes_

# Train / Test split (80% training, 20% testing with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling (Crucial for SVM & KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Dataset Loaded: {len(df)} total samples across {len(class_names)} classes.")
print(f"Training set size: {len(X_train)} samples | Test set size: {len(X_test)} samples\n")

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF Kernel)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate models
results = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    # Fit model (scale for SVM/KNN, raw or scaled work for RF)
    if name == 'Random Forest':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"==================== {name} ====================")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
    axes[idx].set_title(f"{name}\nAccuracy: {acc*100:.1f}%", fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Predicted Label")
    axes[idx].set_ylabel("True Label")

plt.tight_layout()
output_cm_path = '/home/nyx/cwru_dataset/confusion_matrices.png'
plt.savefig(output_cm_path, dpi=300)
print(f"\nConfusion matrices saved to: {output_cm_path}")
