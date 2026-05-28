import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import config

# Must use the same seed as train.py to ensure the same split
val_ds = tf.keras.utils.image_dataset_from_directory(
    config.DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=config.SEED,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    label_mode='categorical',
    shuffle=True
)

model_path = os.path.join(config.MODEL_DIR, "surface_defect_model.keras")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"No model found at {model_path}. Please run train.py first.")

model = tf.keras.models.load_model(model_path)
print(f"Loaded model from {model_path}")

# Extract labels and predictions in sync
print("Evaluating model...")
all_y_true = []
all_y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    all_y_true.append(np.argmax(labels.numpy(), axis=1))
    all_y_pred.append(np.argmax(preds, axis=1))

y_true = np.concatenate(all_y_true)
y_pred = np.concatenate(all_y_pred)

# Explicitly define the labels to match the full set of class names
class_indices = np.arange(len(val_ds.class_names))
report = classification_report(y_true, y_pred, labels=class_indices, target_names=val_ds.class_names, output_dict=True)

# Extract accuracy and clean up the dictionary for a cleaner table
accuracy = report.pop('accuracy')
df_report = pd.DataFrame(report).transpose()

print("\n" + "="*25 + " CLASSIFICATION REPORT " + "="*25)
print(f"Overall Accuracy: {accuracy:.4f}\n")
print(df_report.iloc[:-2, :].round(3).to_string()) # Detailed metrics for classes
print("-" * 73)
print(df_report.iloc[-2:, :].round(3).to_string()) # Averages (macro/weighted)

cm = confusion_matrix(y_true, y_pred, labels=class_indices)
df_cm = pd.DataFrame(cm, index=val_ds.class_names, columns=val_ds.class_names)

print("\n" + "="*25 + " CONFUSION MATRIX " + "="*27)
print(df_cm.to_string())
