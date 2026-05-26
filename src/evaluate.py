import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
import config

# Must use the same seed as train.py to ensure the same split
val_ds = keras.utils.image_dataset_from_directory(
    config.DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=config.SEED,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

model_path = os.path.join(config.MODEL_DIR, "surface_defect_model.keras")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"No model found at {model_path}. Please run train.py first.")

model = keras.models.load_model(model_path)
print(f"Loaded model from {model_path}")

# Efficiently extract true labels from the non-shuffled dataset
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

# Predict the entire dataset at once (much faster)
preds = model.predict(val_ds)
y_pred = np.argmax(preds, axis=1)

print(classification_report(y_true, y_pred, labels=np.arange(len(val_ds.class_names)), target_names=val_ds.class_names))
