from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
import os

DATA_DIR = "data"
MODEL_DIR = "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_ds = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

model_path = os.path.join(MODEL_DIR, "surface_defect_model.h5")
model = load_model(model_path)
print(f"Loaded model from {model_path}")

val_ds.reset()
preds = model.predict(val_ds)
pred_classes = np.argmax(preds, axis=1)
true_classes = val_ds.classes

class_labels = list(val_ds.class_indices.keys())
print(classification_report(true_classes, pred_classes, target_names=class_labels))
