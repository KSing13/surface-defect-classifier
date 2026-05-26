import os
import tensorflow as tf
from tensorflow import keras
import config

os.makedirs(config.MODEL_DIR, exist_ok=True)

# Modern data loading using tf.data
train_ds = keras.utils.image_dataset_from_directory(
    config.DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=config.SEED,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    label_mode='categorical'
)

val_ds = keras.utils.image_dataset_from_directory(
    config.DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=config.SEED,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    label_mode='categorical'
)

# Dynamically determine the number of classes from the dataset
num_classes = len(train_ds.class_names)

# Data Augmentation layers integrated into the model
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

base_model = keras.applications.MobileNetV2(include_top=False, input_shape=(*config.IMG_SIZE, 3), weights='imagenet')
base_model.trainable = False

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(*config.IMG_SIZE, 3)),
    data_augmentation,
    keras.layers.Rescaling(1./127.5, offset=-1), # Preprocessing for MobileNetV2 [-1, 1]
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_path = os.path.join(config.MODEL_DIR, "surface_defect_model.keras")

my_callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'),
    keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
]

history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=config.EPOCHS,
    callbacks=my_callbacks
)

print("Model training complete and saved.")
