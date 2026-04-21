import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# 1. SETUP & PATHS
# ==========================================
# Adjusted for local project structure
dataset_path = os.path.join(os.path.dirname(__file__), "Indian Medicinal Leaves Image Datasets", "Medicinal Leaf dataset")
img_size = (224, 224)
batch_size = 32

# ==========================================
# 2. DATA GENERATORS (Augmentation)
# ==========================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_generator.class_indices)
print(f"🌿 Found {num_classes} classes!")

# SAVE CLASS INDICES (Critical for app.py)
with open("class_indices.json", "w") as f:
    json.dump({v: k for k, v in train_generator.class_indices.items()}, f)
print("✅ class_indices.json updated.")

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Start with frozen base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(num_classes, activation='softmax')
])

# ==========================================
# 4. TRAINING STAGE 1: Classification Head
# ==========================================
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for optimization
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Starting Stage 1 training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[lr_reducer, early_stop]
)

# ==========================================
# 5. TRAINING STAGE 2: Fine-Tuning
# ==========================================
print("Unfreezing base model for fine-tuning...")
base_model.trainable = True

# We only unfreeze the last 50 layers for stability
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=[lr_reducer, early_stop]
)

# ==========================================
# 6. EXPORT
# ==========================================
# Saving as the name expected by app.py or a new name
model_save_path = "medicinal_leaf_80_classes_perfect.h5" 
model.save(model_save_path)
print(f"Training complete and model saved to {model_save_path}.")