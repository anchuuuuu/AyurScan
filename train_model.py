import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# ==========================================
# 1. SETUP & PATHS
# ==========================================
# IMPORTANT: Point this to the local folder that contains your 80 sub-folders
dataset_path = os.path.join(os.path.dirname(__file__), "Indian Medicinal Leaves Image Datasets", "Medicinal Leaf dataset")
img_size = (224, 224)
batch_size = 32

# ==========================================
# 2. ADVANCED DATA AUGMENTATION
# ==========================================
# This makes the model "immune" to different lighting and backgrounds
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_generator.class_indices)
print(f"🌿 SUCCESS: Found {num_classes} classes!")

# Save class mapping (Crucial for correct predictions later)
with open("class_indices.json", "w") as f:
    json.dump({v: k for k, v in train_generator.class_indices.items()}, f)

# ==========================================
# 3. BUILD THE MODEL (MobileNetV2 + Custom Head)
# ==========================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False # Start with the "brain" frozen

from tensorflow.keras import regularizers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(), 
    layers.Dropout(0.5), # Prevents memorizing training photos
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)), # Added L2 regularization
    layers.Dense(num_classes, activation='softmax')
])

# ==========================================
# 4. STAGE 1: INITIAL TRAINING
# ==========================================
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks to handle learning rate and stopping
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("🔥 Stage 1: Training top layers...")
model.fit(train_generator, validation_data=val_generator, epochs=25, callbacks=[lr_scheduler, early_stop])

# ==========================================
# 5. STAGE 2: FINE-TUNING (The Specialist Phase)
# ==========================================
print("\n🔓 Unfreezing Deep Layers for 80-Class Precision...")
base_model.trainable = True

# We freeze the early layers and only train the last 50 layers
# This allows the model to deeply learn tiny textures of all 80 medicinal leaves
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5), # Tiny learning rate is REQUIRED
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🎯 Stage 2: Fine-Tuning...")
model.fit(train_generator, validation_data=val_generator, epochs=40, callbacks=[lr_scheduler, early_stop])

# ==========================================
# 6. EXPORT FINAL MODEL
# ==========================================
model.save("medicinal_leaf_80_classes_perfect.h5")
print("✅ MISSION COMPLETE: Model saved as 'medicinal_leaf_80_classes_perfect.h5'")