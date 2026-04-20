import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Load model
model = tf.keras.models.load_model("medicinal_leaf_final_optimized.h5")

# Setup datagen
dataset_path = os.path.join(os.path.dirname(__file__), "Indian Medicinal Leaves Image Datasets", "Medicinal Leaf dataset")
img_size = (224, 224)
batch_size = 32
target_classes = ['Aloevera', 'Neem', 'Tulsi']

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = val_datagen.flow_from_directory(
    dataset_path, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='validation', classes=target_classes,
    shuffle=False # IMPORTANT: keep order for evaluation
)

# Evaluate
print("Evaluating on Validation Set...")
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc*100:.2f}%")

# Predict
predictions = model.predict(val_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes

print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=target_classes))
