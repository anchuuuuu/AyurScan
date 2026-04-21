import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

dataset_path = os.path.join(os.path.dirname(__file__), "Indian Medicinal Leaves Image Datasets", "Medicinal Leaf dataset")

print("Loading model...")
model = tf.keras.models.load_model("medicinal_leaf_80_classes_perfect.h5")

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Evaluating...")
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc*100:.2f}%")
