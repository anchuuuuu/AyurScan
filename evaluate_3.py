import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = os.path.join(os.path.dirname(__file__), "Indian Medicinal Leaves Image Datasets", "Medicinal Leaf dataset")

print("Loading model...")
model = tf.keras.models.load_model("medicinal_leaf_final_optimized.h5")

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    classes=['Aloevera', 'Neem', 'Tulsi'],
    shuffle=False
)

print("Evaluating...")
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc*100:.2f}%")
