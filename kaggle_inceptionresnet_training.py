import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. Model Creation Function
def create_model(freeze=True):
    # Load InceptionResNetV2 with ImageNet weights, excluding the top layer
    base_model = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3))
    
    # Optional freezing of the base layers
    if freeze:
        base_model.trainable = False
        
    model = tf.keras.Sequential([
        tf.keras.layers.Input((224, 224, 3)),
        # Note: 'resize_and_rescale' and 'data_augmentation' would be custom Keras layers defined earlier in the notebook
        # resize_and_rescale,   
        # data_augmentation,    
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(80, activation='softmax') # 80 classes of medicinal leaves
    ])
    
    return model

# 2. Learning Rate Schedule & Callbacks
boundaries = [20, 35]
values = [0.0001, 0.0005, 0.001]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

checkpoint = ModelCheckpoint(
    filepath='/kaggle/working/mod_{epoch}.keras', # Will need to be changed for local saving
    save_best_only=True,
    monitor='val_accuracy',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.05,
    patience=50,
    baseline=0.5,
    restore_best_weights=True
)

# 3. Model Initialization & Compilation
# Here freeze is set to False, meaning they are fine-tuning the ENTIRE InceptionResNetV2 model
model = create_model(freeze=False)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# 4. Training
# Note: 'train_data' and 'valid_data' need to be defined (e.g., via image_dataset_from_directory)
'''
history = model.fit(train_data,
                    epochs=50,
                    validation_data=valid_data,
                    callbacks=[checkpoint, early_stopping])
'''
