import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, './efficientnetv2') 

# import torch
import tensorflow as tf
# import tensorflow_hub
# import keras
import effnetv2_model

def effnet_model():
    num_classes = 8

# img_augmentation = Sequential(
#     [
#         layers.RandomRotation(factor=0.15),
#         layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
#         layers.RandomFlip(),
#         layers.RandomContrast(factor=0.1),
#     ],
#     name="img_augmentation",
# )

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
        # tf.keras.layers.RandomRotation(factor=0.15),
        # tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        # tf.keras.layers.RandomFlip(),
        # tf.keras.layers.RandomContrast(factor=0.1),
        effnetv2_model.get_model('efficientnetv2-m', include_top=False, weights='imagenet'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.summary()

    return model


