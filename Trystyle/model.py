import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import cv2 as cv2

# Check TensorFlow version
print(tf.__version__)
print(cv2.__version__)

try:
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
except:
    print("TensorFlow setup not working correctly.")

import os
# Directory paths
person_images_dir = '/kaggle/input/high-resolution-viton-zalando-dataset/test/agnostic-v3.2'
cloth_images_dir = '/kaggle/input/high-resolution-viton-zalando-dataset/test/cloth'
mask_images_dir = '/kaggle/input/high-resolution-viton-zalando-dataset/test/image-parse-v3'
output_images_dir = '/kaggle/input/high-resolution-viton-zalando-dataset/test/image'

# Load all images in the directory
person_images = []
cloth_images = []
mask_images = []
output_images = []

for person_filename, cloth_filename, mask_filename, output_filename in zip(sorted(os.listdir(person_images_dir)), sorted(os.listdir(cloth_images_dir)), sorted(os.listdir(mask_images_dir)), sorted(os.listdir(output_images_dir))):
    person_img_path = os.path.join(person_images_dir, person_filename)
    cloth_img_path = os.path.join(cloth_images_dir, cloth_filename)
    mask_img_path = os.path.join(mask_images_dir, mask_filename)
    output_img_path = os.path.join(output_images_dir, output_filename)
    
    # Load, resize, and normalize the images
    person_image = cv2.imread(person_img_path)
    person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
    person_image = cv2.resize(person_image, (128, 128)) / 255.0
    
    cloth_image = cv2.imread(cloth_img_path)
    cloth_image = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2RGB)
    cloth_image = cv2.resize(cloth_image, (128, 128)) / 255.0
    
    mask_image = cv2.imread(mask_img_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (128, 128)) / 255.0
    mask_image = np.mean(mask_image, axis=-1, keepdims=True)
    
    output_image = cv2.imread(output_img_path)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    output_image = cv2.resize(output_image, (128, 128)) / 255.0
    
    person_images.append(person_image)
    cloth_images.append(cloth_image)
    mask_images.append(mask_image)
    output_images.append(output_image)

# Convert to numpy arrays
person_images = np.array(person_images)
cloth_images = np.array(cloth_images)
mask_images = np.array(mask_images)
output_images = np.array(output_images)

# Print the shape of the arrays to verify
print(f"Loaded {person_images.shape[0]} person images.")
print(f"Loaded {cloth_images.shape[0]} cloth images.")
print(f"Loaded {mask_images.shape[0]} mask images.")
print(f"Loaded {output_images.shape[0]} output images.")

# Verify that the images are loaded correctly
print(f"Person Image shape: {person_image.shape}")
print(f"Cloth Image shape: {cloth_image.shape}")
print(f"Mask Image shape: {mask_image.shape}")
print(f"Output Image shape: {output_image.shape}")

# Display the loaded images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Person Image")
plt.imshow(person_image)

plt.subplot(1, 2, 2)
plt.title("Cloth Image")
plt.imshow(cloth_image)

plt.show()

# Build U-Net model for virtual try-on
def build_unet_virtual_tryon_model():
    inputs_person = layers.Input(shape=(128, 128, 3))
    inputs_cloth = layers.Input(shape=(128, 128, 3))
    inputs_mask = layers.Input(shape=(128, 128, 1))

    # Encoder for person image
    x_person_og = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs_person)
    x_person_skip = layers.MaxPooling2D((2, 2))(x_person_og)
    x_person = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x_person_skip)
    x_person = layers.MaxPooling2D((2, 2))(x_person)

    # Encoder for cloth image
    x_cloth_og = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs_cloth)
    x_cloth_skip = layers.MaxPooling2D((2, 2))(x_cloth_og)
    x_cloth = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x_cloth_skip)
    x_cloth = layers.MaxPooling2D((2, 2))(x_cloth)

    # Encoder for segmentation mask
    x_mask_og = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs_mask)
    x_mask_skip = layers.MaxPooling2D((2, 2))(x_mask_og)
    x_mask = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x_mask_skip)
    x_mask = layers.MaxPooling2D((2, 2))(x_mask)

    # Concatenate features from all branches
    concatenated = layers.concatenate([x_person, x_cloth, x_mask], axis=-1)

    # Decoder part with skip connections
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(concatenated)
    x = layers.concatenate([x, x_cloth_skip], axis=-1)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, x_person_og], axis=-1)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Output layer
    output_image = layers.Conv2D(3, (1, 1), activation='sigmoid')(x)

    # Build and compile the model
    model = models.Model(inputs=[inputs_person, inputs_cloth, inputs_mask], outputs=output_image)
    model.compile(optimizer='adam', loss='mse')

    return model

model_unet = build_unet_virtual_tryon_model()
model_unet.summary()

# Save the model as a pickle file
import pickle

with open('unet_virtual_tryon_model.pkl', 'wb') as f:
    pickle.dump(model_unet, f)
