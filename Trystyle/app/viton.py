import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2

# Check TensorFlow version

print(tf.__version__)
print (cv2.__version__)

try:
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
except:
    print("TensorFlow setup not working correctly.")

import os as os
# Directory paths
person_images_dir = '/kaggle/input/high-resolution-viton-zalando-dataset/test/agnostic-v3.2'
cloth_images_dir = '/kaggle/input/high-resolution-viton-zalando-dataset/test/cloth'
mask_images_dir = '/kaggle/input/high-resolution-viton-zalando-dataset/test/image-parse-v3'
output_images_dir= '/kaggle/input/high-resolution-viton-zalando-dataset/test/image'

# Load all images in the directory
person_images = []
cloth_images = []
mask_images = []
output_images = []

for person_filename, cloth_filename , mask_filename, output_filename in zip(sorted(os.listdir(person_images_dir)), sorted(os.listdir(cloth_images_dir)), sorted (os.listdir(mask_images_dir)), sorted(os.listdir(output_images_dir))):
    person_img_path = os.path.join(person_images_dir, person_filename)
    cloth_img_path = os.path.join(cloth_images_dir, cloth_filename)
    mask_img_path = os.path.join(mask_images_dir, mask_filename)
    output_img_path = os.path.join(output_images_dir, output_filename)
    
    # Load, resize, and normalize the images
    person_image = cv2.imread(person_img_path)
    person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
    person_image = cv2.resize(person_image, (128,128)) / 255.0
    
    cloth_image = cv2.imread(cloth_img_path)
    cloth_image = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2RGB)
    cloth_image = cv2.resize(cloth_image, (128,128)) / 255.0
    
    mask_image = cv2.imread(mask_img_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (128,128)) / 255.0
    mask_image = np.mean(mask_image, axis=-1, keepdims=True)
    
    output_image = cv2.imread(output_img_path)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    output_image = cv2.resize(output_image, (128,128)) / 255.0
    
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
print(f"Segmentation Image shape: {segmentation_image.shape}")
print(f"Output Image shape: {output_image.shape}")

# Display the loaded images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Person Image")
plt.imshow(segmentation_image)

plt.subplot(1, 2, 2)
plt.title("Cloth Image")
plt.imshow(cloth_image)

plt.show()

from tensorflow.keras import layers, models

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
    print(x_mask.shape)

    # Concatenate features from all branches
    concatenated = layers.concatenate([x_person, x_cloth, x_mask], axis=-1)
    print(concatenated.shape)

    # Decoder part with skip connections
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(concatenated)
    print(x.shape)
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
# Verify the model structure
print(f"Model has been built with {len(model_deepunet.layers)} layers.")

from tensorflow.keras import layers, models, losses, applications
import tensorflow as tf


# Using VGG19 for perceptual loss
vgg = applications.VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
def perceptual_loss(y_true, y_pred):
    vgg.trainable = False
    feature_extractor = models.Model(inputs=vgg.input, outputs=[vgg.get_layer('block5_conv4').output])

    y_true_features = feature_extractor(y_true)
    y_pred_features = feature_extractor(y_pred)

    return tf.reduce_mean(tf.square(y_true_features - y_pred_features))

def build_unet_virtual_tryon_model():
    inputs_person = layers.Input(shape=(128, 128, 3))
    inputs_cloth = layers.Input(shape=(128, 128, 3))
    inputs_mask = layers.Input(shape=(128, 128, 1))

    # Encoder for person image
    x_person_og = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs_person)
    x_person_skip = layers.MaxPooling2D((2, 2))(x_person_og)
    x_person = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x_person_skip)
    x_person = layers.MaxPooling2D((2, 2))(x_person)
    x_person = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x_person)
    x_person = layers.MaxPooling2D((2, 2))(x_person)

    # Encoder for cloth image
    x_cloth_og = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs_cloth)
    x_cloth_skip = layers.MaxPooling2D((2, 2))(x_cloth_og)
    x_cloth = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x_cloth_skip)
    x_cloth_2 = layers.MaxPooling2D((2, 2))(x_cloth)
    x_cloth = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x_cloth_2)
    x_cloth = layers.MaxPooling2D((2, 2))(x_cloth)

    # Encoder for segmentation mask
    x_mask_og = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs_mask)
    x_mask_skip = layers.MaxPooling2D((2, 2))(x_mask_og)
    x_mask = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x_mask_skip)
    x_mask = layers.MaxPooling2D((2, 2))(x_mask)
    x_mask = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x_mask)
    x_mask = layers.MaxPooling2D((2, 2))(x_mask)

    # Concatenate features from all branches
    concatenated = layers.concatenate([x_person, x_cloth, x_mask], axis=-1)

    # Decoder part with skip connections
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(concatenated)
    x = layers.concatenate([x, x_cloth_2], axis=-1)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    print(x.shape)
    x = layers.concatenate([x, x_cloth_skip], axis=-1)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, x_person_og], axis=-1)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Output layer
    output_image = layers.Conv2D(3, (1, 1), activation='sigmoid')(x)

    # Build and compile the model
    model = models.Model(inputs=[inputs_person, inputs_cloth, inputs_mask], outputs=output_image)
    model.compile(optimizer='adam', loss=perceptual_loss)  # Use perceptual loss

    return model

model_deepunet = build_unet_virtual_tryon_model()
model_deepunet.summary()

X_person = np.array(person_images)
X_cloth = np.array(cloth_images)
Y_output = np.array(output_images)

# Now the shape of segmentation_image will be (256, 256, 1)
X_segmentation = np.array(mask_images)

model_deepunet.fit([X_person, X_cloth, X_segmentation], Y_output, epochs=10)
model_deepunet.save('deepunet_model.h5')  # Save the model in HDF5 format

# Test the model on the same input
predicted_image = model_deepunet.predict([X_person, X_cloth, X_segmentation])

np.save('predicted_image.npy', predicted_image)

# Display the original and predicted images
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Cloth Image")
plt.imshow(X_person[0])

plt.subplot(1, 3, 2)
plt.title("Output Image")
plt.imshow(Y_output[0])

plt.subplot(1, 3, 3)
plt.title("Predicted Try-On Image")
plt.imshow(predicted_image[0])

plt.show()

# from tensorflow.keras.losses import MeanSquaredError

# mse = MeanSquaredError()
# mse_value = mse(Y_output_test, predictions).numpy()
# print(f'MSE: {mse_value}')

def psnr(target, prediction):
    mse = np.mean((target - prediction) ** 2)
    return 10 * np.log10(1.0 / mse)

psnr = psnr(Y_output, predicted_image)
print(psnr)

X_person = np.array(person_images)
X_cloth = np.array(cloth_images)
Y_output = np.array(output_images)

# Now the shape of segmentation_image will be (256, 256, 1)
X_segmentation = np.array(mask_images)

model_unet.fit([X_person, X_cloth, X_segmentation], Y_output, epochs=15)

model_unet.save('unet_model.h5')  # Save the model in HDF5 format

# Verify that the model can run one epoch of training
test_loss = model_unet.evaluate([X_person, X_cloth, X_segmentation], Y_output)
print(f"Test Loss after one epoch: {test_loss}")

# Test the model on the same input
predicted_image = model_unet.predict([X_person, X_cloth, X_segmentation])

np.save('predicted_image.npy', predicted_image)

# Display the original and predicted images
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Cloth Image")
plt.imshow(X_person[0])

plt.subplot(1, 3, 2)
plt.title("Output Image")
plt.imshow(Y_output[0])

plt.subplot(1, 3, 3)
plt.title("Predicted Try-On Image")
plt.imshow(predicted_image[0])

plt.show()

# from tensorflow.keras.losses import MeanSquaredError

# mse = MeanSquaredError()
# mse_value = mse(Y_output_test, predictions).numpy()
# print(f'MSE: {mse_value}')

def psnr(target, prediction):
    mse = np.mean((target - prediction) ** 2)
    return 10 * np.log10(1.0 / mse)

psnr = psnr(Y_output, predicted_image)
print(psnr)

def select_new_image(image_dir, image_type):
    images = sorted(os.listdir(image_dir))
    print(f"Available {image_type} images:")
    for idx, img in enumerate(images[:10]):  # Show first 10 images as an example
        print(f"{idx}: {img}")
    
    selected_idx = int(input(f"Enter the index of the {image_type} image to use: "))
    selected_image_path = os.path.join(image_dir, images[selected_idx])
    
    return selected_image_path

# Get new images
new_person_image_path = select_new_image(person_images_dir, "person")
new_cloth_image_path = select_new_image(cloth_images_dir, "cloth")

# Load the selected images
def load_image(image_path, size=(128, 128), grayscale=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    if grayscale:
        img = np.mean(img, axis=-1, keepdims=True) / 255.0
    else:
        img = img / 255.0
    return img

new_person_image = load_image(new_person_image_path)
new_cloth_image = load_image(new_cloth_image_path)

# Predict with the new user-selected inputs
new_prediction = model_reduced.predict([np.expand_dims(new_person_image, axis=0),
                                        np.expand_dims(new_cloth_image, axis=0),
                                        np.expand_dims(mask_images[0], axis=0)])

# Display Results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Selected Person Image")
plt.imshow(new_person_image)

plt.subplot(1, 3, 2)
plt.title("Selected Cloth Image")
plt.imshow(new_cloth_image)

plt.subplot(1, 3, 3)
plt.title("Predicted Try-On Image")
plt.imshow(new_prediction[0])

plt.show()

