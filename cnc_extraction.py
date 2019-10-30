from pathlib import Path
import keras
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16

def add_images_with_labels(image_path, label_number):
    for img in image_path.glob("*.png"):
        img = image.load_img(img)
        image_array = image.img_to_array(img)

        images.append(image_array)
        labels.append(formatted_card_labels[label_number])
    return
dir_name = "test_creature_color"

# Path to folders with training data
card = Path(dir_name) / "cards"
nocard = Path(dir_name) / "not_cards"
images = []
card_labels = [
    0, 1
]

formatted_card_labels = keras.utils.to_categorical(card_labels, 2)
print(formatted_card_labels)


labels = []
add_images_with_labels(card, 0)
add_images_with_labels(nocard, 1)

# Create a single numpy array with all the images we loaded
x_train = np.array(images)

# Also convert the labels to a numpy array
y_train = np.array(labels)

# Normalize image data to 0-to-1 range
x_train = vgg16.preprocess_input(x_train)

# Load a pre-trained neural network to use as a feature extractor
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features for each image (all in one pass)
features_x = pretrained_nn.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "x_train.dat")

# Save the matching array of expected values to a file
joblib.dump(y_train, "y_train.dat")



