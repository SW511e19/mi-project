from pathlib import Path
import keras
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16

def add_images_with_labels(image_path, label_number):
    for img in image_path.glob("*.jpg"):
        img = image.load_img(img)
        image_array = image.img_to_array(img)

        images.append(image_array)
        labels.append(formatted_card_labels[label_number])
    return
dir_name = "real_training_cards"

# Path to folders with training data
blue = Path(dir_name) / "blue"
red = Path(dir_name) / "red"
green = Path(dir_name) / "green"
black = Path(dir_name) / "black"
white = Path(dir_name) / "white"
yellow = Path(dir_name) / "yellow"
oblue = Path(dir_name) / "oblue"
ored = Path(dir_name) / "ored"
ogreen = Path(dir_name) / "ogreen"
oblack = Path(dir_name) / "oblack"
owhite = Path(dir_name) / "owhite"
colorless = Path(dir_name) / "colorless"
images = []
card_labels = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
]

formatted_card_labels = keras.utils.to_categorical(card_labels, 12)
print(formatted_card_labels)


labels = []
add_images_with_labels(blue, 0)
add_images_with_labels(red, 1)
add_images_with_labels(green, 2)
add_images_with_labels(black, 3)
add_images_with_labels(white, 4)
add_images_with_labels(yellow, 5)
add_images_with_labels(oblue, 6)
add_images_with_labels(ored, 7)
add_images_with_labels(ogreen, 8)
add_images_with_labels(oblack, 9)
add_images_with_labels(owhite, 10)
add_images_with_labels(colorless, 11)

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



