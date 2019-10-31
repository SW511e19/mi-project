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
blue = Path(dir_name) / "new_blue"
red = Path(dir_name) / "new_red"
green = Path(dir_name) / "new_green"
black = Path(dir_name) / "new_black"
white = Path(dir_name) / "new_white"
multi = Path(dir_name) / "new_multicolour"
oblue = Path(dir_name) / "old_blue"
ored = Path(dir_name) / "old_red"
ogreen = Path(dir_name) / "old_green"
oblack = Path(dir_name) / "old_black"
owhite = Path(dir_name) / "old_white"
omulti = Path(dir_name) / "old_multicolour"
colorless = Path(dir_name) / "colourless"
images = []
card_labels = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
]

categories = [blue, red, green, black, white, multi, oblue, ored, ogreen, oblack, owhite, omulti, colorless]
formatted_card_labels = keras.utils.to_categorical(card_labels, len(card_labels))


labels = []
count = 0
for x in categories:
    add_images_with_labels(categories[count], count)
    count += 1
print(len(labels))
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



