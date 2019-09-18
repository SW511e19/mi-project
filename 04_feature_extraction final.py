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
dir_name = "training_data"

# Path to folders with training data
not_cards = Path(dir_name) / "Not_card"
#angel = Path(dir_name) / "Angel"
demon = Path(dir_name) / "Demon"
#goblin = Path(dir_name) / "Goblin"
human = Path(dir_name) / "Human"
#merfolk = Path(dir_name) / "Merfolk"
spirit = Path(dir_name) / "Spirit"
#zombie = Path(dir_name) / "Zombie"

images = []
card_labels = [
    0, 1, 2, 3,
]

formatted_card_labels = keras.utils.to_categorical(card_labels, 4)
print(formatted_card_labels)


labels = []
# Load all the not-card images

add_images_with_labels(not_cards, 0)
add_images_with_labels(demon, 1)
add_images_with_labels(human, 2)
add_images_with_labels(spirit, 3)

#add_images_with_labels(angel, 1)
#add_images_with_labels(goblin, 3)
#add_images_with_labels(merfolk, 5)

#add_images_with_labels(zombie, 7)

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



