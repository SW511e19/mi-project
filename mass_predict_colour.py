#TODO: Make agent only load once instead of for every folder
#TODO: Split fileIterator into multiple functions to make it less yucky

import numpy
from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16

trainPath = "test_creature_color/tests/"

class_label_names = [
    "blue",
    "red",
    "green",
    "black",
    "white",
    "new_multicolour",
    "colourless"
]

result_int_list = []

def image_extractor(path, dir_name):
    dir_path = Path(path + dir_name + "/")
    image_list = []
    for img_path in dir_path.glob("*.png"):
        img = image.load_img(str(img_path), target_size=(224, 224))
        # Convert the image to a numpy array
        image_converted_to_array = image.img_to_array(img)
        image_list.append(image_converted_to_array)

    images_as_numpy_array = numpy.array(image_list)

    # Normalize the data
    images_as_numpy_array = vgg16.preprocess_input(images_as_numpy_array)

    # Use the pre-trained neural network to extract features from our test image (the same way we did to train the model)
    feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = feature_extraction_model.predict(images_as_numpy_array)

    return features

def predict(dir_count, dir_list, model, features):
    # Given the extracted features, make a final prediction using our own model
    results = model.predict(features)
    count = 0
    for result in results:
        most_likely_class_index = int(np.argmax(result))

        # Checks if the prediction matches the expected category
        dir_label = class_label_names[dir_count]
        result_label = class_label_names[most_likely_class_index]
        gotCorrectResult = dir_label == result_label

        result_as_int = 0
        if gotCorrectResult:
            result_as_int = 1
        dir_list.append(result_as_int)
        count += 1

        # Prints what was predicted and what was expected for each image
        class_label = class_label_names[most_likely_class_index]
        print("got " + class_label + " expected " + class_label_names[dir_count])

def load_model():
    # Load the json file that contains the model's structure
    f = Path("model_structure.json")
    model_structure = f.read_text()

    # Recreate the Keras model object from the json data
    model = model_from_json(model_structure)

    # Re-load the model's trained weights
    model.load_weights("model_weights.h5")

    return model

def directory_iterator(path, dir, model):
    dir_count = 0
    for dir_name in dir:
        dir_list = []
        features = image_extractor(path, dir_name)
        predict(dir_count, dir_list, model, features)
        result_int_list.append(dir_list)
        dir_count += 1

def tally():
    correct_results = 0
    for dir in range(len(result_int_list)):
        for result in range(len(result_int_list[dir])):
            correct_results += result_int_list[dir][result]

    dataset_size = (12 * 5)

    correct_percentage = ((correct_results / dataset_size) * 100)

    print("Got " + str(correct_results) + " out of " + str(dataset_size) + ". Accuracy: " + str(correct_percentage))

directory_iterator(trainPath, class_label_names, load_model())
tally()



