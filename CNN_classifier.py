import sys
import os
import pickle
import random
import cv2
import numpy as np

# from webscraper import photos

# import matplotlib.pyplot as plt

data_dir = "C:/Users/Nathan Gillespie/PycharmProjects/machine_learning_final_project/Resources/au_images"  # os.path()
labels = ["Black", "Blue", "Brown", "Cyan", "Green", "Lime", "Orange", "Pink", "Purple", "Red", "White", "Yellow"]
training_data = []
IMG_h = 100
IMG_w = 200
for lbl in labels:
    path = os.path.join(data_dir, lbl)

    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)


new_array = cv2.resize(img_array, (IMG_h, IMG_w))
def pickle_file(x, y)
    pickle_out = open("x.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

def create_training_dataset():
    for lbl in labels:
        path = os.path.join(data_dir, lbl)
        class_num = labels.index(lbl)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_h, IMG_w))
            training_data.append([new_array, class_num])


create_training_dataset()
print(len(training_data))

# shuffle data

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X_train = []
y = []
for features, label in training_data:
    X_train.append(features)
    y.append(label)
X_train = np.array(X_train).reshape(-1, IMG_h, IMG_w, 1)  # changing to 3 would give this rgb
y = np.array(y)

pickle_file(X_train, y)
