import glob
import os
import pickle

import cv2
import numpy as np

training_img = []
training_label = []
for dir_path in glob.glob("training/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.png")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        training_img.append(img)
        training_label.append(img_label)
training_img = np.array(training_img)
training_label = np.array(training_label)

training_img_file = open('training_img_file', 'wb')
pickle.dump(training_img, training_img_file)
training_img_file.close()

training_label_file = open('training_label_file', 'wb')
pickle.dump(training_label, training_label_file)
training_label_file.close()

test_img = []
test_label = []
for dir_path in glob.glob("testing/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.png")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_img.append(img)
        test_label.append(img_label)
test_img = np.array(test_img)
test_label = np.array(test_label)

test_img_file = open('test_img_file', 'wb')
pickle.dump(test_img, test_img_file)
test_img_file.close()

test_label_file = open('test_label_file', 'wb')
pickle.dump(test_label, test_label_file)
test_label_file.close()
