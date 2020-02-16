import os
import shutil


def split_data():
    if os.path.isdir("data"):
        class_names = os.listdir("data")

    # Creating the training and test folder
    if not os.path.isdir("training"):
        os.mkdir("training")
    if not os.path.isdir("testing"):
        os.mkdir("testing")

    for folder_name in class_names:
        images = os.listdir("data/" + folder_name)
        if not os.path.isdir("testing/" + folder_name):
            os.mkdir("testing/" + folder_name)
        if not os.path.isdir("training/" + folder_name):
            os.mkdir("training/" + folder_name)
        test_images = images[:8]
        train_images = images[8:]
        for image in test_images:
            shutil.copy2("data/" + folder_name + "/" + image, "testing/" + folder_name)
            print("done", image, "in test")
        for image in train_images:
            shutil.copy2("data/" + folder_name + "/" + image, "training/" + folder_name)
            print("done", image, "in train")


if __name__ == "__main__":
    split_data()
