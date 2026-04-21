import numpy as np
from PIL import Image
import os


## Read the image path
## Resize to target size
## RGB channels to Array
## Flatten the array to 1D
## Normalize to a vector of 3072 numbers

def load_image(image_path,target_size=(32,32)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_flat = img_array.reshape(-1)
    img_normalized = img_flat/255.0
    return img_normalized

## Read all images from path

def load_folder(folder_path,target_size=(32,32)):
    images = []
    labels = []

    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path,class_name)

        if not os.path.isdir(class_path):
            continue

        for image_file in os.listdir(class_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(class_path, image_file)
            img = load_image(image_path, target_size)
            
            images.append(img)
            labels.append(class_name)

    images_array = np.array(images)
    labels_array = np.array(labels)

    return images_array, labels_array