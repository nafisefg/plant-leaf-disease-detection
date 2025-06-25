import os
import cv2


# Define the directories
directories = ['D:/plant-disease/data/powdery mildew',
               'D:/plant-disease/data/rust',
               'D:/plant-disease/data/bacterial blight',
               'D:/plant-disease/data/Cercospora leaf spot']


# load images from directories
def load_images_from_directories(directories):
    images = []
    labels = []
    for directory in directories:
        label = os.path.basename(directory)   # last part(tail) of the path
        for filename in os.listdir(directory):  # list of all files and directories in the specified path

            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels


# preprocess images
def preprocess_image(img, size=(256, 256)):
    # Resize image
    resized_img = cv2.resize(img, size)

    # Convert to HSV format
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

    # Apply median filter
    median_filtered_img = cv2.medianBlur(hsv_img, 5)

    # Enhance image contrast using histogram equalization
    # Focus on Value channel because it directly corresponds to the brightness of the image
    hsv_img[:, :, 2] = cv2.equalizeHist(median_filtered_img[:, :, 2])

    # Convert back to BGR
    enhanced_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    return enhanced_img

