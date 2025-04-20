import os
import cv2
import numpy as np
from glob import glob

# Définition des chemins
input_folder = "CompressedDataSetCreated"
output_folder = "AugmentedDataSet"
os.makedirs(output_folder, exist_ok=True)

# Fonctions d'augmentation
def flip_image(image):
    return cv2.flip(image, 1)  # Miroir horizontal

def adjust_brightness(image, factor=1.2):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def add_gaussian_noise(image, mean=0, var=10):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gauss)
    return noisy_image

def rotate_image(image, angle=15):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def apply_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Lecture et augmentation des images
for subfolder in sorted(os.listdir(input_folder)):
    subfolder_path = os.path.join(input_folder, subfolder)
    output_subfolder_path = os.path.join(output_folder, subfolder)
    os.makedirs(output_subfolder_path, exist_ok=True)

    if os.path.isdir(subfolder_path):
        image_files = glob(os.path.join(subfolder_path, "*.jpg"))
        
        for img_path in image_files:
            image = cv2.imread(img_path)
            filename = os.path.basename(img_path).split('.')[0]
            
            transformations = {
                "flip": flip_image(image),
                "bright": adjust_brightness(image, 1.5),
                "noise": add_gaussian_noise(image),
                "rotate": rotate_image(image, 15),
                "blur": apply_blur(image)
            }
            
            cv2.imwrite(os.path.join(output_subfolder_path, f"{filename}.jpg"), image)
            for key, img_trans in transformations.items():
                cv2.imwrite(os.path.join(output_subfolder_path, f"{filename}_{key}.jpg"), img_trans)

print("Augmentation terminée !")
