from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2
import numpy as np
from preprocessing import directories, load_images_from_directories, preprocess_image


# extract GLCM and LBP features
def extract_features(img):
    if img.ndim == 2:  # ndim: number of dimensions
        gray_img = img
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid image format")
    # Extract GLCM features
    # the GLCM matrix will be symmetric (ignoring the order of value pairs)
    # levels=256: for an 8-bit input image (number of gray-levels)
    glcm = graycomatrix(gray_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    #  Measures the intensity contrast
    contrast = graycoprops(glcm, 'contrast')[0]  # The function returns a 2D array (one row per distance-angle pair)
    # similar to contrast but weighs differences linearly rather than quadratically
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0]
    # Higher values indicate more homogeneity
    homogeneity = graycoprops(glcm, 'homogeneity')[0]
    # Higher values indicate more uniformity
    energy = graycoprops(glcm, 'energy')[0]
    # ranges between -1 and 1. Higher values indicate a positive correlation
    correlation = graycoprops(glcm, 'correlation')[0]
    # indicates image texture uniformity
    asm = graycoprops(glcm, 'ASM')[0]

    # Extract LBP features
    # P: num of points  R:radius  uniform: rotation invariant
    lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
    # Numpy histogram gives the numerical representation of the dataset
    # lbp.ravel(): flattens the 2D LBP array into a 1D array
    # np.arange(start, stop): [Start, Stop) -> [0 1 2 3 4 5 6 7 8 9 10]
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))  # we consider LBP values from 0 to 9
    lbp_hist = lbp_hist.astype("float")  # This is necessary for normalization
    lbp_hist /= lbp_hist.sum()  # Normalize the histogram

    # Combine features into a single feature vector
    features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, asm, lbp_hist])
    return features


if __name__ == "__main__":
    # Load images and labels
    images, labels = load_images_from_directories(directories)
    extract_features(images[0])

