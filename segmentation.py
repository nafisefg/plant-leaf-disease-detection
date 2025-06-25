import cv2
import numpy as np
from sklearn.cluster import KMeans
from preprocessing import load_images_from_directories, directories


# k-means clustering for image segmentation
def segment_image(img, k=3):
    # Reshape the image into a 2D array where each row represents a pixel and its BGR values
    x = img.reshape((-1, 3))  # row dim: unknown(pixels)  col dim:3(b,g,r)   image.shape -> (unknown, 3)

    # fix the seed of the random number generator used for centroid initialization (random_state=0)
    # ensures that the K-Means algorithm produces the same results each time it's running on the same data
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x)

    # Get the cluster centers (which are the dominant colors) and assign each pixel to its nearest cluster center
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape the segmented image back to its original shape
    segmented_img = segmented_img.reshape(img.shape)

    # Convert the segmented image to unsigned 8-bit integer format (the values were floating points)
    segmented_img = segmented_img.astype(np.uint8)

    # returns the trained K-Means clustering model and the segmented image
    return segmented_img, kmeans


def select_infected_segment(img, label, kmeans):
    # Calculate the mean color of each segment
    mean_colors = kmeans.cluster_centers_  # mean_colors.shape --> (k, 3)

    # Calculate distance(column-sum norm) between each mean color and a reference color for infection
    if label == 'powdery mildew':
        reference_color = np.array([200, 200, 200])  # Very light gray color
    elif label == 'rust':
        reference_color = np.array([139, 69, 19])  # Reddish-brown color
    elif label == 'bacterial blight':
        reference_color = np.array([139, 69, 19])  # Darker reddish-brown
    else:
        reference_color = np.array([100, 50, 50]),  # Dark brownish

    distances = np.linalg.norm(mean_colors - reference_color, axis=1)  # axis=1: row wise
    # distances.shape --> (3,1)  each row represents the distance of each cluster from the reference color

    # Select the segment with the smallest distance to the reference color
    infected_segment_label = np.argmin(distances)

    # Create a mask for the infected segment
    infected_mask = (kmeans.labels_ == infected_segment_label).reshape(img.shape[:2])  # [:2]: height & width
    # (65536,1)-->(256,256)
    # Extract the infected segment from the original image
    infected_segment = cv2.bitwise_and(img, img, mask=infected_mask.astype(np.uint8))

    return infected_segment, infected_mask

