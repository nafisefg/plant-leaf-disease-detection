import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import directories, load_images_from_directories, preprocess_image
from segmentation import segment_image, select_infected_segment
from feature_extraction import extract_features


def prepare_date(features, labels):
    # Prepare data for SVM classification
    X = np.array(features)
    y = np.array(labels)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data into training and testing sets
    # random_state: pass an int for reproducible output across multiple function calls
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,test_size=0.2, random_state=42)

    # standardize the feature data ---> SVM performs better
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # save the scaler
    joblib.dump(scaler, 'scaler.joblib')

    return X_train, X_test, y_train, y_test, le


if __name__ == '__main__':

    # Load images and labels
    images, labels = load_images_from_directories(directories)

    # Preprocess all images
    preprocessed_images = [preprocess_image(img) for img in images]

    # Segmentation
    infected_segments = []
    for i, img in enumerate(preprocessed_images):
        segmented_img, kmeans = segment_image(img)
        infected_segment, infected_mask = select_infected_segment(img, labels[i], kmeans)
        infected_segments.append(infected_segment)

    # Extract features for all images
    features = [extract_features(img) for img in infected_segments]  # a list of np arrays

    X_train, X_test, y_train, y_test, le = prepare_date(features, labels)

    # Train the SVM classifier
    # linear kernel  when the data is linearly separable (low cost)
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    # Save the trained SVM model
    joblib.dump(svm, 'trained_svm_model.joblib')
    print("Model saved as 'trained_svm_model.joblib'")

    # Predict and evaluate the model
    y_predicted = svm.predict(X_test)

    # Display accuracy
    accuracy = accuracy_score(y_test, y_predicted)
    print(f'Accuracy: {accuracy * 100:.2f}%')

