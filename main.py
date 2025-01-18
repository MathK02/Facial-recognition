import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import re

class DatasetLoader:
    @staticmethod
    def get_dataset_config(data_path):
        """
        Determine dataset type and return appropriate configuration
        """
        base_folder = os.path.basename(data_path)
        if base_folder == "CroppedYale":
            return {
                'folder_pattern': r'yaleB(\d+)',  # Pattern to match yaleB01, yaleB02, etc.
                'id_extraction': lambda x: int(re.search(r'yaleB(\d+)', x).group(1)),
                'name': 'Yale Database'
            }
        elif base_folder == "DataSet":
            return {
                'folder_pattern': r's(\d+)',  # Pattern to match s1, s2, etc.
                'id_extraction': lambda x: int(re.search(r's(\d+)', x).group(1)),
                'name': 'DataSet Database'
            }
        else:
            raise ValueError(f"Unknown dataset: {base_folder}")

    @staticmethod
    def load_faces(data_path):
        """
        Load face database images and labels with flexible folder structure
        """
        images = []
        labels = []
        image_shape = None
        
        # Get dataset configuration
        config = DatasetLoader.get_dataset_config(data_path)
        print(f"\nLoading {config['name']}...")
        
        # Get valid subject directories
        subject_dirs = [d for d in os.listdir(data_path) 
                       if os.path.isdir(os.path.join(data_path, d)) 
                       and re.match(config['folder_pattern'], d)]
        
        for subject_dir in tqdm(sorted(subject_dirs), desc="Processing subjects"):
            subject_id = config['id_extraction'](subject_dir)
            subject_path = os.path.join(data_path, subject_dir)
            
            image_files = [f for f in os.listdir(subject_path) if f.lower().endswith('.pgm')]
            
            for img_file in image_files:
                try:
                    img_path = os.path.join(subject_path, img_file)
                    img = Image.open(img_path).convert('L')
                    
                    if image_shape is None:
                        image_shape = np.array(img).shape
                    
                    if np.array(img).shape == image_shape:
                        img_array = np.array(img, dtype=np.float64)
                        img_array = DatasetLoader.normalize_image(img_array)
                        images.append(img_array.flatten())
                        labels.append(subject_id)
                
                except Exception as e:
                    print(f"Error loading {img_file}: {str(e)}")
                    continue
        
        X = np.array(images)
        y = np.array(labels)
        
        print(f"\nLoaded {len(images)} images from {len(set(labels))} subjects")
        print(f"Image dimensions: {image_shape}")
        
        return X, y, image_shape

    @staticmethod
    def normalize_image(img):
        """
        Apply advanced image normalization
        """
        # Histogram equalization
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        img_normalized = cdf[img.astype('uint8')]
        
        # Scale to 0-1
        img_normalized = img_normalized / 255.0
        
        return img_normalized

class EigenfaceRecognizer:
    def __init__(self, n_components=150):
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
        self.mean_face = None
        self.labels = None
    
    def fit(self, X, y):
        """
        Train the eigenface recognizer
        """
        print("\nTraining Eigenface Recognizer...")
        
        # Scale the data
        print("Scaling data...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Store mean face
        self.mean_face = self.scaler.mean_
        
        # Compute eigenfaces using PCA
        print("Computing eigenfaces using SVD...")
        start_time = time.time()
        self.pca = PCA(n_components=self.n_components, svd_solver='randomized', random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Train KNN classifier
        print("Training KNN classifier...")
        self.knn.fit(X_pca, y)
        
        print(f"Completed in {time.time() - start_time:.2f} seconds")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.2%}")
        
        self.labels = y
        print("Training completed!")
    
    def predict(self, X):
        """
        Predict labels using the trained classifier
        """
        print("\nPredicting labels for test faces...")
        
        # Scale and project test data
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Predict using KNN
        predictions = self.knn.predict(X_pca)
        
        return predictions
    
    def visualize_eigenfaces(self, image_shape, n_eigenfaces=5):
        """
        Visualize eigenfaces
        """
        print("\nVisualizing eigenfaces...")
        fig, axes = plt.subplots(1, n_eigenfaces, figsize=(2*n_eigenfaces, 2))
        for i in range(n_eigenfaces):
            eigenface = self.pca.components_[i].reshape(image_shape)
            axes[i].imshow(eigenface, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Eigenface {i+1}')
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
    def visualize_predictions(self, X_test, y_test, predictions, image_shape, X_train, y_train, n_samples=5):
        """
        Visualize random test images and their predictions, showing both the test image and the predicted image.
        """
        print("\nVisualizing random predictions...")
        
        # Randomly select n_samples indices
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        
        # Create a figure with n_samples rows and 2 columns
        fig, axes = plt.subplots(n_samples, 2, figsize=(8, 3*n_samples))
        fig.suptitle('Test Images vs Predictions\nLeft: Original, Right: Closest Match', fontsize=12)
        
        for idx, (ax_left, ax_right) in enumerate(axes):
            # Get the test image
            test_idx = indices[idx]
            test_image = X_test[test_idx].reshape(image_shape)
            true_label = y_test[test_idx]
            pred_label = predictions[test_idx]
            
            # Find the closest training image with the predicted label
            closest_idx = np.where(y_train == pred_label)[0]
            if len(closest_idx) > 0:
                closest_image = X_train[closest_idx[0]].reshape(image_shape)  # Take the first match
            else:
                closest_image = np.zeros_like(test_image)  # Fallback to a blank image if no match found
            
            # Show test image
            ax_left.imshow(test_image, cmap='gray')
            ax_left.axis('off')
            ax_left.set_title(f'True Label: {true_label}')
            
            # Show predicted image
            ax_right.imshow(closest_image, cmap='gray')
            ax_right.axis('off')
            if true_label == pred_label:
                color = 'green'
                result = 'Correct'
            else:
                color = 'red'
                result = 'Wrong'
            ax_right.set_title(f'Predicted: {pred_label}\n({result})', color=color)
        
        plt.tight_layout()
        plt.show()

def main():
    total_start_time = time.time()
    
    # Change this line to switch between databases
    data_path = "CroppedYale"  # or "DataSet"
    
    # Load and preprocess data
    X, y, image_shape = DatasetLoader.load_faces(data_path)
    
    # Split data with stratification
    print("\nSplitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)} images")
    print(f"Test set size: {len(X_test)} images")
    
    # Create and train recognizer
    recognizer = EigenfaceRecognizer(n_components=150)
    recognizer.fit(X_train, y_train)
    
    # Make predictions
    predictions = recognizer.predict(X_test)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nRecognition accuracy: {accuracy:.2%}")
    
    # Visualize results
    recognizer.visualize_eigenfaces(image_shape, n_eigenfaces=5)
    recognizer.plot_confusion_matrix(y_test, predictions)
    
    # Add visualization of random predictions
    recognizer.visualize_predictions(X_test, y_test, predictions, image_shape, X_train, y_train, n_samples=5)
    
    print(f"\nTotal execution time: {time.time() - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
