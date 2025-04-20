import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import tensorflow as tf
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class DatasetLoader:
    @staticmethod
    def get_dataset_config(data_path):
        """
        Determine dataset type and return appropriate configuration
        """
        base_folder = os.path.basename(data_path)
        if base_folder == "CroppedYale":
            return {
                'folder_pattern': r'yaleB(\d+)',
                'id_extraction': lambda x: int(re.search(r'yaleB(\d+)', x).group(1)),
                'name': 'Yale Database'
            }
        elif base_folder == "DataSetCreated":
            return {
                'folder_pattern': r's(\d+)',
                'id_extraction': lambda x: int(re.search(r's(\d+)', x).group(1)),
                'name': 'DataSet Database'
            }
        elif base_folder == "CompressedDataSetCreated":
            return {
                'folder_pattern': r's(\d+)',
                'id_extraction': lambda x: int(re.search(r's(\d+)', x).group(1)),
                'name': 'Compressed DataSet'
            }
        else:
            raise ValueError(f"Unknown dataset: {base_folder}")

    @staticmethod
    def load_faces(data_path, target_size=(64, 64)):
        """
        Load face database images and labels with flexible folder structure
        """
        images = []
        labels = []
        
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
            
            image_files = [f for f in os.listdir(subject_path) if f.lower().endswith(('.pgm', '.jpg'))]
            
            for img_file in image_files:
                try:
                    img_path = os.path.join(subject_path, img_file)
                    img = Image.open(img_path).convert('L').resize(target_size)
                    images.append(np.array(img, dtype=np.float32) / 255.0)
                    labels.append(subject_id)
                except Exception as e:
                    print(f"Error loading {img_file}: {str(e)}")
                    continue
        
        X = np.array(images).reshape(-1, target_size[0], target_size[1], 1)
        y = np.array(labels)
        
        print(f"\nLoaded {len(images)} images from {len(set(labels))} subjects")
        print(f"Image dimensions: {target_size}")
        
        return X, y

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def visualize_predictions(X_test, y_test, predictions, image_shape, X_train, y_train, n_samples=5):
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
        pred_label = np.argmax(predictions[test_idx])

        # Find the closest training image with the predicted label
        closest_idx = np.where(np.argmax(y_train, axis=1) == pred_label)[0]
        if len(closest_idx) > 0:
            closest_image = X_train[closest_idx[0]].reshape(image_shape)  # Take the first match
        else:
            closest_image = np.zeros_like(test_image)  # Fallback to a blank image if no match found
        
        # Show test image
        ax_left.imshow(test_image.squeeze(), cmap='gray')
        ax_left.axis('off')
        ax_left.set_title(f'True Label: {true_label}')
        
        # Show predicted image
        ax_right.imshow(closest_image.squeeze(), cmap='gray')
        ax_right.axis('off')
        if np.argmax(y_test[test_idx]) == pred_label:
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
    data_path = "CompressedDataSetCreated"
    
    # Load and preprocess data
    X, y = DatasetLoader.load_faces(data_path)
    
    # Encode labels
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    num_classes = len(lb.classes_)
    
    # Split data with stratification
    print("\nSplitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)} images")
    print(f"Test set size: {len(X_test)} images")
    
    # Build and train CNN model
    print("\nBuilding and training CNN...")
    cnn = build_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)
    history = cnn.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32
    )
    
    # Evaluate the model
    print("\nEvaluating CNN...")
    loss, accuracy = cnn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2%}")
    
    # Confusion matrix
    y_pred = cnn.predict(X_test)
    visualize_predictions(X_test, y_test, y_pred, X_train[0].shape, X_train, y_train, n_samples=5)

    print(f"\nTotal execution time: {time.time() - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
