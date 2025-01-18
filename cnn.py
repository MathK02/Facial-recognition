import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import re

class DatasetLoader:
    @staticmethod
    def load_faces(data_path):
        images = []
        labels = []
        image_shape = None

        config = {
            'folder_pattern': r'yaleB(\d+)',
            'id_extraction': lambda x: int(re.search(r'yaleB(\d+)', x).group(1))
        }

        subject_dirs = [d for d in os.listdir(data_path)
                        if os.path.isdir(os.path.join(data_path, d)) and re.match(config['folder_pattern'], d)]

        for subject_dir in sorted(subject_dirs):
            subject_id = config['id_extraction'](subject_dir)
            subject_path = os.path.join(data_path, subject_dir)

            image_files = [f for f in os.listdir(subject_path) if f.lower().endswith('.pgm')]

            for img_file in image_files:
                try:
                    img_path = os.path.join(subject_path, img_file)
                    img = Image.open(img_path).convert('L')  # Grayscale
                    
                    # Redimensionner toutes les images à une taille fixe
                    if image_shape is None:
                        image_shape = (100, 100)  # Taille fixe (à ajuster si besoin)
                    img = img.resize(image_shape)
                    
                    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to 0-1
                    images.append(img_array)
                    labels.append(subject_id)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue

        X = np.array(images)
        y = np.array(labels)
        return X, y, image_shape


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


def main():
    data_path = "CroppedYale"  # Change to your dataset path
    X, y, image_shape = DatasetLoader.load_faces(data_path)

    # Reshape X for CNN and one-hot encode y
    X = X.reshape(-1, image_shape[0], image_shape[1], 1)  # Add channel dimension
    y = to_categorical(y)

    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Build and train CNN
    model = build_cnn(input_shape=(image_shape[0], image_shape[1], 1), num_classes=y.shape[1])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2%}")

    # Confusion matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
