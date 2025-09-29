import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import layers, models, regularizers

# Constants
DATA_DIR = '/kaggle/input/tomato-lef-disease-augmented/'
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 4
FRAC_CLIENTS = 0.5
NUM_CLIENTS = 20

# Load dataset paths and labels
def load_data(data_dir):
    images, labels = [], []
    class_names = os.listdir(data_dir)

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    images.append(img_path)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels), class_names

# Custom generator to load images from file paths
def image_generator(file_paths, labels, batch_size):
    while True:
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            images = []
            for path in batch_paths:
                try:
                    img = load_img(path, target_size=(224, 224))  # Resize for ResNet50V2
                    img = img_to_array(img) / 255.0  # Normalize
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")

            yield np.array(images), np.array(batch_labels)

# Plot class distribution for each client
def plot_class_distribution(clients_data, class_names):
    plt.figure(figsize=(28, 14))
    for client_id, (x, y) in enumerate(clients_data):
        count = Counter(y)
        distribution = [count.get(i, 0) for i in range(NUM_CLASSES)]
        plt.subplot(2, (NUM_CLIENTS + 1) // 2, client_id + 1)
        bars = plt.bar(class_names, distribution)
        plt.title(f'Client {client_id + 1} Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Create a custom model using ResNet50V2
def create_custom_model():
    base_model = ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model layers

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Federated Learning Process
class FederatedLearning:
    def __init__(self, clients_data):
        self.clients_data = clients_data
        self.global_model = create_custom_model()
        self.global_weights = self.global_model.get_weights()

    def client_update(self, client_data):
        model = tf.keras.models.clone_model(self.global_model)
        model.set_weights(self.global_weights)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        train_gen = image_generator(client_data[0], client_data[1], BATCH_SIZE)
        steps_per_epoch = len(client_data[0]) // BATCH_SIZE

        val_size = int(0.1 * len(client_data[0]))
        x_val, y_val = client_data[0][:val_size], client_data[1][:val_size]
        val_gen = image_generator(x_val, y_val, BATCH_SIZE)

        history = model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps_per_epoch,
                            validation_steps=len(x_val) // BATCH_SIZE, epochs=EPOCHS, verbose=1)

        return model.get_weights(), history.history

    def federated_averaging(self, client_weights):
        new_weights = []
        for layer_weights in zip(*client_weights):
            new_weights.append(np.mean(layer_weights, axis=0))
        return new_weights

    def train(self):
        # Train the global model first
        print('Training global model...')
        self.global_model.fit(image_generator(self.clients_data[0][0], self.clients_data[0][1], BATCH_SIZE),
                               steps_per_epoch=len(self.clients_data[0][0]) // BATCH_SIZE,
                               epochs=EPOCHS, verbose=1)

        for round in range(1, 3):  # Train for 10 rounds
            print(f'Round {round}/10')
            selected_clients = np.random.choice(range(len(self.clients_data)),
                                                int(FRAC_CLIENTS * len(self.clients_data)), replace=False)
            client_weights = []
            val_losses = []
            val_accuracies = []

            for client in selected_clients:
                weights, history = self.client_update(self.clients_data[client])
                client_weights.append(weights)
                val_losses.append(history['val_loss'][-1])
                val_accuracies.append(history['val_accuracy'][-1])

            self.global_weights = self.federated_averaging(client_weights)
            print(f'Validation Loss: {np.mean(val_losses):.4f}, Validation Accuracy: {np.mean(val_accuracies):.4f}')

# Load data and split for clients
image_paths, labels, class_names = load_data(DATA_DIR)
x_train, x_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Split data among clients
num_samples_per_client = len(x_train) // NUM_CLIENTS
clients_data = [
    (x_train[i * num_samples_per_client:(i + 1) * num_samples_per_client],
     y_train[i * num_samples_per_client:(i + 1) * num_samples_per_client])
    for i in range(NUM_CLIENTS)
]

# Plot class distribution for each client
plot_class_distribution(clients_data, class_names)

# Train the federated learning model
federated_learning = FederatedLearning(clients_data)
federated_learning.train()

# Evaluate on test data using custom image generator
test_gen = image_generator(x_test, y_test, BATCH_SIZE)

# Evaluate model on test data
model = create_custom_model()
model.set_weights(federated_learning.global_weights)

# Collect predictions
y_pred = []
y_true = []

# Calculate number of steps needed
num_steps = (len(x_test) + BATCH_SIZE - 1) // BATCH_SIZE

for _ in range(num_steps):
    batch_x, batch_y = next(test_gen)
    predictions = model.predict(batch_x, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(batch_y)

# Convert predictions and true labels to numpy arrays
y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Calculate test accuracy
test_accuracy = np.mean(y_pred == y_true)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Classification report
if len(y_true) == len(y_pred):
    print(classification_report(y_true, y_pred, target_names=class_names))
else:
    print("Warning: Mismatch in length between true labels and predictions.")

# Confusion Matrix
if len(y_true) == len(y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add counts inside the boxes
    thresh = conf_matrix.max() / 2.  # Threshold for color contrast
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.show()
else:
    print("Unable to plot confusion matrix due to length mismatch.")