import os
import cv2
import numpy as np
import shutil
import time
from flask import Flask, request, jsonify
import requests
from datetime import datetime
from werkzeug.utils import secure_filename
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.regularizers import l2
import pandas as pd
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import base64
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, app, request, jsonify
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from flask import Flask, request, jsonify
from flask_cors import CORS

# source venv/bin/activate
# Invoke-WebRequest -Uri "https://desktop-3mj7q7f.tail98551e.ts.net/web_train" -Method Post
# ./run.sh
# https://desktop-3mj7q7f.tail98551e.ts.net/

###############
# MODEL LOGIC #
###############

# Initialize the Flask app
app = Flask(__name__)
# Allow Cross-Origin Resource Sharing
CORS(app)

# Define constants
IMAGE_SIZE = (100, 100)

CLASS_NAMES = ['fall_armyworm', 'snail', 'stem_borer', 'unknown']

MODEL_DIR = os.path.abspath("models/harvest_model.keras")

CLASS_LABELS_FILE = os.path.abspath('class_labels.csv')

DATA_DIR = os.path.abspath("dataset_images/")

GRAPH_SAVE_PATH = os.path.abspath("graphs")

ADMIN_FOLDER = 'admin_upload'

USER_FOLDER = 'user_capture'

# Flask app configuration
app.config['ADMIN_FOLDER'] = ADMIN_FOLDER
app.config['USER_FOLDER'] = USER_FOLDER


# Ensure necessary directories exist
os.makedirs(ADMIN_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
os.makedirs(GRAPH_SAVE_PATH, exist_ok=True)

# Global variables for the model
model = None
vgg19 = None

# Load the trained model and for feature extraction
def load_model_and_vgg19():
    global model, vgg19
    vgg19 = VGG19(include_top=False, weights='imagenet')
    model = load_model('models/harvest_model.keras')
    return model, vgg19

model, vgg19 = load_model_and_vgg19()

# Load class labels from a CSV file
def load_class_labels():
    if os.path.exists(CLASS_LABELS_FILE):
        df = pd.read_csv(CLASS_LABELS_FILE, header=None, names=['image_path', 'target'])
        return df
    return pd.DataFrame(columns=['image_path', 'target'])

# Move image subfolders that contain images to a target directory
def move_subfolders_with_images(source_dir, target_dir):
    # Define common image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

    # Loop through all items in the source directory
    for subfolder in os.listdir(source_dir):
        subfolder_path = os.path.join(source_dir, subfolder)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # Check if there is at least one image in the subfolder
            contains_image = any(
                file.endswith(image_extensions)
                for file in os.listdir(subfolder_path)
            )

            # If the subfolder contains an image, move it to the target directory
            if contains_image:
                target_subfolder_path = os.path.join(target_dir, subfolder)
                shutil.move(subfolder_path, target_subfolder_path)
                print(f"Moved folder: {subfolder} to {target_dir}")

# Plot and save training history graphs (loss and accuracy)
def plot_training_history(history, save_path):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'accuracy_plot.png'))
    plt.close()

# Function to plot and save the confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

# Function to save classification report as a text file
def save_classification_report(y_true, y_pred, class_names, save_path):
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(save_path, 'classification_report.txt'), 'w') as file:
        file.write(report)

# Data Loading and Preprocessing
def data_dictionary(base_data_dir=DATA_DIR, class_names=CLASS_NAMES):
    data_dict = {"image_path": [], "target": []}

    # Load existing data
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(base_data_dir, class_name)

        if os.path.exists(class_dir):
            image_files = os.listdir(class_dir)
            for img_file in image_files:
                data_dict["image_path"].append(os.path.join(class_dir, img_file))
                data_dict["target"].append(idx)

    # Load new uploads from CSV
    new_labels = load_class_labels()
    for _, row in new_labels.iterrows():
        data_dict["image_path"].append(row['image_path'])
        data_dict["target"].append(row['target'])

    return pd.DataFrame(data_dict)

# Load dataset and split it into training, validation, and testing sets
def load_data(image_size=IMAGE_SIZE):
    df = data_dictionary()
    
    # Stratified splitting
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['target'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['target'], random_state=42)

    # Read and resize images for training, validation, and testing
    x_train = np.array([cv2.resize(cv2.imread(path), image_size) for path in train_df['image_path']])
    y_train = np.array(train_df['target'])

    x_val = np.array([cv2.resize(cv2.imread(path), image_size) for path in val_df['image_path']])
    y_val = np.array(val_df['target'])

    x_test = np.array([cv2.resize(cv2.imread(path), image_size) for path in test_df['image_path']])
    y_test = np.array(test_df['target'])

    return x_train, x_val, x_test, y_train, y_val, y_test

# Model Training
def build_model(input_shape):
    inputs = Input(shape=(input_shape,))
    
    # First Dense Layer with L2 regularization and Batch Normalization
    x = Dense(1024, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = BatchNormalization()(x)  # Batch Normalization before activation
    x = tf.keras.layers.Activation('relu')(x)  # Apply activation after batch normalization
    x = Dropout(0.5)(x)  # Dropout for regularization
    
    # Second Dense Layer with L2 regularization and Batch Normalization
    x = Dense(512, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    # Third Dense Layer with L2 regularization and Batch Normalization
    x = Dense(256, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    # Fourth Dense Layer with L2 regularization and Batch Normalization
    x = Dense(128, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Output Layer: No batch normalization or dropout on the output
    outputs = Dense(len(CLASS_NAMES), activation='softmax')(x)

    # Compile the model with a sparse categorical cross-entropy loss function
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    return model

# Extract features using the VGG19 model for image data
def extract_features(vgg_model, x_data):
    # Preprocess the input data for VGG19
    x_data = preprocess_input(x_data)
    
    # Predict features using the VGG model
    features = vgg_model.predict(x_data)
    
    # Reshape the features to flatten them for input into the classifier
    features = features.reshape(features.shape[0], -1)  # Flatten the features
    
    return features

# Train the model using extracted features
def train_model():
    print("Loading data...")
    x_train, x_val, x_test, y_train, y_val, y_test = load_data()
    print("Data loaded successfully.")

    # Load the pre-trained VGG19 model for feature extraction
    print("Loading VGG19 model for feature extraction...")
    vgg19 = VGG19(include_top=False, weights='imagenet')

    print("Extracting features for training data...")
    features_train = extract_features(vgg19, x_train)
    print("Training features extracted.")

    print("Extracting features for validation data...")
    features_val = extract_features(vgg19, x_val)
    print("Validation features extracted.")

    print("Extracting features for testing data...")
    features_test = extract_features(vgg19, x_test)
    print("Testing features extracted.")

    # Build and train the model
    print("Building the model...")
    model = build_model(features_train.shape[1])
    print("Model built successfully.")

    print("Starting training...")
    
    # Capture the history of training for plotting
    history = model.fit(features_train, y_train, batch_size=32, epochs=40,
                        validation_data=(features_val, y_val))

    print("Training completed successfully.")

    # Plot and save the training history graphs
    plot_training_history(history, GRAPH_SAVE_PATH)

    # Predict on the test set
    y_pred = model.predict(features_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Plot and save the confusion matrix
    plot_confusion_matrix(y_test, y_pred_labels, CLASS_NAMES, GRAPH_SAVE_PATH)

    # Save the classification report
    save_classification_report(y_test, y_pred_labels, CLASS_NAMES, GRAPH_SAVE_PATH)

    # Save the trained model
    print("Saving the trained model...")
    model.save(MODEL_DIR)
    print("Model saved successfully.")

    return model

def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    np_img = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return img

def preprocess_image(image, target_size=IMAGE_SIZE):
    img_resized = cv2.resize(image, target_size)
    img_array = np.array([img_resized])
    img_preprocessed = preprocess_input(img_array)
    return img_preprocessed     

# Function to update pest information and recommendations for new classes

# def add_class_info(class_name, pest_info, recommendation):
#     # Load existing pest information if it exists
#     if os.path.exists(CLASS_LABELS_FILE):
#         class_labels = pd.read_csv(CLASS_LABELS_FILE, header=None, names=['class_name', 'pest_info', 'recommendation'])
#     else:
#         class_labels = pd.DataFrame(columns=['class_name', 'pest_info', 'recommendation'])
    
#     # Append new class information
#     new_class = pd.DataFrame([[class_name, pest_info, recommendation]], columns=['class_name', 'pest_info', 'recommendation'])
#     class_labels = pd.concat([class_labels, new_class], ignore_index=True)
    
#     # Save updated class information to CSV
#     class_labels.to_csv(CLASS_LABELS_FILE, index=False, header=False)

# STORES USER CAPTURED IMAGES
@app.route('/user_upload', methods=['POST'])
def upload_user_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    image = request.files['image']

    # Ensure the image has a filename
    if image.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    # Secure the filename to prevent security issues
    filename = secure_filename(image.filename)

    # Define the path to store the image in the 'user_capture' folder
    image_path = os.path.join(app.config['USER_FOLDER'], filename)

    # Save the image to the user_capture folder
    image.save(image_path)

    return jsonify({'message': 'Image uploaded successfully', 'file_path': image_path}), 200

# ADMIN UPLOAD MODIFICATION THAT ACCEPTS ADDITIONAL INFORMATION FOR EACH CLASS IN ADMIN_TRAINING
#     @app.route('/admin_upload', methods=['POST'])
# def upload_images():
#     if 'class_name' not in request.form:
#         return jsonify({'error': 'Class name is required'}), 400
    
#     class_name = request.form['class_name']
#     pest_info = request.form.get('pest_info', 'No information provided')
#     recommendation = request.form.get('recommendation', 'No recommendation available')

#     # Create the subdirectory based on class_name if it doesn't exist
#     class_folder = os.path.join(app.config['ADMIN_FOLDER'], secure_filename(class_name))
#     if not os.path.exists(class_folder):
#         os.makedirs(class_folder)

#     # Check if files are present in the request
#     if 'files[]' not in request.files:
#         return jsonify({'error': 'No files part in the request'}), 400

#     files = request.files.getlist('files[]')
#     if not files:
#         return jsonify({'error': 'No files selected'}), 400

#     saved_files = []
#     for file in files:
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'}), 400

#         # Secure the filename and save it to the class-specific folder
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(class_folder, filename)
#         file.save(file_path)
#         saved_files.append(filename)

#     # Register the new class name if it's not already in CLASS_NAMES
#     if class_name not in CLASS_NAMES:
#         CLASS_NAMES.append(class_name)

#     # Add pest information and recommendation
#     add_class_info(class_name, pest_info, recommendation)

#     return jsonify({'message': f'{len(saved_files)} file(s) uploaded successfully', 'files': saved_files}), 200


# STORES ADMIN UPLOADED IMAGES
@app.route('/admin_upload', methods=['POST'])
def upload_images():
    if 'class_name' not in request.form:
        return jsonify({'error': 'Class name is required'}), 400
    
    class_name = request.form['class_name']
    class_folder = os.path.join(ADMIN_FOLDER, secure_filename(class_name))

    # Create class folder if it doesn't exist
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    if 'files' not in request.files:  # Adjusted key to match PHP
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files')  # Adjusted key to match PHP
    if not files:
        return jsonify({'error': 'No files selected'}), 400

    # Count current images in the class folder
    current_image_count = len([f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))])

    # Check if uploading these files would exceed the limit
    if current_image_count + len(files) > 4000:
        return jsonify({'error': f"Uploading these files would exceed the limit of 4,000 images for class '{class_name}'."}), 400

    saved_files = []
    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Secure the filename and make it unique
        timestamp = int(time.time())
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        file_path = os.path.join(class_folder, filename)
        file.save(file_path)
        saved_files.append(filename)

    # Add the class name if it's new
    if class_name not in CLASS_NAMES:
        CLASS_NAMES.append(class_name)

    return jsonify({'message': f'{len(saved_files)} file(s) uploaded successfully', 'files': saved_files}), 200

# @app.route('/admin_upload', methods=['POST'])
# def upload_images():
#     if 'class_name' not in request.form:
#         return jsonify({'error': 'Class name is required'}), 400
    
#     class_name = request.form['class_name']

#     # Create the subdirectory based on class_name if it doesn't exist
#     class_folder = os.path.join(app.config['ADMIN_FOLDER'], secure_filename(class_name))
#     if not os.path.exists(class_folder):
#         os.makedirs(class_folder)

#     # Check if files are present in the request
#     if 'files[]' not in request.files:
#         return jsonify({'error': 'No files part in the request'}), 400

#     files = request.files.getlist('files[]')
#     if not files:
#         return jsonify({'error': 'No files selected'}), 400

#     saved_files = []
#     for file in files:
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'}), 400

#         # Secure the filename and save it to the class-specific folder
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(class_folder, filename)
#         file.save(file_path)
#         saved_files.append(filename)

#     # Register the new class name if it's not already in CLASS_NAMES
#     if class_name not in CLASS_NAMES:
#         CLASS_NAMES.append(class_name)

#     return jsonify({'message': f'{len(saved_files)} file(s) uploaded successfully', 'files': saved_files}), 200

# TRIGGERS TRAINING #
@app.route('/web_train', methods=['POST'])
def web_train():
    try:
        move_subfolders_with_images('admin_upload', 'dataset_images')
        model = train_model()  # Call the training function
        return jsonify({"message": "Model trained successfully!"}), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500


# Load class information from CSV file
def load_class_info():
    if os.path.exists(CLASS_LABELS_FILE):
        return pd.read_csv(CLASS_LABELS_FILE, header=None, names=['class_name', 'pest_info', 'recommendation'])
    return pd.DataFrame(columns=['class_name', 'pest_info', 'recommendation'])

# PREDICT THAT RETRIEVE PEST RECOMMENDATION INPUT FROM ADMIN_TRAINING
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the base64 image, latitude, longitude, and address from the request
#         base64_image = request.form.get('image')
#         latitude = request.form.get('latitude')
#         longitude = request.form.get('longitude')
#         address = request.form.get('address')

#         if base64_image is None:
#             return jsonify({'error': 'No image provided'}), 400

#         if latitude is None or longitude is None:
#             return jsonify({'error': 'Location data not provided'}), 400

#         if address is None:
#             return jsonify({'error': 'Address not provided'}), 400

#         # Get the current date and time
#         current_date = datetime.now().strftime("%Y-%m-%d")

#         # Decode and preprocess the image
#         image = decode_base64_image(base64_image)
#         preprocessed_image = preprocess_image(image)

#         # Extract features using VGG19
#         features = vgg19.predict(preprocessed_image)
#         features = features.reshape(features.shape[0], -1)

#         # Perform prediction using the trained model
#         predictions = model.predict(features)
#         predicted_class = np.argmax(predictions, axis=1)[0]

#         # Return the class name based on the predicted index
#         class_name = CLASS_NAMES[predicted_class]
#         print(class_name)

#         # Load class information from CSV file
#         class_info_df = load_class_info()

#         # Get the recommendation based on the predicted class
#         class_info = class_info_df[class_info_df['class_name'] == class_name]
#         if not class_info.empty:
#             recommendation = class_info['recommendation'].values[0]
#         else:
#             recommendation = "No recommendation available."

#         # Prepare the response for Android Studio
#         android_response = {
#             'pest_type': class_name,
#             'recommendation': recommendation,
#             'latitude': latitude,
#             'longitude': longitude,
#             'address': address,
#             'date_reported': current_date
#         }

#         # Return the JSON response directly to the Android client
#         return jsonify(android_response), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# PREDICTS USER CAPTURED IMAGES #
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the base64 image, latitude, longitude, and address from the request
        base64_image = request.form.get('image')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        address = request.form.get('address')

        if base64_image is None:
            return jsonify({'error': 'No image provided'}), 400

        if latitude is None or longitude is None:
            return jsonify({'error': 'Location data not provided'}), 400

        if address is None:
            return jsonify({'error': 'Address not provided'}), 400

        # Get the current date and time
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Decode and preprocess the image
        image = decode_base64_image(base64_image)
        preprocessed_image = preprocess_image(image)

        # Extract features using VGG19
        features = vgg19.predict(preprocessed_image)
        features = features.reshape(features.shape[0], -1)

        # Perform prediction using the trained model
        predictions = model.predict(features)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Return the class name based on the predicted index
        class_name = CLASS_NAMES[predicted_class]
        print(class_name)

        # Recommendations for each class
        recommendations = {
            'fall_armyworm': "Use insecticidal sprays and practice crop rotation.",
            'stem_borer': "Implement pheromone traps and remove infected plants.",
            'snail': "Hand-pick snails or use barriers to prevent them.",
            'unknown': "Consult a local expert for identification."
        }

        # Get the recommendation based on the predicted class
        recommendation = recommendations.get(class_name, "No recommendation available.")

        # Prepare the response for Android Studio
        android_response = {
            'pest_type': class_name,
            'recommendation': recommendation,
            'latitude': latitude,
            'longitude': longitude,
            'address': address,
            'date_reported': current_date
        }

        # Return the JSON response directly to the Android client
        return jsonify(android_response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
if __name__ == '__main__':
    load_model_and_vgg19()
    app.run(debug=True, host='0.0.0.0', port=5000)