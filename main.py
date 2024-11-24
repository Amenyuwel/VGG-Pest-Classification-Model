from flask import Flask, request, jsonify
from flask_cors import CORS
from app import CLASS_NAMES, load_model_and_vgg19, extract_features
from datetime import datetime
from app import train_model
import traceback
import os
import cv2
import base64
import numpy as np
import sqlite3

# source venv/bin/activate

#######################################
# ANDROID STUDIO LOGICS AND ENDPOINTS #
#######################################
app = Flask(__name__)
CORS(app)

# ANDROID STUDIO
USER_FOLDER = 'user_capture'
app.config['USER_FOLDER'] = USER_FOLDER

# Load your pre-trained model once at the start
model, vgg19 = load_model_and_vgg19()



# Endpoint for training from admin
@app.route('/train', methods=['POST'])
def train():
    try:
        print("Starting training process...")
        model = train_model()  # Call the train_model function from harvest.py
        return jsonify({'message': 'Training started successfully.'}), 202
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Function to create the database
def create_database():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Create the pests table
    cursor.execute('''CREATE TABLE IF NOT EXISTS pests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image TEXT NOT NULL,
        pest_type TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        address TEXT
    )''')

    conn.commit()
    conn.close()
    print("Database and tables created successfully.")

# Function to insert prediction data into the database
def insert_prediction_into_db(image_data, predicted_class, latitude, longitude, address):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Save the classified image
    image_filename = save_base64_image(image_data, predicted_class)

    # Insert the prediction into the pests table
    cursor.execute('''INSERT INTO pests (image, pest_type, latitude, longitude, address)
                      VALUES (?, ?, ?, ?, ?)''',
                   (image_filename, predicted_class, latitude, longitude, address))

    conn.commit()
    conn.close()

# Function to save base64-encoded image
def save_base64_image(base64_image, pest_type, folder=USER_FOLDER):
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{pest_type}_{timestamp}.png"

    if base64_image.startswith('data:image'):
        base64_image = base64_image.split(',')[1]

    image_data = base64.b64decode(base64_image)
    file_path = os.path.join(folder, filename)
    with open(file_path, 'wb') as f:
        f.write(image_data)

    return filename

def predict_base64_image(base64_image):
    if not model or not vgg19:
        raise ValueError("Model is not loaded.")

    if base64_image.startswith('data:image'):
        base64_image = base64_image.split(',')[1]

    # Convert base64 string to a NumPy array
    image_data = base64.b64decode(base64_image)

    # Convert the bytes to a numpy array and reshape
    np_image = np.frombuffer(image_data, np.uint8)

    # Decode the image
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Resize the image to the expected input size
    image_resized = cv2.resize(image, (224, 224))  # VGG19 requires (224, 224)

    # Expand dimensions to match the input shape (1, height, width, channels)
    image_array = np.expand_dims(image_resized, axis=0)

    # Extract features using the VGG19 model
    features = extract_features(vgg19, image_array)

    # Ensure it matches the classifier's expected input shape
    # This is where we need to match the input shape for your classifier
    features = np.reshape(features, (1, -1))  # Reshape if needed to (1, features)

    # Predict the class using the model
    predictions = model.predict(features)

    # Get the predicted class index and confidence
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    return predicted_index, confidence


# Function to generate recommendations based on the pest type
def generate_recommendations(pest_type):
    recommendations = {
        'fall_armyworm': "Use biological control agents like Trichogramma.",
        'snail': "Introduce natural predators such as ducks or use organic repellents.",
        'stem_borer': "Use pheromone traps and ensure proper crop rotation."
    }
    return recommendations.get(pest_type, "No recommendations available.")

if __name__ == '__main__':
    create_database()
    app.run(debug=True, host='0.0.0.0', port=5000)
