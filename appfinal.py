from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS to enable cross-origin requests
import cv2
from deepface import DeepFace
from datetime import datetime
import numpy as np
import logging
import os
import traceback

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)  # Allow all origins by default

# Path to the reference image for face recognition
reference_image_path = os.path.join(os.path.dirname(__file__), 'chawki.jpg')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/facerecognition', methods=['POST'])  # Define the route for face recognition
def face_recognition():
    # Check if an image is provided in the request
    if 'image' not in request.files:
        logging.warning("No image provided in the request.")
        return jsonify({"message": "No image provided"}), 400
    
    image_file = request.files['image']
    
    try:
        # Read the image file into a NumPy array
        in_memory_file = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

        if frame is None:
            logging.error("Failed to decode image.")
            return jsonify({"message": "Failed to decode image"}), 400

        # Verify the received image against the reference image
        result = DeepFace.verify(frame, reference_image_path, model_name='VGG-Face', detector_backend='mtcnn')

        # Check if the person is verified
        if result['verified']:
            response_message = "This is YOU!"
            logging.info("User verified as the reference person.")
        else:
            # Save the current frame if the user is not verified
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f'intrus_{timestamp}.jpg'
            cv2.imwrite(filename, frame)  # Save the image
            response_message = "Not You!"
            logging.info(f"User NOT verified. Intruder image saved as '{filename}'.")

    except Exception as e:
        logging.error(f"Face recognition failed: {e}")
        logging.error(traceback.format_exc())  # Log the full traceback
        return jsonify({"message": "Face recognition failed"}), 500

    return jsonify({"face_recognition_result": result['verified'], "message": response_message}), 200

# Error handler for unhandled routes
@app.errorhandler(404)
def not_found(e):
    return jsonify({"message": "Resource not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode for development
