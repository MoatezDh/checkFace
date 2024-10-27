from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS  # Import CORS
import cv2
from deepface import DeepFace
from datetime import datetime
import numpy as np
import logging
import os
import traceback

# Initialize Flask app and configure it
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins
api = Api(app)

# Define path to the reference image
reference_image_path = os.path.join(os.path.dirname(__file__), 'chawki.jpg')
logging.basicConfig(level=logging.DEBUG)

class FaceRecognition(Resource):
    def post(self):
        # Check if an image is provided in the request
        if 'image' not in request.files:
            logging.warning("No image provided in the request.")
            return {"message": "No image provided"}, 400
        
        image_file = request.files['image']
        
        try:
            # Read the image file as a NumPy array
            in_memory_file = np.frombuffer(image_file.read(), np.uint8)
            frame = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

            if frame is None:
                logging.error("Failed to decode image.")
                return {"message": "Failed to decode image"}, 400

            # Use DeepFace to verify the received image against the reference image
            result = DeepFace.verify(frame, reference_image_path, model_name='VGG-Face', detector_backend='mtcnn')

            # Check if the person is the same as in the saved image
            if result['verified']:
                response_message = "This is YOU!"
                logging.info("User verified as the reference person.")
            else:
                # Create a timestamp for the filename
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f'intrus_{timestamp}.jpg'
                cv2.imwrite(filename, frame)  # Save the current frame as an image
                response_message = "Not You!"
                logging.info(f"User NOT verified. Intruder image saved as '{filename}'.")

        except Exception as e:
            logging.error(f"Face recognition failed: {e}")
            logging.error(traceback.format_exc())  # Log the full traceback
            return {"message": "Face recognition failed"}, 500

        return {"face_recognition_result": result['verified'], "message": response_message}, 200

# Register the FaceRecognition resource
api.add_resource(FaceRecognition, '/facerecognition')

# Error handler for unhandled routes
@app.errorhandler(404)
def not_found(e):
    return {"message": "Resource not found"}, 404

if __name__ == '__main__':
    app.run(debug=True)
