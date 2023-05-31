import cv2
import numpy as np
from keras.models import load_model

# Load the trained leaf detection model
model = load_model('leaf_model_augmented.h5')

# Set the dimensions for resizing the webcam feed
frame_width = 640
frame_height = 480

# Set the image dimensions for the model
image_size = (64, 64)

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

# Define the class labels for leaf detection
class_labels = ['Azadirachta Indica (Neem)',
                'Citrus Limon (Lemon)',
                'Ficus Religiosa (Peepal Tree)',
                'Murraya Koenigii (Curry)',
                'Ocimum Tenuiflorum (Tulsi)']

# Define the medical properties associated with each class label
medical_properties = [
    'Azadirachta indica (Neem) is known for its antibacterial, antifungal, antiviral, and anti-inflammatory properties.',
    'Citrus limon (Lemon) is rich in antioxidants, has antimicrobial properties, and aids in digestion.',
    'Ficus religiosa (Peepal Tree) possesses anti-inflammatory, antioxidant, and antidiabetic properties.',
    'Murraya koenigii (Curry) leaves are known for their antioxidant, anti-inflammatory, and anti-diarrheal effects.',
    'Ocimum tenuiflorum (Tulsi) has antibacterial, antiviral, antifungal, and anti-inflammatory properties.'
]

# Variable to track if detection is active
detection_active = True

# Set the confidence threshold for prediction
confidence_threshold = 0.5

# Create a separate window for displaying medical properties
cv2.namedWindow('Medical Properties', cv2.WINDOW_NORMAL)
cv2.moveWindow('Medical Properties', 0, 0)

# Create a blank white image for displaying medical properties
properties_image = np.ones((200, 800), dtype=np.uint8) * 255

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if detection is active
    if detection_active:
        # Preprocess the frame
        resized_frame = cv2.resize(frame, image_size)
        normalized_frame = resized_frame / 255.0
        input_data = np.expand_dims(normalized_frame, axis=0)

        # Make predictions on the frame
        predictions = model.predict(input_data)
        predicted_class_index = np.argmax(predictions)
        confidence = predictions[0][predicted_class_index]

        # If a leaf is detected with sufficient confidence, display the class label and medical properties
        if confidence >= confidence_threshold:
            predicted_class_label = class_labels[predicted_class_index]
            predicted_medical_properties = medical_properties[predicted_class_index]

            # Display the class label in the camera feed
            cv2.putText(frame, 'Leaf Detected: ' + predicted_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

            # Update the medical properties in the separate window
            properties_image = np.ones((200, 800), dtype=np.uint8) * 255
            cv2.putText(properties_image, 'Medical Properties:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(properties_image, predicted_medical_properties, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2)
            cv2.imshow('Medical Properties', properties_image)

            # Stop the detection process after a leaf is detected
            detection_active = False
        else:
            # Reset the medical properties to a blank image
            properties_image = np.ones((200, 800), dtype=np.uint8) * 255
            cv2.putText(properties_image, 'No Leaf Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.imshow('Medical Properties', properties_image)

    # Display the camera feed
    cv2.imshow('Leaf Detection', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Start detection on 'S' key press
    if key == ord('s'):
        detection_active = True

    # Exit the loop on 'Q' key press
    if key == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
