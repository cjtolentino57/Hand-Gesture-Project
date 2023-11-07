import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('CNN_Model.h5')

# Define the gesture classes
gesture_names = {0: 'palm',
 1: 'l',
 2: 'fist',
 3: 'fist_moved',
 4: 'thumb',
 5: 'index',
 6: 'ok',
 7: 'palm_moved',
 8: 'c',
 9: 'down'}

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Get a frame from the webcam
    ret, frame = cap.read()

    # Define a ROI (Region of Interest) that you want to classify
    x, y, w, h = 200, 200, 200, 200
    roi = frame[y:y+h, x:x+w]

    # Resize the ROI to 28x28
    roi = cv2.resize(roi, (120, 320))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values
    gray = gray / 255

    # Add a dimension to the frame for the model
    gray = np.expand_dims(gray, 0)
    gray = np.expand_dims(gray, -1)

    # Get the predictions from the model
    predictions = model.predict(gray)[0]

    # Get the index of the highest prediction
    max_index = np.argmax(predictions)

    # Get the name of the gesture
    gesture = gesture_names[max_index]

    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the gesture on the frame
    cv2.putText(frame, gesture, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Webcam', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()