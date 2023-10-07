#---------------------------- Importing Libraries-------------------------------------------------------
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Loading the Gender detection model. Check 'Binary_Image_Classification_Lab3_Part2.ipynb'
# For Storage issue I have not uploaded the model file, but you can find the model link in the README file.
# Download the model from readme and keep it in your directory.
gender_model = load_model('1st_model.h5')  

# Using OpenCV's Pre trained moddel "haarcascade" for detecting Front Face.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Code for predicting the gender of the image/live image
def predict_gender(face):

    face = cv2.resize(face, (224, 224))
    face = np.expand_dims(face, axis=0)
    
    prediction = gender_model.predict(face)
    predicted_class = (prediction >= 0.5).astype(int)[0][0]
    if predicted_class ==0:
        return 'Male'
    else:
        return 'Female'
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_roi = frame[y:y+h, x:x+w]
        gender = predict_gender(face_roi)
        cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Gender Classification', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
