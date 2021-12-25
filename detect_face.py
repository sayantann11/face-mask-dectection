import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

def detect_face(frame, faceNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (110, 170, 120))
	
	# set the input to the pre-trained deep learning network and obtain
	# the output predicted probabilities for each of the classes.
	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []

	# Looping over all the face detected.
	for i in range(0, detections.shape[2]):
		# confidence associated with the detection
		confidence = detections[0, 0, i, 2]

		if(confidence > 0.5):
			# Bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# To ensure that the bounding box is within the dimensions 
			# of the frame.
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			
			# ROI
			face = frame[startY:endY, startX:endX]
			cv2.imshow("Face", face)
			if(len(face)>0):
				# Since, cv2 reads the frame in BGR format, coverting
				# it into RGB
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				# Appending the face abd bbox in to their respective lists.
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	return (locs, faces)

def predict(faces, maskNet):
	preds = []
	if(len(faces) > 0):
		# Making predictions in batches for faster inference
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=16)

	return preds

if __name__=="__main__":
	# Loading face detection model.
	print("[INFO] Loading face detector model...")
	# prototxtPath = r"face_detector\deploy.prototxt"
	prototxtPath = os.path.join("face_detector", "deploy.prototxt")
	# weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
	weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# Loading the face mask trained model.
	print("[INFO] Loading face mask detector model...")
	maskNet = load_model("mask_detector.model")

	# Starting videostream
	print("[INFO] Starting video stream...")
	cap = cv2.VideoCapture(0)

	while(1):
		ret, frame = cap.read()
		(locs, faces) = detect_face(frame, faceNet)
		preds = predict(faces, maskNet)

		# Loop over the detected faces and their corresponding locations
		for (box, pred) in zip(locs, preds):
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# Determine the resultant class and display its probability
			if(mask > withoutMask):
				label = "{}: {:.2f}%".format("Mask", max(mask, withoutMask) * 100)
				color = (0, 255, 0)
			else:
				label = "{}: {:.2f}%".format("No Mask", max(mask, withoutMask) * 100)
				color = (0, 0, 255)
	
			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		# Total number of faces detected in the frame.
		cv2.putText(frame, "Total no. of face detected: {0}".format(len(locs)), 
			(20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)	

		# Output Frame
		cv2.imshow("Frame", frame)
		if(cv2.waitKey(1) & 0xFF == ord('q')):
			break

	cv2.destroyAllWindows()