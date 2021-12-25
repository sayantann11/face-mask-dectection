import numpy as np
import matplotlib.pyplot as plt
import os
from imutils import paths

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__=="__main__":
	directory = r"dataset"
	categories = ["with_mask", "without_mask"]

	print("[INFO] Loading Images...")
	data = []
	labels = []

	for category in categories:
		path = os.path.join(directory, category)
		for img in os.listdir(path):
			img_path = os.path.join(path, img)
			image = load_img(img_path, target_size=(224, 224))
			image = img_to_array(image)
			image = preprocess_input(image)

			# Append the image and labels to the dedicated lists.
			data.append(image)
			labels.append(category)

	print("[INFO] All Images Loaded")

	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)
	labels = to_categorical(labels)
	# print(labels[:10])

	data = np.array(data, dtype="float32")
	labels = np.array(labels)

	(trainX, testX, trainY, testY) = train_test_split(data, labels,
		test_size=0.20, stratify=labels, random_state=20)

	# Data Augmentation
	aug = ImageDataGenerator(rotation_range=40,
							 width_shift_range=0.2,
							height_shift_range=0.2,
							shear_range=0.2,
							zoom_range=0.2,
							horizontal_flip=True,
							fill_mode='nearest')

	# load the MobileNetV2 network, ensuring the head FC layer sets are
	# left off
	baseModel = MobileNetV2(weights="imagenet", include_top=False,
		input_tensor=Input(shape=(224, 224, 3)))

	# construct the head of the model that will be placed on top of the
	# the base model
	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
	headModel = Flatten()(headModel)
	headModel = Dense(128, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(32, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)

	# place the head FC model on top of the base model (this will become
	# the actual model we will train)
	model = Model(inputs=baseModel.input, outputs=headModel)

	# loop over all layers in the base model and freeze them so they will
	# *not* be updated during the first training process
	for layer in baseModel.layers:
		layer.trainable = False

	# LR = 1e-4
	epochs = 20
	BS = 32
	# compile our model
	print("[INFO] Compiling model...")
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	# train the head of the network
	print("[INFO] Training head...")
	H = model.fit(
		aug.flow(trainX, trainY, batch_size=BS),
		steps_per_epoch=len(trainX) // BS,
		validation_data=(testX, testY),
		validation_steps=len(testX) // BS,
		epochs=epochs)

	# make predictions on the testing set
	print("[INFO] Evaluating network...")
	predIdxs = model.predict(testX, batch_size=BS)

	# for each image in the testing set we need to find the index of the
	# label with corresponding largest predicted probability
	predIdxs = np.argmax(predIdxs, axis=1)

	# show a nicely formatted classification report
	print(classification_report(testY.argmax(axis=1), predIdxs,
		target_names=lb.classes_))

	# serialize the model to disk
	print("[INFO] saving mask detector model...")
	model.save("mask_detector.model", save_format="h5")

	# plot the training loss and accuracy
	N = epochs
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig("plot.png")
