import numpy as np
import cv2
import os
import io
import contextlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

os.environ["PYTHONIOENCODING"] = "UTF-8"

tiles_folder = "tiles"
training_folder = "processed_tiles"


def prepare_images_for_cnn(tiles_folder):
    X = []
    y = []

    for img_name in os.listdir(tiles_folder):
        if img_name.endswith(".png"):
            img_path = os.path.join(tiles_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            img = img / 255.0
            X.append(np.expand_dims(img, axis=-1))

            label = int(img_name.split()[0].split("(")[0])
            y.append(label)

    y = to_categorical(y, num_classes=14)

    return np.array(X), np.array(y)


X, y = prepare_images_for_cnn(training_folder)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(14, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=2)

model.save("my_digit_model.keras")


def detect_number_in_image(image_path, keras_model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    with contextlib.redirect_stdout(io.StringIO()) as f:
        prediction = keras_model.predict(img)

    return np.argmax(prediction)


keras_model = load_model("my_digit_model.keras")
predictions = []

for img_name in sorted(os.listdir("tiles"), key=lambda x: int(x.split("_")[0])):
    if img_name.endswith(".png"):
        img_path = os.path.join("tiles", img_name)
        prediction = detect_number_in_image(img_path, keras_model)
        name_without_number = img_name.split("_")[1].split(".")[0]
        predictions.append(f"{prediction}_{name_without_number}")

print("CNN:", predictions)
