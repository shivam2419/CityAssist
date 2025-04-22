import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2, os

def classify_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found at {img_path}")
    # Load trained model
    model = tf.keras.models.load_model("C:\\Django\\CityAssist\\City-Assist\\api\\road_classifier_cnn.h5")

    # Load & preprocess image
    img = image.load_img(img_path, target_size=(128, 128))  # Change path
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img)[0][0]
    result = f"Clean Road" if prediction < 0.5 else "Dirty Road"
    return result




# You have to wirte the <script> code in your project, in jsx or whateveer you are using, as k chatgpt to modify the code accordingly.