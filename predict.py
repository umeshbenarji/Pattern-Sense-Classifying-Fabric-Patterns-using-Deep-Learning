
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

model = tf.keras.models.load_model('models/pattern_classifier.h5')
class_names = ['floral', 'geometric', 'polka_dots', 'stripes']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    print(f"Predicted pattern: {predicted_class}")

if __name__ == "__main__":
    predict_image(sys.argv[1])
