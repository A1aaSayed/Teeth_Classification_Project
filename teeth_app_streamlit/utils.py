import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

IMG_SIZE = (224, 224)

def load_and_process(image):
  image = image.resize(IMG_SIZE)
  image = np.array(image)

  if image.shape[-1] == 4:  # RGBA → RGB
    image = image[:, :, :3]

  image = preprocess_input(image)
  image = np.expand_dims(image, axis=0)
  return image

def predict_image(model, image, class_names):
  processed_image = load_and_process(image)
  preds = model.predict(processed_image)[0]

  predicted_class = class_names[np.argmax(preds)]
  probabilities = (preds * 100).round(2)

  return predicted_class, probabilities
