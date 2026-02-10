from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model('../results/models/best_base_classification_model.h5')

class_names = [
  'CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT'
]

IMG_SIZE = 224

def preprocess_image(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
  img = img / 255.0
  img = np.expand_dims(img, axis=0)
  return img

@app.route('/', methods=['GET', 'POST'])
def index():
  prediction = None
  confidence = None
  image_path = None
  probabilities = None

  if request.method == 'POST':
    file = request.files['image']
    if file:
      image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
      file.save(image_path)

      img = preprocess_image(image_path)
      preds = model.predict(img)[0]

      prediction = class_names[np.argmax(preds)]
      confidence = np.max(preds) * 100
      probabilities = (preds * 100).round(2).tolist()

  return render_template(
    'index.html',
    prediction=prediction,
    confidence=confidence,
    image_path=image_path,
    probabilities=probabilities,
    class_names=class_names
  )

if __name__ == '__main__':
  app.run(debug=True)
