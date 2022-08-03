from tensorflow import keras
import numpy as np
from pathlib import Path
THIS_DIR = Path(__file__).parent
model_path = THIS_DIR / 'model/mnist_model.h5'

def load_model():
    return keras.models.load_model(str(model_path))

def predict_digits(digits, model):
    res = model.predict(np.array(digits))
    vals = []
    for x in res:
        vals.append(x.argmax())
    return vals