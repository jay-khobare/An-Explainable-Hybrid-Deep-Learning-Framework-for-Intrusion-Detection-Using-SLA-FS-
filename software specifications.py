import sys
import tensorflow as tf
import keras
import sklearn
import pandas as pd
import numpy as np
import matplotlib
import shap
import lime

from importlib.metadata import version

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("SHAP:", shap.__version__)
print("LIME:", version("lime"))