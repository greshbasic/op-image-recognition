import matplotlib.pyplot as plt
import numpy as np
import PIL

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from random import randint

characters = [
        "Ace", "Akainu", "Blackbeard", "Brook", "Chopper", "Crocodile", "Franky", 
        "Jinbei", "Law", "Luffy", "Mihawk", "Nami", 
        "Rayleigh", "Robin", "Sanji", "Shanks", "Usopp", "Zoro"
]