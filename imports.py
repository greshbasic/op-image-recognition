import matplotlib.pyplot as plt
import numpy as np
import PIL

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from random import randint

characters = [
        "Ace", "Akainu", "Brook", "Chopper", "Crocodile", "Franky", 
        "Jinbei", "Blackbeard", "Law", "Luffy", "Mihawk", "Nami", 
        "Rayleigh", "Robin", "Sanji", "Shanks", "Usopp", "Zoro"
    ]