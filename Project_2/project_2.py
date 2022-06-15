# Utilities & Plumbing
import os
import random
import morse_talk as mtalk
import numpy as np
import pandas as pd
import json

# Keras for modeling
from keras.models import Model
from keras.layers import Input, LSTM, Dense

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
datadir = os.path.join(ROOT_DIR, 'Project_2', "data")


def create_morse_dataset():  # Creates a seed from Alphabet then generates database of random words according to specs
    # ----------------Parameters
    seed_amount = 10
    word_length = 8
    db_entries = 500
    # ----------------Data Structures
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "V", "W",
                "X", "Y", "Z", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    morse_seed = []
    morse_data = []
    alphabet_data = []

    # ----------------Instructions
    seed = random.choices(alphabet, k=seed_amount, )
    for item in seed:
        translation = mtalk.encode(item)
        morse_seed.append(translation)

    while len(morse_data) < db_entries:
        word = ""
        while len(word) < word_length:
            word = word + random.choice(seed)
        morse_data.append(mtalk.encode(word))
        alphabet_data.append(word)

    # ----------------Convert to DataFrames and save after creating directory
    try:
        os.mkdir(os.path.join(ROOT_DIR, 'Project_2', "data"))
    except:
        pass
    alphabet_data = pd.DataFrame(alphabet_data)
    morse_data = pd.DataFrame(morse_data)
    alphabet_data.to_json(os.path.join(datadir, 'alphabet_data.json'), indent=2)
    morse_data.to_json(os.path.join(datadir, 'morse_data.json'), indent=2)

def load_data():    # Load created data from directory to
    alphabet_data = pd.read_json(os.path.join(datadir, 'alphabet_data.json'))
    morse_data = pd.read_json(os.path.join(datadir, 'morse_data.json'))

    return alphabet_data, morse_data


def run_project2():
    #create_morse_dataset()
    alphabet_data, morse_data = load_data()
    print(alphabet_data)
    print(morse_data)
