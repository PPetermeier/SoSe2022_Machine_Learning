import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers

from sklearn.model_selection import train_test_split


def run():
# ------------------- Data was preprocessed using MongoDB, we just have to import and delete id from mongodb
    pd.set_option('display.max_columns', None)
    data = pd.read_json(path_or_buf='C:\Repositories\SoSe2022_Machine_Learning\Project_1\Date_Fruit_Datasets\Dates_Doubles.json',
                        orient='records', lines=True)
    data.drop(columns=["_id"], inplace=True)

    # ------------------- Getting an overview of target categories as instructed
    expressionslist = data.Class.value_counts()
    expressions = data.Class.value_counts(normalize=True).mul(100)  # Get % of Classes

    plt.style.use('_mpl-gallery-nogrid')  # Visualize
    colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(expressions)))
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(600 * px, 600 * px))
    ax.pie(expressions, colors=colors, radius=12, center=(16, 16),
           wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True, labels=expressions.index.values.tolist()
           , autopct="%.2f")

    ax.set(xlim=(0, 32), xticks=np.arange(1, 32),
           ylim=(0, 32), yticks=np.arange(1, 32))

    plt.show()
    # -------------------Preprocessing

    stringlookuplayer = layers.StringLookup(output_mode="int") # Encode Classes
    stringlookuplayer.adapt(data["Class"])
    data["Class"] = stringlookuplayer(data["Class"])

    #print(stringlookuplayer.get_vocabulary())
    #print(data["Class"])
    processed_data = tf.keras.utils.normalize(data)
    x = processed_data.drop("Class", axis=1)
    y = processed_data["Class"]
    #Splitting the Data, it is important to stratify according to classes to avoid sampling bias as there is unequal distribution of classes
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)




