#Utilities & PLumbing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#Preprocessing utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#MachineLearning components
import tensorflow as tf
from keras import layers
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt

#Object needed for relative paths
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
logdir = os.path.join(ROOT_DIR, 'Project_1', "logs")

# Define a function with parameter and a model architecture over which the KerasTuner can run, returns an instance of a model for it
def build_model(hp):
    # ------------------- Variables, for ease of change
    layers_sizes = [256, 512, ]
    activations = ['relu', ]
    dropout = [0.2, 0.3, ]
    losses = ["categorical_crossentropy", ]
    optimizers = ["adam", ]
    classifier = ["softmax", ]
    # ------------------- Architecture of the model itself (Which kind of layer where is fixed here)
    model = tf.keras.Sequential()
    model.add(layers.Dense(units=hp.Choice("units", layers_sizes), activation=hp.Choice("activation", activations), input_shape=(34,)))
    model.add(layers.Dropout(hp.Choice("dropout", dropout)))
    model.add(layers.Dense(units=hp.Choice("units2", layers_sizes), activation=hp.Choice("activation2", activations)))
    model.add(layers.Dropout(hp.Choice("dropout2", dropout)))
    model.add(layers.Dense(7, activation=hp.Choice("classification", classifier)))
    model.compile(loss=hp.Choice("loss", losses), optimizer=hp.Choice("optimizers", optimizers), metrics=["mae", "acc"])
    return model

def run_project1():
# ------------------- Data was preprocessed using MongoDB, we just have to import and delete id from mongodb
    pd.set_option('display.max_columns', None)
    data = pd.read_json(path_or_buf=os.path.join(ROOT_DIR, 'Project_1', "Date_Fruit_Datasets", "Dates_Doubles.json"),
                        orient='records', lines=True)
    data.drop(columns=["_id"], inplace=True)

    '''
    # ------------------- Getting an overview of target categories as instructed
    expressionslist = data.Class.value_counts()
    print(data.shape)
    print(expressionslist)
    expressions = data.Class.value_counts(normalize=True).mul(100)  # Get % of Classes
    # ------------------- Plotting
    plt.style.use('_mpl-gallery-nogrid')
    colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(expressions)))
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(600 * px, 600 * px))
    ax.pie(expressions, colors=colors, radius=12, center=(16, 16),
           wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True, labels=expressions.index.values.tolist()
           , autopct="%.2f")

    ax.set(xlim=(0, 32), xticks=np.arange(1, 32),
           ylim=(0, 32), yticks=np.arange(1, 32))

    plt.show()
    '''
    #-------------------Preprocessing
    class_labels = {
        'DOKOL': 0,
        'SAFAVI': 1,
        'ROTANA': 2,
        'DEGLET': 3,
        'SOGAY': 4,
        'IRAQI': 5,
        'BERHI': 6
    }
    y = data["Class"]
    y = to_categorical(y.map(class_labels))
    x = data.drop(["Class"], axis=1)
    x = StandardScaler().fit_transform(x)

    #Splitting the Data, it is important to stratify according to classes to avoid sampling bias as there is unequal distribution of classes
    random_state = 42
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

# -------------------Keras-Tuner for Optimization, saves best model into special directory & logs everything for Tensorboard
    tuner = kt.RandomSearch(hypermodel=build_model,
                            objective="val_acc",
                            seed=random_state,
                            max_trials=15,
                            executions_per_trial=2,
                            overwrite=True,
                            directory=os.path.join(ROOT_DIR, 'Project_1', "model", "Tuner_Dump")
                            )
    tuner.search(x_train, y_train, epochs=25, validation_data=(x_test, y_test),
                 callbacks=[tf.keras.callbacks.TensorBoard(logdir)])
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(os.path.join(ROOT_DIR, 'Project_1', "model", "best_model"))
    best_model.summary
# -------------------
#TODO:Simple Model Architecture documentation with screenshots, exprimenting with architecture of model, finish documentation
'''

    
    custom_model = tf.keras.Sequential()
    custom_model.add(layers.Dense(512, activation='relu'))
    custom_model.add(layers.Dropout(0.3))
    custom_model.add(layers.Dense(7, 'softmax'))
    custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae', 'acc'])

    custom_model.fit(x_train, y_train, epochs=25,
                        validation_data=(x_test, y_test), verbose=2,
                        callbacks=callbacks)
                        '''