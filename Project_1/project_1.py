# Utilities & PLumbing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Preprocessing utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# MachineLearning components
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras import layers
import keras_tuner as kt




# Objects needed for relative paths
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
logdir = os.path.join(ROOT_DIR, 'Project_1', "logs")


def read_data(): #Reads .json with relative path & drops ids from MongoDB
    pd.set_option('display.max_columns', None)
    data = pd.read_json(path_or_buf=os.path.join(ROOT_DIR, 'Project_1', "Date_Fruit_Datasets", "Dates_Doubles.json"),
                        orient='records', lines=True)
    data.drop(columns=["_id"], inplace=True)
    return data


def get_overview_and_plot(data): #Getting an overview of the data, plot class distribution
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


def preprocessing(data): #Create Dict of target classes, map onto those, scale inputs, create test-train-split with stratify
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

    # Splitting the Data, it is important to stratify according to classes to avoid sampling bias as there is unequal distribution of classes
    random_state = 42
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    return x_train, x_test, y_train, y_test

def run_first_model(x_train, x_test, y_train, y_test): # First model for documentation purposes
    logdir_first_model = os.path.join(ROOT_DIR, 'Project_1', "logs", "first_model")

    first_model = tf.keras.Sequential()
    first_model.add(layers.Dense(512, activation='relu'))
    first_model.add(layers.GaussianDropout(0.3))
    first_model.add(layers.Dense(7, 'softmax'))
    first_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["categorical_accuracy"])

    first_model.fit(x_train, y_train, epochs=25,
                    validation_data=(x_test, y_test), verbose=1,
                    callbacks=[tf.keras.callbacks.TensorBoard(logdir_first_model)])
    first_model.summary()
    first_model.save(os.path.join(ROOT_DIR, 'Project_1', "model", "first_model"))



def build_model(hp): #Builds a model for the KerasTuner to run over together with search-space definition. Architecture & Hyperparameter
    # ------------------- Variables, for ease of change
    layers_sizes = [128, 256, 512, ]
    activations = ['relu', ]
    noise = [0.2, 0.3, 0.4, ]
    losses = ["categorical_crossentropy", ]
    optimizers = ["adam", ]
    classifier = ["softmax", ]
    # ------------------- Architecture of the model itself (Which kind of layer where is fixed here)
    model = tf.keras.Sequential()
    model.add(layers.Dense(units=hp.Choice("units", layers_sizes), activation=hp.Choice("activation", activations),
                           input_shape=(34,)))
    model.add(layers.GaussianNoise(hp.Choice("noise", noise)))
    model.add(layers.Dense(units=hp.Choice("units2", layers_sizes), activation=hp.Choice("activation2", activations)))
    model.add(layers.GaussianNoise(hp.Choice("noise2", noise)))
    model.add(layers.Dense(7, activation=hp.Choice("classification", classifier)))
    model.compile(loss=hp.Choice("loss", losses), optimizer=hp.Choice("optimizers", optimizers), metrics=["mae", "acc"])
    return model


def run_kerasTuner(x_train, x_test, y_train, y_test): #Options here are how to search, not where. Saves best model with given parameters.
    random_state = 42
    tuner = kt.BayesianOptimization(hypermodel=build_model,
                            objective="val_acc",
                            seed=random_state,
                            max_trials=50,
                            executions_per_trial=2,
                            overwrite=True,
                            directory=os.path.join(ROOT_DIR, 'Project_1', "model", "Tuner_Dump")
                            )
    tuner.search(x_train, y_train, epochs=25, validation_data=(x_test, y_test),
                 callbacks=[tf.keras.callbacks.TensorBoard(logdir)])
    tuner.results_summary()
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(os.path.join(ROOT_DIR, 'Project_1', "model", "best_model"))
    print("BestModelSummary:")
    best_model.summary

def project_summary(): #Recalls existing results, this is a dummy method for first orientation
    print("Summary of the first model:")
    first_model = tf.keras.models.load_model(
        os.path.join(ROOT_DIR, 'Project_1', "model", "first_model")
    )
    first_model.summary()
    print("")
    print("")

    print("Summary of the best model found:")
    best_model = tf.keras.models.load_model(
        os.path.join(ROOT_DIR, 'Project_1', "model", "best_model")
    )
    best_model.summary()
    print("Rest of the code is commented out to save resources, just change function calls in project_1.py")


def run_project1(): #Uncomment to run first model of tuner, data overview also probably not interesting
    data = read_data()
    #get_overview_and_plot(data)
    x_train, x_test, y_train, y_test = preprocessing(data)
    #run_first_model(x_train, x_test, y_train, y_test)
    #run_kerasTuner(x_train, x_test, y_train, y_test)
    project_summary()
