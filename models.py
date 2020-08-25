import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm
from IPython.display import display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def mult_reg(x, y):
    reg = LinearRegression().fit(x, y)  # To get coefficients
    coeffs = reg.coef_
    # To get intercept
    b = reg.intercept_
    # To evaluate performance
    score = reg.score
    return coeffs, b, score

def create_model(lr, num_layers, input_shape):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(input_shape,)))
    for i in range (num_layers):
        model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer= opt, loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def plot_model(history, lr, num_layers):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_rmse = history.history['root_mean_squared_error']
    val_rmse = history.history['val_root_mean_squared_error']
    print('final training loss: ', train_loss[-1], ' final val loss: ', val_loss[-1])
    plt.plot([i for i in range(len(train_loss))], train_loss, color='r', label='train')
    plt.plot([i for i in range(len(val_loss))], val_loss, color='b', label='val')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('mse')
    plt.title('Loss vs Epochs, LR: {}, Num_Layers: {}'.format(lr, num_layers))
    plt.show()
    plt.clf()
    return val_loss[-1]

def grid_hyperparamsearch(lrs, num_layers, XR_train_R, YR_train_R, XR_val, YR_val, XR_test, YR_test, numepochs=200):
    models = dict()
    model_scores = dict()
    for lr in lrs:
        for num_layer in num_layers:
            model = create_model(lr, num_layer, XR_train_R.shape [1])
            history = model.fit(XR_train_R, YR_train_R, validation_data= (XR_val, YR_val), epochs=numepochs)
            model_scores [(lr, num_layer)] = plot_model(history, lr, num_layer)
            models [(lr, num_layer)] = model

    best_hyperparam_pair = sorted(model_scores.items(), key=lambda x: x[1], reverse=False)[0][0]
    print(model_scores)
    print(best_hyperparam_pair)

    best_model = models[best_hyperparam_pair]
    results = best_model.evaluate(XR_test, YR_test)
    print("test loss: {}, test RMSE: {}".format(results[0], results[1]))
    return best_model, results

class MyModel(tf.keras.Model):
#### Constructor Function#################
  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)