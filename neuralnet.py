import MetaTrader5 as mt5
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from keras import optimizers
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

import onnx
import tf2onnx

tf.random.set_seed(42)

if not mt5.initialize():  # This will open MT5 app in your pc
    print("initialize() failed, error code =",mt5.last_error())
    quit()

    
terminal_info = mt5.terminal_info()

data_path = terminal_info.data_path
dataset_path = data_path + "\\MQL5\\Files\\ONNX Datafolder"

    
if not os.path.exists(dataset_path):
    print("Dataset folder doesn't exist | Be sure you are referring to the correct path and the data is collected from MT5 side of things")
    quit()

class NeuralNetworkClass():
    def __init__(self, csv_name, target_column, batch_size=32):

    # Loading the dataset and storing to a variable Array        
        self.data = pd.read_csv(dataset_path+"\\"+csv_name)

        if self.data.empty:
            print(f"No such dataset or Empty dataset csv = {csv_name}")
            quit() # quit the program
        

        print(self.data.head()) # Print 5 first rows of a given data

        self.target_column = target_column
        # spliting the data into training and testing samples

        X = self.data.drop(columns=self.target_column).to_numpy() # droping the targeted column, the rest is x variables
        Y = self.data[self.target_column].to_numpy() # We convert data arrays to numpy arrays compartible with sklearn and tensorflow
                
        
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, Y, test_size=0.3, random_state=42) # splitting the data into training and testing samples 
        
        print(f"train x shape {self.train_x.shape}\ntest x shape {self.test_x.shape}")
                
        self.input_size = self.train_x.shape[-1] # obtaining the number of columns in x variable as our inputs
        
        self.output_size = 1 # We are solving for a regression problem we need to have a single output neuron
        
        self.batch_size = batch_size
        
        self.model = None # Object to store the model
        
        self.plots_directory = "Plots"
        self.models_directory = "Models"
                
        
    def BuildNeuralNetwork(self, activation_function='relu', neurons = 10):

        # Create a Feedforward Neural Network model
        self.model = keras.Sequential([
            keras.layers.Input(shape=(self.input_size,)),  # Input layer
            keras.layers.Dense(units=neurons, activation=activation_function, activity_regularizer=l2(0.01), kernel_initializer="he_uniform"),  # Hidden layer with an activation function
            keras.layers.Dense(units=self.output_size, activation='linear', activity_regularizer=l2(0.01), kernel_initializer="he_uniform")  
        ])

        # Print a summary of the model's architecture.
        self.model.summary()



    def train_network(self, epochs=100, learning_rate=0.001, loss='mean_squared_error'):

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Early stoppage mechanism | stop training when there is no major change in loss in the last to epochs, defined by the variable patience

        adam = optimizers.Adam(learning_rate=learning_rate) # Adam optimizer >> https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    
        # Compile the model: Specify the loss function, optimizer, and evaluation metrics.
        self.model.compile(loss=loss, optimizer=adam, metrics=['mae'])    

        # One hot encode the validation and train target variables
         
        validation_y = self.test_y
        y = self.train_y

        history = self.model.fit(self.train_x, y, epochs=epochs, batch_size=self.batch_size, validation_data=(self.test_x, validation_y), callbacks=[early_stopping], verbose=2)
        
        if not os.path.exists(self.plots_directory): #create plots path if it doesn't exist for saving the train-test plots
            os.makedirs(self.plots_directory)
        
        # save the loss and validation loss plot
        
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        title = 'Training and Validation Loss Curves'
        plt.title(title)
        plt.savefig(fname=f"{self.plots_directory}\\"+title)

        
        # use the trained model to make predictions on the trained data 
        
        pred = self.model.predict(self.train_x)

        acc = metrics.r2_score(self.train_y, pred)

        # Plot actual & pred
        count = [i*0.1 for i in range(len(self.train_y))]

        title = f'MLP {self.target_column} - Train'
        
        # Saving the plot containing information about predictions and actual values
        
        plt.figure(figsize=(7, 5))
        plt.plot(count, self.train_y, label = "Actual")
        plt.plot(count, pred,  label = "forecast")
        plt.xlabel('Actuals')
        plt.ylabel('Preds')
        plt.title(title+f" | Train acc={acc}")
        plt.legend()
        plt.savefig(fname=f"{self.plots_directory}\\"+title)    

        self.model.save(f"Models\\MLP.REG.{self.target_column}.{self.data.shape[0]}.h5") #saving the model in h5 format, just for the record | not important really
        self.saveONNXModel()


    def saveONNXModel(self, folder="ONNX Models"):
        
        path = data_path + "\\MQL5\\Files\\" + folder 
        
        if not os.path.exists(path): # create this path if it doesn't exist
            os.makedirs(path)
        
        onnx_model_name = f"MLP.REG.{self.target_column}.{self.data.shape[0]}.onnx"
        path +=  "\\" + onnx_model_name
        
        
        loaded_keras_model = load_model(f"Models\\MLP.REG.{self.target_column}.{self.data.shape[0]}.h5") 
        
        onnx_model, _ = tf2onnx.convert.from_keras(loaded_keras_model, output_path=path)

        onnx.save(onnx_model, path )
        
        print(f'Saved model to {path}')
        
    
    def test_network(self):
        # Plot actual & pred
        
        count = [i*0.1 for i in range(len(self.test_y))]

        title = f'MLP {self.target_column} - Test'
        

        pred = self.model.predict(self.test_x)

        acc = metrics.r2_score(self.test_y, pred)

        
        # Saving the plot containing information about predictions and actual values
        
        plt.figure(figsize=(7, 5))
        plt.plot(count, self.test_y, label = "Actual")
        plt.plot(count, pred,  label = "forecast")
        plt.xlabel('Actuals')
        plt.ylabel('Preds')
        plt.title(title+f" | Train acc={acc}")
        plt.legend()
        plt.savefig(fname=f"{self.plots_directory}\\"+title)    
        
        if not os.path.exists(self.plots_directory): #create plots path if it doesn't exist for saving the train-test plots
            os.makedirs(self.plots_directory)
        
        plt.savefig(fname=f"{self.plots_directory}\\"+title)    
        
        return acc


csv_name = "EURUSD.PERIOD_H1.10000.targ=CLOSE.csv"

nn = NeuralNetworkClass(csv_name=csv_name, target_column="CLOSE", batch_size=32)

nn.BuildNeuralNetwork(activation_function="relu", neurons=10)
nn.train_network(epochs=50, learning_rate=0.01, loss='mse')


print("Test accuracy =",nn.test_network())


mt5.shutdown() # This closes the program