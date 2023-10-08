import MetaTrader5 as mt5
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import EarlyStopping



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
        

        print(self.data.head())

        self.target_column = target_column
        # spliting the data into training and testing samples

        X = self.data.drop(columns=self.target_column).to_numpy() # droping the targeted column, the rest is x variables
        Y = self.data[self.target_column].to_numpy() # We convert data arrays to numpy arrays compartible with sklearn and tensorflow

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, Y, test_size=0.3, random_state=42)

        self.input_size = X.shape[1]

        self.classes = np.unique(Y) # Get the avaiable classes from a target variable
        
        self.output_size = len(self.classes) # count those number of classes
        
        self.batch_size = batch_size
        
        self.model = None # Object to store the model
                

    def BuildNeuralNetwork(self, activation_function='sigmoid', neurons = 10, dropout_rate=0.5):

        self.model = Sequential()

        # Input layer: Specify the input shape (number of features) in the 'input_shape' argument.
        self.model.add(Dense(units=64, activation=activation_function, input_shape=(self.input_size,), kernel_initializer='he_uniform'))

        # Hidden layer: Add as many hidden layers as you need.
        self.model.add(Dense(units=32, activation=activation_function, kernel_initializer='he_uniform'))
        self.model.add(Dropout(dropout_rate)) 
        
        # Hidden layer: Add as many hidden layers as you need.
        '''
        self.model.add(Dense(units=10, activation=activation_function, kernel_initializer='he_uniform'))
        self.model.add(Dropout(dropout_rate)) 
        '''
        # Output layer: Specify the number of output neurons (units) and activation function.
        self.model.add(Dense(units=self.output_size, activation='softmax', kernel_initializer='he_uniform'))

        # Print a summary of the model's architecture.
        self.model.summary()


    def train_network(self, epochs=100, learning_rate=1e-3, loss='binary_crossentropy'):

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        adam = optimizers.Adam(learning_rate=learning_rate)
    
        # Compile the model: Specify the loss function, optimizer, and evaluation metrics.
        self.model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])    

        # One hot encode the validation and train target variables
         
        validation_y = to_categorical(self.test_y, num_classes=len(self.classes))
        y = to_categorical(self.train_y, num_classes=len(self.classes))
        
        self.model.fit(self.train_x, y, epochs=epochs, batch_size=self.batch_size, validation_data=(self.test_x, validation_y), callbacks=[early_stopping], verbose=2) #Training the model
        
        forecast = self.model.predict(self.train_x)
        actual = self.train_y
        
        #obtain the outcome
        
        forecast = np.argmax(forecast, axis=1) # Extracting a predicted binary

        # Compute confusion matrix
        cm = metrics.confusion_matrix(y_true=actual, y_pred=forecast, labels= self.classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')
        title = f'MLP-NN {self.target_column} Confusion Matrix - Train'
        plt.title(title)
        
        directory = "Plots"

        if not os.path.exists(directory): #create plots path if it doesn't exist for saving the train-test plots
            os.makedirs(directory)
        
        plt.savefig(fname=f"{directory}\\"+title)    
        
        
        self.model.save(f"Models\\lstm-pat.{self.target_column}.h5") 


    def _acc(self, confusion_matrix):
        diagonal_sum = np.trace(confusion_matrix)
        total_sum = np.sum(confusion_matrix)
        overall_accuracy = diagonal_sum / total_sum
        return overall_accuracy




csv_name = "EURUSD.PERIOD_H1.1000.targ=MOVEMENT.csv"

nn = NeuralNetworkClass(csv_name=csv_name, target_column="MOVEMENT")
nn.BuildNeuralNetwork(activation_function="tanh", neurons=10)
nn.train_network(epochs=100, learning_rate=1e-4)


mt5.shutdown() # This closes the program