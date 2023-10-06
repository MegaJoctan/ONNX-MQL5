import MetaTrader5 as mt5
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


if not mt5.initialize():  # This will open MT5 app in your pc
    print("initialize() failed, error code =",mt5.last_error())
    quit()

    
terminal_info = mt5.terminal_info()

data_path = terminal_info.data_path
dataset_path = data_path + "\\MQL5\\Files\\ONNX Datafolder"

    
if not os.path.exists(dataset_path):
    print("Dataset folder doesn't exist | Be sure you are referring to the correct path and the data is collected from MT5 side of things")
    quit()


# Loading the dataset 

csv_name = "EURUSD.PERIOD_H1.1000.targ=MOVEMENT.csv"

data = pd.read_csv(dataset_path+"\\"+csv_name)

if data.empty:
    print(f"No such dataset or Empty dataset csv = {csv_name}")
    quit() # quit the program
    

input_size = data.shape[1]

print("Input size = ",input_size)


print(data.head())





mt5.shutdown() # This closes the program