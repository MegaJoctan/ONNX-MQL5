import MetaTrader5 as mt5
import os

if not mt5.initialize():  # This will open MT5 app in your pc
    print("initialize() failed, error code =",mt5.last_error())
    quit()

    
terminal_info = mt5.terminal_info()

data_path = terminal_info.data_path
dataset_path = data_path + "\\MQL5\\Files\\ONNX Datafolder"

    
if not os.path.exists(dataset_path):
    print("Dataset folder doesn't exist | Be sure you are referring to the correct path and the data is collected from MT5 side of things")
    quit()







mt5.shutdown() # This closes the program