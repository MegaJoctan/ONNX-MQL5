//+------------------------------------------------------------------+
//|                                                     ONNX mt5.mq5 |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"


#include <MALE5\preprocessing.mqh>
CPreprocessing<vectorf, matrixf> *norm_x;

input uint start_bar = 100;
input uint total_bars = 1000;

input int bb_period = 20;
input int atr_period = 13;

input norm_technique NORM =   NORM_STANDARDIZATION;
string csv_name_ = "";
string normparams_folder = "ONNX Normparams\\";
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

  if (!LoadNormParams())
    {
      Print("Failed to load Norm params");
      return INIT_FAILED;
    }
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
   
   
  }
//+------------------------------------------------------------------+
//| This Function obtains the total variable sum of historical data  |
//| starting at the bar located at the start variable                |
//+------------------------------------------------------------------+
bool LoadNormParams()
 {
    vectorf min = {}, max ={}, mean={} , std = {};
    
    csv_name_ = Symbol()+"."+EnumToString(Period())+"."+string(total_bars);
    
    switch(NORM)
      {
       case  NORM_MEAN_NORM:
          
          mean = ReadCsvVector(normparams_folder+csv_name_+".mean_norm_scaler.mean.csv"); //--- Loading the mean
          min = ReadCsvVector(normparams_folder+csv_name_+".mean_norm_scaler.min.csv"); //--- Loading the min 
          max = ReadCsvVector(normparams_folder+csv_name_+".mean_norm_scaler.max.csv"); //--- Loading the max
          
          
          norm_x = new CPreprocessing<vectorf,matrixf>(max, mean, min);
           
          if (mean.Sum()<=0 && min.Sum()<=0 && max.Sum() <=0)
              return false;  

         break;
         
       case NORM_MIN_MAX_SCALER:
          
          min = ReadCsvVector(normparams_folder+csv_name_+".min_max_scaler.min.csv"); //--- Loading the min
          max = ReadCsvVector(normparams_folder+csv_name_+".min_max_scaler.max.csv"); //--- Loading the max  
       
           
          norm_x = new CPreprocessing<vectorf,matrixf>(max, min);
          
          
          if (min.Sum()<=0 && max.Sum() <=0)
            return false;
            
          break;
          
       case NORM_STANDARDIZATION:
          
          mean = ReadCsvVector(normparams_folder+csv_name_+".standardization_scaler.mean.csv"); //--- Loading the mean
          std = ReadCsvVector(normparams_folder+csv_name_+".standardization_scaler.std.csv"); //--- Loading the std
         
             
           norm_x = new CPreprocessing<vectorf,matrixf>(mean, std, NORM_STANDARDIZATION);
            
          if (mean.Sum()<=0 && std.Sum() <=0)
            return false;
            
          break;
      }
      
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

string csv_headerArr[];

vectorf ReadCsvVector(string file_name, string delimiter=",")
  {
   matrix mat_ = {};

   int rows_total=0;

   int handle = FileOpen(file_name,FILE_READ|FILE_CSV|FILE_ANSI,delimiter);

   ResetLastError();
   
   if(handle == INVALID_HANDLE)
     {
      printf("Invalid %s handle Error %d ",file_name,GetLastError());
      Print(GetLastError()==0?" TIP | File Might be in use Somewhere else or in another Directory":"");
     }

   else
     {
      int column = 0, rows=0;

      while(!FileIsEnding(handle) && !IsStopped())
        {
         string data = FileReadString(handle);

         //---
         if(rows ==0)
           {
            ArrayResize(csv_headerArr,column+1);
            csv_headerArr[column] = data;
           }

         if(rows>0)  //Avoid the first column which contains the column's header
            mat_[rows-1,column] = (double(data));

         column++;

         //---

         if(FileIsLineEnding(handle))
           {
            rows++;

            mat_.Resize(rows,column);

            column = 0;
           }
        }

      rows_total = rows;

      FileClose(handle);
     }
   
   
   mat_.Resize(rows_total-1,mat_.Cols());
   vectorf vec = {};
   vec.Assign(mat_);
   
   return(vec);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
