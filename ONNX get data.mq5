//+------------------------------------------------------------------+
//|                                                ONNX get data.mq5 |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property description "Script for collecting data for a simple ONNX tutorial project"
#property version   "1.00"
#property script_show_inputs

#include <MALE5\preprocessing.mqh>
CPreprocessing<vectorf, matrixf> *norm_x;

input uint start_bar = 100;
input uint total_bars = 1000;

input int bb_period = 20;
input int atr_period = 13;

input norm_technique NORM =   NORM_STANDARDIZATION;

int bb_handle, 
    atr_handle;

vectorf BB_UP, 
        BB_MID, 
        BB_LOW, 
        ATR; 

string csv_header = "";
string x_vars = ""; //Independent variable names for saving them into a csv file
string csv_name_ = "";
string normparams_folder = "ONNX Normparams\\";

bool loaded_norm = false;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

   bb_handle = iBands(Symbol(),PERIOD_CURRENT, bb_period,0,2.0, PRICE_OPEN);
   atr_handle = iATR(Symbol(),PERIOD_CURRENT, atr_period);
   
//---

  if (LoadNormParams())
    {
      loaded_norm = true;
      Print("init | Normalization parameters was loaded");
    }
    
   
         
    matrixf dataset = GetData(start_bar, total_bars, true);
    
//    Print("Train data\n",dataset);
//    
//    matrixf dataset = GetData(0, 1);
//    
//    Print("Live data\n",dataset);
    
    
    while (CheckPointer(norm_x) != POINTER_INVALID)
      delete (norm_x);
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
           
          if (mean.Sum()<=0 && min.Sum()<=0 && max.Sum() <=0)
              return false;  
         
          norm_x = new CPreprocessing<vectorf, matrixf>(max, mean, min);
          
        break;
         
       case NORM_MIN_MAX_SCALER:
         
          min = ReadCsvVector(normparams_folder+csv_name_+".min_max_scaler.min.csv"); //--- Loading the min
          max = ReadCsvVector(normparams_folder+csv_name_+".min_max_scaler.max.csv"); //--- Loading the max           
          
          if (min.Sum()<=0 && max.Sum() <=0)
            return false;
            
          norm_x = new CPreprocessing<vectorf, matrixf>(max, min);
          
         break;
          
       case NORM_STANDARDIZATION:
         
          mean = ReadCsvVector(normparams_folder+csv_name_+".standardization_scaler.mean.csv"); //--- Loading the mean
          std = ReadCsvVector(normparams_folder+csv_name_+".standardization_scaler.std.csv"); //--- Loading the std
            
          if (mean.Sum()<=0 && std.Sum() <=0)
            return false;
          
          norm_x = new CPreprocessing<vectorf, matrixf>(mean, std, NORM_STANDARDIZATION);
          
          Print("Means ",norm_x.standardization_scaler.mean);
          Print("Stds ",norm_x.standardization_scaler.mean);

        
        break;
      }
      
   return true;
 }
//+------------------------------------------------------------------+
//| This Function obtains the total variable sum of historical data  |
//| starting at the bar located at the start variable                |
//+------------------------------------------------------------------+
matrixf GetData(uint start, uint total, bool train=false)
 {
   matrixf return_matrix(total, 4);
   vectorf target(total); //Vector for storing the target column
   
   ulong last_col;
   
   
    BB_UP.CopyIndicatorBuffer(bb_handle, 0, start, total);
    BB_LOW.CopyIndicatorBuffer(bb_handle, 1, start, total);
    BB_MID.CopyIndicatorBuffer(bb_handle, 2, start, total);
    ATR.CopyIndicatorBuffer(atr_handle, 0, start, total);
    
    return_matrix.Col(BB_UP, 0);
    return_matrix.Col(BB_LOW, 1);
    return_matrix.Col(BB_MID, 2);
    return_matrix.Col(ATR, 3);
    
    vectorf open, close;
    
    matrixf norm_params = {};
    
    csv_name_ = Symbol()+"."+EnumToString(Period())+"."+string(total_bars);
    
       
    if (!train)
       norm_x.Normalization(return_matrix); //Normalizing each new data while not on training
    
    else
     {
       x_vars = "BB_UP,BB_LOW,BB_MID,ATR";
       
       if (loaded_norm) 
         { 
           if (!norm_x.Normalization(return_matrix))
             Print("Failed to Normalize");  
             
             //for (ulong i=0; i<return_matrix.Rows(); i++)
             //  Print("[",i,"]",return_matrix.Row(i));
         }
       else
        {
          while (CheckPointer(norm_x) != POINTER_INVALID)
            delete (norm_x);
            
          norm_x = new CPreprocessing<vectorf, matrixf>(return_matrix, NORM);
        }
       
       
       //--- Saving the normalization prameters
       
       switch(NORM)
         {
          case  NORM_MEAN_NORM:
            
             //--- saving the mean
             
             norm_params.Assign(norm_x.mean_norm_scaler.mean);
             WriteCsv(normparams_folder+csv_name_+".mean_norm_scaler.mean.csv",norm_params,x_vars);
             
             //--- saving the min
             
             norm_params.Assign(norm_x.mean_norm_scaler.min);
             WriteCsv(normparams_folder+csv_name_+".mean_norm_scaler.min.csv",norm_params,x_vars);
             
             //--- saving the max
             
             norm_params.Assign(norm_x.mean_norm_scaler.max);
             WriteCsv(normparams_folder+csv_name_+".mean_norm_scaler.max.csv",norm_params,x_vars);
             
            break;
            
          case NORM_MIN_MAX_SCALER:
             
             //--- saving the min
             
             norm_params.Assign(norm_x.min_max_scaler.min);
             WriteCsv(normparams_folder+csv_name_+".min_max_scaler.min.csv",norm_params,x_vars);
             
             //--- saving the max
             
             norm_params.Assign(norm_x.min_max_scaler.max);
             WriteCsv(normparams_folder+csv_name_+".min_max_scaler.max.csv",norm_params,x_vars);
             
             
             break;
             
          case NORM_STANDARDIZATION:
             
             //--- saving the mean
             
             norm_params.Assign(norm_x.standardization_scaler.mean);
             WriteCsv(normparams_folder+csv_name_+".standardization_scaler.mean.csv",norm_params,x_vars);
             
             //--- saving the std
             
             norm_params.Assign(norm_x.standardization_scaler.std);
             WriteCsv(normparams_folder+csv_name_+".standardization_scaler.std.csv",norm_params,x_vars);
             
             
             break;
         }
       
       
       return_matrix.Resize(total, 5); //if we are collecting the train data collect the target variable also
       
       last_col = return_matrix.Cols()-1; //Column located at the last index is the last column
       
       open.CopyRates(Symbol(),PERIOD_CURRENT, COPY_RATES_OPEN, start, total);
       close.CopyRates(Symbol(),PERIOD_CURRENT, COPY_RATES_CLOSE, start, total);
        
        
       for (ulong i=0; i<(ulong)total; i++)
         {
            if (close[i] > open[i]) //bullish candle
               target[i] = 1;
            else 
               target[i] = 0;
         }               
       
       
       csv_name_ +=".targ=MOVEMENT";
       
       csv_header = x_vars + ",MOVEMENT";
       
       return_matrix.Col(target, last_col);
         
       WriteCsv("ONNX Datafolder\\"+csv_name_+".csv", return_matrix, csv_header);
     }
    
   return return_matrix;
 }
 
//+------------------------------------------------------------------+
//|   Writting a matrix to a csv file                                |
//+------------------------------------------------------------------+

bool WriteCsv(string csv_name, const matrixf &matrix_, string header_string)
  {
   FileDelete(csv_name);
   int handle = FileOpen(csv_name,FILE_WRITE|FILE_CSV|FILE_ANSI,",",CP_UTF8);

   ResetLastError();

   if(handle == INVALID_HANDLE)
     {
       printf("Invalid %s handle Error %d ",csv_name,GetLastError());
       return (false);
     }
            
   string concstring;
   vectorf row = {};
   
   datetime time_start = GetTickCount(), current_time;
   
   string header[];
   
   ushort u_sep;
   u_sep = StringGetCharacter(",",0);
   StringSplit(header_string,u_sep, header);
   
   vectorf colsinrows = matrix_.Row(0);
   
   if (ArraySize(header) != (int)colsinrows.Size())
      {
         printf("headers=%d and columns=%d from the matrix vary is size ",ArraySize(header),colsinrows.Size());
         return false;
      }

//---

   string header_str = "";
   for (int i=0; i<ArraySize(header); i++)
      header_str += header[i] + (i+1 == colsinrows.Size() ? "" : ",");
   
   FileWrite(handle,header_str);
   
   FileSeek(handle,0,SEEK_SET);
   
   for(ulong i=0; i<matrix_.Rows() && !IsStopped(); i++)
     {
      ZeroMemory(concstring);

      row = matrix_.Row(i);
      
      for(ulong j=0, cols =1; j<row.Size() && !IsStopped(); j++, cols++)
        {
         current_time = GetTickCount();
         
         Comment("Writting ",csv_name," record [",i+1,"/",matrix_.Rows(),"] Time taken | ",ConvertTime((current_time - time_start) / 1000.0));
         
         concstring += (string)row[j] + (cols == matrix_.Cols() ? "" : ",");
        }

      FileSeek(handle,0,SEEK_END);
      FileWrite(handle,concstring);
     }
        
   FileClose(handle);
   
   return (true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

string ConvertTime(double seconds)
{
    string time_str = "";
    uint minutes = 0, hours = 0;

    if (seconds >= 60)
    {
        minutes = (uint)(seconds / 60.0) ;
        seconds = fmod(seconds, 1.0) * 60;
        time_str = StringFormat("%d Minutes and %.3f Seconds", minutes, seconds);
    }
    
    if (minutes >= 60)
    {
        hours = (uint)(minutes / 60.0);
        minutes = minutes % 60;
        time_str = StringFormat("%d Hours and %d Minutes", hours, minutes);
    }

    if (time_str == "")
    {
        time_str = StringFormat("%.3f Seconds", seconds);
    }

    return time_str;
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
