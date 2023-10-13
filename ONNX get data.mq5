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
input uint total_bars = 10000;

input norm_technique NORM =   NORM_STANDARDIZATION;

vectorf OPEN,
        HIGH,
        LOW, 
        CLOSE; 

string csv_header = "";
string x_vars = ""; //Independent variable names for saving them into a csv file
string csv_name_ = "";
string normparams_folder = "ONNX Normparams\\";

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
  
//---

    matrixf dataset = GetTrainData(start_bar, total_bars);
    
    Print("Train data\n",dataset);


//    matrixf dataset = GetTrainData(0, 1);
//    
//    Print("Live data\n",dataset);
    
    
  while (CheckPointer(norm_x) != POINTER_INVALID)
    delete (norm_x);
  }
//+------------------------------------------------------------------+
//| This Function obtains the total variable sum of historical data  |
//| starting at the bar located at the start variable                |
//+------------------------------------------------------------------+
matrixf GetTrainData(uint start, uint total)
 {
   matrixf return_matrix(total, 3);
   
   ulong last_col;
   
   
    OPEN.CopyRates(Symbol(), PERIOD_CURRENT, COPY_RATES_OPEN, start, total);
    HIGH.CopyRates(Symbol(), PERIOD_CURRENT, COPY_RATES_HIGH, start, total);
    LOW.CopyRates(Symbol(), PERIOD_CURRENT, COPY_RATES_LOW, start, total);
    CLOSE.CopyRates(Symbol(), PERIOD_CURRENT, COPY_RATES_CLOSE, start, total);
    
    return_matrix.Col(OPEN, 0);
    return_matrix.Col(HIGH, 1);
    return_matrix.Col(LOW, 2);
    
    matrixf norm_params = {};
    
    csv_name_ = Symbol()+"."+EnumToString(Period())+"."+string(total_bars);
    
       
      x_vars = "OPEN,HIGH,LOW";
      
       while (CheckPointer(norm_x) != POINTER_INVALID)
         delete (norm_x);
         
       norm_x = new CPreprocessing<vectorf, matrixf>(return_matrix, NORM);
    
       
 
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
 
       return_matrix.Resize(total, 4); //if we are collecting the train data collect the target variable also
       
       last_col = return_matrix.Cols()-1; //Column located at the last index is the last column
       
       return_matrix.Col(CLOSE, last_col); //put the close price information in the last column of a matrix
       
       
       csv_name_ +=".targ=CLOSE";
       
       csv_header = x_vars + ",CLOSE";
         
       if (!WriteCsv("ONNX Datafolder\\"+csv_name_+".csv", return_matrix, csv_header))
         Print("Failed to Write to a csv file");
       else
         Print("Data saved to a csv file successfully");
     
    
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
