//+------------------------------------------------------------------+
//|                                                     ONNX mt5.mq5 |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"

#include <Trade\Trade.mqh>
CTrade m_trade;
#include <Trade\PositionInfo.mqh>
CPositionInfo m_position;

#resource "\\Files\\ONNX Models\\MLP.REG.CLOSE.10000.onnx" as uchar RNNModel[]

#define UNDEFINED_REPLACE 1

#include <MALE5\preprocessing.mqh> //https://github.com/MegaJoctan/MALE5/blob/master/preprocessing.mqh
CPreprocessing<vectorf, matrixf> *norm_x;


input string info = "The inputs values should match the collect data script";

input uint start_bar = 100;
input uint total_bars = 10000;

input group "TRADE PARAMS";
input uint MAGIC_NUMBER = 7102023;
input uint SLIPPAGE = 100;
input uint STOPLOSS = 200;

input norm_technique NORM =   NORM_STANDARDIZATION;
string csv_name_ = "";
string normparams_folder = "ONNX Normparams\\";

uint batch_size =1;
long mlp_onnxhandle;
int inputs[], outputs[];

vectorf OPEN,
       HIGH, 
       LOW;

matrixf input_data = {};
vectorf output_data(1); //It is very crucial to resize this vector

MqlTick ticks;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
  
  //normparams_folder = (MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_OPTIMIZATION)) ? "" : normparams_folder;
   
  if (!LoadNormParams()) //Load the normalization parameters saved once
    {
      Print("Normalization parameters csv files couldn't be found \nEnsure you are collecting data and Normalizing them using [ONNX get data.ex5] Script \nTrain the Python model again if necessary");
      return INIT_FAILED;
    }
   
//--- ONNX SETTINGS
 
  mlp_onnxhandle = OnnxCreateFromBuffer(RNNModel, MQLInfoInteger(MQL_DEBUG) ? ONNX_DEBUG_LOGS : ONNX_DEFAULT); //creating onnx handle buffer | rUN DEGUG MODE during debug mode
  
  if (mlp_onnxhandle == INVALID_HANDLE)
    {
       Print("OnnxCreateFromBuffer Error = ",GetLastError());
       return INIT_FAILED;
    }

//--- since not all sizes defined in the input tensor we must set them explicitly
//--- first index - batch size, second index - series size, third index - number of series (only Close)
   
   OnnxTypeInfo type_info; //Getting onnx information for Reference In case you forgot what the loaded ONNX is all about

   long input_count=OnnxGetInputCount(mlp_onnxhandle);
   Print("model has ",input_count," input(s)");
   for(long i=0; i<input_count; i++)
     {
      string input_name=OnnxGetInputName(mlp_onnxhandle,i);
      Print(i," input name is ",input_name);
      if(OnnxGetInputTypeInfo(mlp_onnxhandle,i,type_info))
        {
          PrintTypeInfo(i,"input",type_info);
          ArrayCopy(inputs, type_info.tensor.dimensions);
        }
     }

   long output_count=OnnxGetOutputCount(mlp_onnxhandle);
   Print("model has ",output_count," output(s)");
   for(long i=0; i<output_count; i++)
     {
      string output_name=OnnxGetOutputName(mlp_onnxhandle,i);
      Print(i," output name is ",output_name);
      if(OnnxGetOutputTypeInfo(mlp_onnxhandle,i,type_info))
       {
         PrintTypeInfo(i,"output",type_info);
         ArrayCopy(outputs, type_info.tensor.dimensions);
       }
     }
   
//---

   if (MQLInfoInteger(MQL_DEBUG))
    {
      Print("Inputs & Outputs");
      ArrayPrint(inputs);
      ArrayPrint(outputs);
    }
   
   
   const long input_shape[] = {batch_size, 3};
   
   if (!OnnxSetInputShape(mlp_onnxhandle, 0, input_shape)) //Giving the Onnx handle the input shape
     {
       printf("Failed to set the input shape Err=%d",GetLastError());
       return INIT_FAILED;
     }
   
   const long output_shape[] = {batch_size, 1};
   
   if (!OnnxSetOutputShape(mlp_onnxhandle, 0, output_shape)) //giving the onnx handle the output shape
     {
       printf("Failed to set the input shape Err=%d",GetLastError());
       return INIT_FAILED;
     }
   
//--- Trade Libraries init
    
   m_trade.SetExpertMagicNumber(MAGIC_NUMBER);
   m_trade.SetDeviationInPoints(SLIPPAGE);
   m_trade.SetTypeFillingBySymbol(Symbol());
   m_trade.SetMarginMode();
   

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   
  while (CheckPointer(norm_x) != POINTER_INVALID)
    delete (norm_x);
    
   OnnxRelease(mlp_onnxhandle);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

   input_data = GetLiveData(0,1); 
   
   if (!OnnxRun(mlp_onnxhandle, ONNX_NO_CONVERSION, input_data, output_data))
     {
       Print("Failed to Get the Predictions Err=",GetLastError());
       return;
     }
   
   Comment("inputs_data\n",input_data,"\npredictions\n",output_data);
  
//--- Simple trading
   
   SymbolInfoTick(Symbol(), ticks);
   
   double min_vol = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
   int stops_level = (int)SymbolInfoInteger(Symbol(), SYMBOL_TRADE_STOPS_LEVEL);
   
   if (!TradeExists(POSITION_TYPE_BUY) && output_data[0] > ticks.ask+stops_level*Point()) //if there is no open trade of such kind and the predicted price is higher than the current ask
     m_trade.Buy(min_vol, Symbol(), ticks.ask, ticks.bid - (stops_level + STOPLOSS)*Point(), output_data[0]);
     
   if (!TradeExists(POSITION_TYPE_SELL) && output_data[0] < ticks.bid-stops_level*Point())
     m_trade.Sell(min_vol, Symbol(), ticks.bid, ticks.ask + (stops_level + STOPLOSS)*Point(), output_data[0]);
   
  }
//+------------------------------------------------------------------+
//| This Function Loads the normalization parameters saved inside    |
//| csv files and applies them to new CPreprocessing instance for    |
//| constistent normalization of the parameters to get accurate      |
//| predictions in out of sample data                                |                
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
          
          if (MQLInfoInteger(MQL_DEBUG))
              Print(EnumToString(NORM),"\nMean ",mean,"\nMin ",min,"\nMax ",max);
          
          norm_x = new CPreprocessing<vectorf,matrixf>(max, mean, min);
           
          if (mean.Sum()<=0 && min.Sum()<=0 && max.Sum() <=0)
              return false;  

         break;
         
       case NORM_MIN_MAX_SCALER:
          
          min = ReadCsvVector(normparams_folder+csv_name_+".min_max_scaler.min.csv"); //--- Loading the min
          max = ReadCsvVector(normparams_folder+csv_name_+".min_max_scaler.max.csv"); //--- Loading the max  
       
           
          if (MQLInfoInteger(MQL_DEBUG))
              Print(EnumToString(NORM),"\nMin ",min,"\nMax ",max);
              
          norm_x = new CPreprocessing<vectorf,matrixf>(max, min);
          
          
          if (min.Sum()<=0 && max.Sum() <=0)
            return false;
            
          break;
          
       case NORM_STANDARDIZATION:
          
          mean = ReadCsvVector(normparams_folder+csv_name_+".standardization_scaler.mean.csv"); //--- Loading the mean
          std = ReadCsvVector(normparams_folder+csv_name_+".standardization_scaler.std.csv"); //--- Loading the std
         
          if (MQLInfoInteger(MQL_DEBUG))
              Print(EnumToString(NORM),"\nMean ",mean,"\nStd ",std);
             
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

   //ResetLastError();
   
   if(handle == INVALID_HANDLE)
     {
      printf("Invalid %s handle Error %d ",file_name,GetLastError());
      Print(GetLastError()==0?" TIP | File Might be in use Somewhere else or in another Directory":"");
      
      FileClose(handle);
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

void PrintTypeInfo(const long num,const string layer,const OnnxTypeInfo& type_info)
  {
   Print("   type ",EnumToString(type_info.type));
   Print("   data type ",EnumToString(type_info.type));

   if(type_info.tensor.dimensions.Size()>0)
     {
      bool   dim_defined=(type_info.tensor.dimensions[0]>0);
      string dimensions=IntegerToString(type_info.tensor.dimensions[0]);
      
      
      for(long n=1; n<type_info.tensor.dimensions.Size(); n++)
        {
         if(type_info.tensor.dimensions[n]<=0)
            dim_defined=false;
         dimensions+=", ";
         dimensions+=IntegerToString(type_info.tensor.dimensions[n]);
        }
      Print("   shape [",dimensions,"]");
      //--- not all dimensions defined
      if(!dim_defined)
         PrintFormat("   %I64d %s shape must be defined explicitly before model inference",num,layer);
      //--- reduce shape
      uint reduced=0;
      long dims[];
      for(long n=0; n<type_info.tensor.dimensions.Size(); n++)
        {
         long dimension=type_info.tensor.dimensions[n];
         //--- replace undefined dimension
         if(dimension<=0)
            dimension=UNDEFINED_REPLACE;
         //--- 1 can be reduced
         if(dimension>1)
           {
            ArrayResize(dims,reduced+1);
            dims[reduced++]=dimension;
           }
        }
      //--- all dimensions assumed 1
      if(reduced==0)
        {
         ArrayResize(dims,1);
         dims[reduced++]=1;
        }
      //--- shape was reduced
      if(reduced<type_info.tensor.dimensions.Size())
        {
         dimensions=IntegerToString(dims[0]);
         for(long n=1; n<dims.Size(); n++)
           {
            dimensions+=", ";
            dimensions+=IntegerToString(dims[n]);
           }
         string sentence="";
         if(!dim_defined)
            sentence=" if undefined dimension set to "+(string)UNDEFINED_REPLACE;
         PrintFormat("   shape of %s data can be reduced to [%s]%s",layer,dimensions,sentence);
        }
     }
   else
      PrintFormat("no dimensions defined for %I64d %s",num,layer);
  }
  
  
//+------------------------------------------------------------------+
//| This Function obtains the total variable sum of historical data  |
//| starting at the bar located at the start variable                |
//+------------------------------------------------------------------+

matrixf GetLiveData(uint start, uint total)
 {
   matrixf return_matrix(total, 3);
   
   
    OPEN.CopyRates(Symbol(), PERIOD_CURRENT,COPY_RATES_OPEN, start, total);
    HIGH.CopyRates(Symbol(), PERIOD_CURRENT,COPY_RATES_HIGH, start, total);
    LOW.CopyRates(Symbol(), PERIOD_CURRENT,COPY_RATES_LOW, start, total);
        
    return_matrix.Col(OPEN, 0);
    return_matrix.Col(HIGH, 1);
    return_matrix.Col(LOW, 2);
    
        
     if (!norm_x.Normalization(return_matrix))
        Print("Failed to Normalize");  
      
      
   return return_matrix;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TradeExists(ENUM_POSITION_TYPE type)
 {   
   bool exists = false;
   
   for (int i=PositionsTotal()-1;  i>=0; i--)
      if (m_position.SelectByIndex(i))
         if (m_position.Magic() == MAGIC_NUMBER && m_position.Symbol() == Symbol() && m_position.PositionType() == type)
            {
              exists = true;
              break;  
            }
   
   return exists;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
