//+------------------------------------------------------------------+
//|                                                      Signall.mq4 |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
#property indicator_separate_window
#property indicator_buffers 2
#property indicator_plots   2
//--- plot SignalBar
#property indicator_label1  "SignalBar"
#property indicator_type1   DRAW_HISTOGRAM
#property indicator_color1  clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
#property indicator_label2  "Temp"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrBlue
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1
//--- parameters
int      MA1=35;
int      MA2=3;
int      RSI=14;
int      Ma1Diff_from = 0;
int      Ma1Diff_to   = 1000;
int      Ma1DiffDiff_from = -1000;
int      Ma1DiffDiff_to   = 1000;

int      Ma2Diff_from = -1000;
int      Ma2Diff_to   = 1000;
int      Ma2DiffDiff_from = -1000;
int      Ma2DiffDiff_to   = 0;

int      Rsi_from = 0;
int      Rsi_to   = 100;
int      RsiDiff_from = -1000;
int      RsiDiff_to   = 0;


//--- indicator buffers
double         SignalBarBuffer[];
double         TempBuffer[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0,SignalBarBuffer);
   SetIndexBuffer(1,TempBuffer);
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
//---
   double Ma1Previous0,Ma1Previous1,Ma1Previous2;
   double Ma2Previous0,Ma2Previous1,Ma2Previous2;
   double RsiPrevious0,RsiPrevious1,RsiPrevious2;
   double Ma1Diff,Ma2Diff;
   double Ma1DiffDiff,Ma2DiffDiff;
   double RsiDiff;
   double atr;
   int i,j,limit;

//---
   limit=rates_total-prev_calculated;
   if(prev_calculated>0)
      limit++;

   for(i=0; i<limit; i++)
   {
      j = i + 0;
      Ma1Previous0=iMA(NULL,0,MA1,0,MODE_SMA,PRICE_CLOSE,j+0);
      Ma1Previous1=iMA(NULL,0,MA1,0,MODE_SMA,PRICE_CLOSE,j+1);
      Ma1Previous2=iMA(NULL,0,MA1,0,MODE_SMA,PRICE_CLOSE,j+2);
      Ma1Diff = Ma1Previous0-Ma1Previous1;
      Ma1DiffDiff = Ma1Previous0-Ma1Previous1 - (Ma1Previous1-Ma1Previous2);
      
      Ma2Previous0=iMA(NULL,0,MA2,0,MODE_SMA,PRICE_CLOSE,j+0);
      Ma2Previous1=iMA(NULL,0,MA2,0,MODE_SMA,PRICE_CLOSE,j+1);
      Ma2Previous2=iMA(NULL,0,MA2,0,MODE_SMA,PRICE_CLOSE,j+2);
      Ma2Diff = Ma2Previous0-Ma2Previous1;
      Ma2DiffDiff = Ma2Previous0-Ma2Previous1 - (Ma2Previous1-Ma2Previous2);
      
      RsiPrevious0=iRSI(NULL,0,RSI,0,j+0);
      RsiPrevious1=iRSI(NULL,0,RSI,0,j+1);
      RsiPrevious2=iRSI(NULL,0,RSI,0,j+2);
      RsiDiff = RsiPrevious0-RsiPrevious1;
      
      TempBuffer[i]=Ma2DiffDiff;
      
      atr = iATR(NULL,0,20,j);
      
      
      if (Ma1Diff>Ma1Diff_from && Ma1Diff<=Ma1Diff_to && Ma1DiffDiff>Ma1DiffDiff_from && Ma1DiffDiff<=Ma1DiffDiff_to 
      && Ma2Diff>Ma2Diff_from && Ma2Diff<=Ma2Diff_to && Ma2DiffDiff>Ma2DiffDiff_from && Ma2DiffDiff<=Ma2DiffDiff_to 
      && RsiPrevious0>Rsi_from && RsiPrevious0<=Rsi_to && RsiDiff>RsiDiff_from && RsiDiff<=RsiDiff_to
      && atr<=0.015)
      {
         SignalBarBuffer[i]=1;
      }
      else
      {
         SignalBarBuffer[i]=0;
      }
   }   
   
//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
