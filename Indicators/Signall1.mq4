//+------------------------------------------------------------------+
//|                                                      Signall1.mq4 |
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
int      ATR=20;

double   MinMADiff = 0;//0.0001;
double   MinMADiffDiff = 0;//0.001;
double   MinRSIDiff = 0;//1;
double   MinRSIDiffDiff = 0;//10;

double   Atr_from = -1000;
double   Atr_to   = 0.015;

string   MA1dt = "12";
string   MA2dt = "1234";
string   RSIdt = "34";
double   Rsi_from = 0;
double   Rsi_to   = 60;

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
   double RsiDiff,RsiDiffDiff;
   double atr;
   int i,j,limit;
   int signal;

//---
   limit=rates_total-prev_calculated;
   if(prev_calculated>0)
      limit++;
   
   string difftype;
   difftype = "123";
   if(StringFind(difftype,"1",0)>-1)
   {
      Print("ok");
   }

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
      RsiDiffDiff = RsiPrevious0-RsiPrevious1 - (RsiPrevious1-RsiPrevious2);
      
//      TempBuffer[i]=Ma2DiffDiff;
      
      atr = iATR(NULL,0,ATR,j);
      
      signal = CheckDiffType(MA1dt,Ma1Diff,Ma1DiffDiff,MinMADiff,MinMADiffDiff);
      if (signal!=0)
         signal = CheckDiffType(MA2dt,Ma2Diff,Ma2DiffDiff,MinMADiff,MinMADiffDiff);
      if (signal!=0)
         signal = CheckDiffType(RSIdt,RsiDiff,RsiDiffDiff,MinRSIDiff,MinRSIDiffDiff);
      if (signal!=0)
         if (RsiPrevious0>=Rsi_from && RsiPrevious0<Rsi_to && atr>=Atr_from && atr<Atr_to)
            signal = 1;
         else
            signal = 0;   

      SignalBarBuffer[i]=signal;
   }   
   
//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+

int CheckDiffType(string dt,double diff,double diffdiff,double mindiff,double mindiffdiff)
{
   int signal = 0;
   if(     StringFind(dt,"1",0)>-1 && diff>=mindiff && diffdiff>=mindiffdiff)
      signal = 1;
   else if(StringFind(dt,"2",0)>-1 && diff>=mindiff && diffdiff<-mindiffdiff)
      signal = 1;
   else if(StringFind(dt,"3",0)>-1 && diff<-mindiff && diffdiff<-mindiffdiff)
      signal = 1;
   else if(StringFind(dt,"4",0)>-1 && diff<-mindiff && diffdiff>=mindiffdiff)
      signal = 1;

   return signal;
}
