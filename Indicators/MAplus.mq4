//+------------------------------------------------------------------+
//|                                                       MAplus.mq4 |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
#property indicator_separate_window
#property indicator_buffers 4
#property indicator_plots   4
//--- plot SMA
#property indicator_label1  "SMA"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDarkTurquoise
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
//--- plot SMAdiffdiff
#property indicator_label2  "SMAdiffdiff"
#property indicator_type2   DRAW_HISTOGRAM
#property indicator_color2  clrRed
#property indicator_style2  STYLE_SOLID
#property indicator_width2  4
//--- plot SMAdiff2
#property indicator_label3  "SMAdiff2"
#property indicator_type3   DRAW_HISTOGRAM
#property indicator_color3  clrOrange
#property indicator_style3  STYLE_SOLID
#property indicator_width3  2
//--- plot SMAclose
#property indicator_label4  "SMAclose"
#property indicator_type4   DRAW_HISTOGRAM
#property indicator_color4  clrMediumSeaGreen
#property indicator_style4  STYLE_SOLID
#property indicator_width4  1
//--- indicator buffers
double         SMABuffer[];
double         SMAdiffdiffBuffer[];
double         SMAdiff2Buffer[];
double         SMAdiffBuffer[];
double         SMAcloseBuffer[];

extern int SMAperiod = 5;
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0,SMABuffer);
   SetIndexBuffer(1,SMAdiffdiffBuffer);
   SetIndexBuffer(2,SMAdiff2Buffer);
   SetIndexBuffer(3,SMAcloseBuffer);
   
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
   int i,limit;

//---
   limit=rates_total-prev_calculated;
   if(prev_calculated>0)
      limit++;
   ArrayResize(SMAdiffBuffer,limit,10);      
   for(i=0; i<limit; i++)
      SMABuffer[i]=iMA(NULL,0,SMAperiod,0,MODE_SMA,PRICE_CLOSE,i);
   for(i=0; i<limit-1; i++)
      SMAdiffBuffer[i]=SMABuffer[i]-SMABuffer[i+1];
   for(i=0; i<limit-2; i++)
      SMAdiffdiffBuffer[i]=(SMAdiffBuffer[i]-SMAdiffBuffer[i+1])*100;
   for(i=0; i<limit-2; i++)
      SMAdiff2Buffer[i]=(SMABuffer[i]-SMABuffer[i+2])*100;
   for(i=0; i<limit-1; i++)
      SMAcloseBuffer[i]=(Close[i]-SMABuffer[i])*100;
   
//--- return value of prev_calculated for next call
   return(rates_total);
}
//+------------------------------------------------------------------+
