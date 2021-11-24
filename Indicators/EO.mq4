//+------------------------------------------------------------------+
//|                                                           EO.mq4 |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
#property indicator_chart_window
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+

string      sObject;

int OnInit()
  {
//--- indicator buffers mapping
   sObject = "EO "+Symbol();
   DeleteOwnObjects();
   
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
   int counted_bars=IndicatorCounted();
   int limit = Bars-counted_bars-1;
   for(int i=limit-1; i>=0; i--)
   {
      if (MathMax(Open[i],Close[i])>=MathMax(Open[i+1],Close[i+1]) && MathMin(Open[i],Close[i])<=MathMin(Open[i+1],Close[i+1])&& High[i]>=High[i+1] && Low[i]<=Low[i+1])
         if (Close[i]>Open[i])
            PutSymbolOnChartB(i,"X");  
         else
            PutSymbolOnChartT(i,"X");  
      else if (MathMax(Open[i],Close[i])>=MathMax(Open[i+1],Close[i+1]) && MathMin(Open[i],Close[i])<=MathMin(Open[i+1],Close[i+1]))
         if (Close[i]>Open[i])
            PutSymbolOnChartB(i,"E");  
         else
            PutSymbolOnChartT(i,"E");  
//      else if (High[i]>=High[i+1] && Low[i]<=Low[i+1])
//         if (Close[i]>Open[i])
//            PutSymbolOnChartB(i,"O");  
//         else
//            PutSymbolOnChartT(i,"O");  
      else if ((High[i]-MathMax(Open[i],Close[i]))>=2*MathAbs(Open[i]-Close[i]) && (MathMin(Open[i],Close[i])-Low[i])<1*MathAbs(Open[i]-Close[i]))
         PutSymbolOnChartT(i,"S");  
      else if ((MathMin(Open[i],Close[i])-Low[i])>=2*MathAbs(Open[i]-Close[i]) && (High[i]-MathMax(Open[i],Close[i]))<1*MathAbs(Open[i]-Close[i]))
         PutSymbolOnChartB(i,"H");  
         
         
//      if (Period()==PERIOD_D1)
//         if(TimeDayOfWeek(Time[i])==1)
//         {
//            string name = "HL"+IntegerToString(i);
//            ObjectCreate(0,name,OBJ_VLINE,0,Time[i],0);  
//            ObjectSetInteger(0,name,OBJPROP_STYLE,STYLE_DOT);
//            ObjectSetInteger(0,name,OBJPROP_COLOR,Gray); 
//         }   
            
   
   }   
//--- return value of prev_calculated for next call
   return(rates_total);
  }
  
void DeleteOwnObjects()
{
   int i=0;
   while (i <= ObjectsTotal()) {
      if (StringFind(ObjectName(i), sObject) >= 0) ObjectDelete(ObjectName(i));
      else
      i++;
   }
   return;
}

void PutSymbolOnChartT(int Pos,string ss,int shift=12,int tclr=Red)
{
   double priceperpixel = (ChartGetDouble(0,CHART_PRICE_MAX,0)-ChartGetDouble(0,CHART_PRICE_MIN,0))/ChartGetInteger(0,CHART_HEIGHT_IN_PIXELS,0);
   string strObjectName = sObject+IntegerToString(Pos);
   ObjectDelete(strObjectName);
   ObjectCreate(strObjectName, OBJ_TEXT, 0, Time[Pos], High[Pos]+shift*priceperpixel, Time[Pos], High[Pos]+shift*priceperpixel);
   ObjectSetText(strObjectName,ss,6,"Arial Black",tclr);   

}
void PutSymbolOnChartB(int Pos,string ss,int shift=2,int tclr=Red)
{
   double priceperpixel = (ChartGetDouble(0,CHART_PRICE_MAX,0)-ChartGetDouble(0,CHART_PRICE_MIN,0))/ChartGetInteger(0,CHART_HEIGHT_IN_PIXELS,0);
   string strObjectName = sObject+IntegerToString(Pos);
   ObjectDelete(strObjectName);
   ObjectCreate(strObjectName, OBJ_TEXT, 0, Time[Pos], Low[Pos]-shift*priceperpixel, Time[Pos], Low[Pos]-shift*priceperpixel);
   ObjectSetText(strObjectName,ss,6,"Arial Black",tclr);   

}
  
//+------------------------------------------------------------------+
