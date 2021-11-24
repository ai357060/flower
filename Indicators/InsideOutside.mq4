//+------------------------------------------------------------------+
//|                                                InsideOutside.mq4 |
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
extern int     slip = 50;
extern color ucolor = clrRed;
extern color dcolor = clrGreen;

int OnInit()
  {
//--- indicator buffers mapping
   sObject = "IO "+Symbol();
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
   int j;
   double h,l;
   color clr;
   double IOBars[];
   ArrayResize(IOBars,limit,10);
   IOBars[limit-1]=1;
   IOBars[limit-2]=1;
   IOBars[limit-3]=1;
   double slipvalue = slip * MarketInfo(Symbol(),MODE_POINT);
   for(int i=limit-3; i>=0; i--)
   {
      j=i+1;
      while (IOBars[j]==0)
         j+=1;
      if ((Close[i]>MathMax(Close[j],Close[j+1])+slipvalue))
      {
         IOBars[i] = 1;
         PutSymbolOnChartB(i,"X");  
      }   
      else if ((Close[i]<MathMin(Close[j],Close[j+1])-slipvalue))
      {
         IOBars[i] = -1;
         PutSymbolOnChartT(i,"X");
      }   
      else
         IOBars[i] = 0;
      
   }   
   for(int i=limit-1; i>=0; i--)
   {
      if (IOBars[i]!=0)
      {
         j=i-1;
         while (IOBars[j]==0 && j>=0)
            j-=1;
         if (Close[i]<Close[i+1])
         {
            clr = ucolor;
            h = Close[i+1]+slipvalue;
            l = Close[i]-slipvalue;
         }
         else
         {
            clr = dcolor;
            h = Close[i]+slipvalue;
            l = Close[i+1]-slipvalue;
         }   
         RectangleCreate(0,"RR1"+IntegerToString(Period())+IntegerToString(i),0,Time[i],h,Time[j],l,clr);
      }
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
bool RectangleCreate(const long            chart_ID=0,        // chart's ID 
                     string                name="Rectangle",  // rectangle name 
                     const int             sub_window=0,      // subwindow index  
                     datetime              time1=0,           // first point time 
                     double                price1=0,          // first point price 
                     datetime              time2=0,           // second point time 
                     double                price2=0,          // second point price 
                     const color           clr=clrRed,        // rectangle color 
                     int                   timeframe=OBJ_PERIOD_D1,
                     const ENUM_LINE_STYLE style=STYLE_SOLID, // style of rectangle lines 
                     const int             width=1,           // width of rectangle lines 
                     const bool            fill=false,        // filling rectangle with color 
                     const bool            back=false,        // in the background 
                     const bool            selection=false,    // highlight to move 
                     const bool            hidden=false,       // hidden in the object list 
                     const long            z_order=0)         // priority for mouse click 
                     
{ 
//--- set anchor points' coordinates if they are not set 
//   ChangeRectangleEmptyPoints(time1,price1,time2,price2); 
//--- reset the error value 
   ResetLastError(); 
//--- create a rectangle by the given coordinates 
   name = sObject+name;
   if(!ObjectCreate(chart_ID,name,OBJ_RECTANGLE,sub_window,time1,price1,time2,price2)) 
     { 
      Print(__FUNCTION__, 
            ": failed to create a rectangle! Error code = ",GetLastError()); 
      return(false); 
     } 
//--- set rectangle color 
   ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr); 
//--- set the style of rectangle lines 
   ObjectSetInteger(chart_ID,name,OBJPROP_STYLE,style); 
//--- set width of the rectangle lines 
   ObjectSetInteger(chart_ID,name,OBJPROP_WIDTH,width); 
//--- enable (true) or disable (false) the mode of filling the rectangle 
   ObjectSetInteger(chart_ID,name,OBJPROP_FILL,fill); 
//--- display in the foreground (false) or background (true) 
   ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back); 
//--- enable (true) or disable (false) the mode of highlighting the rectangle for moving 
//--- when creating a graphical object using ObjectCreate function, the object cannot be 
//--- highlighted and moved by default. Inside this method, selection parameter 
//--- is true by default making it possible to highlight and move the object 
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection); 
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection); 
//--- hide (true) or display (false) graphical object name in the object list 
   ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden); 
//--- set the priority for receiving the event of a mouse click in the chart 
   ObjectSetInteger(chart_ID,name,OBJPROP_ZORDER,z_order);
   if (timeframe== PERIOD_H4) timeframe =  OBJ_PERIOD_H4;
   else if (timeframe== PERIOD_D1) timeframe =  OBJ_PERIOD_D1;
   else if (timeframe== PERIOD_W1) timeframe =  OBJ_PERIOD_W1;
   else if (timeframe== PERIOD_MN1) timeframe =  OBJ_PERIOD_MN1;
   else timeframe =  OBJ_PERIOD_D1;
   ObjectSetInteger(chart_ID,name,OBJPROP_TIMEFRAMES,timeframe); 
   
//--- successful execution 
   return(true);   
}
  
//+------------------------------------------------------------------+
