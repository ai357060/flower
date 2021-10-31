//+------------------------------------------------------------------+
//|                                            TradeHelper 7.mq4     |
//|                                            © 2016       PO       |
//|                                                                  |
//+------------------------------------------------------------------+

#include <stdlib.mqh>

#property copyright "© 2008.10.31 PO"
#property link      ""

#property indicator_chart_window

#property indicator_buffers 1
#property indicator_color1 Green
#property indicator_width1 2

#property indicator_levelcolor Black

//---- inputs --------------------------------------------------------

extern int     e_Corner    = 1;  //0=Upper Left, 1=Upper Right, 2=lower left , 3=lower right
extern double  SLPLN       = 300;
extern int     RhRange     = 200;
extern int     AtrPeriod   = 14;
extern double  atrWSP      = 1;
extern double  slippagePLN = 0;
extern double  slippageTICK_entry = 10;
extern double  slippageTICK_sl = 50;
extern double  slippagePerc= 0;
extern bool    showTable   = False;
extern bool    Debug = False;

//--------------------------------------------------------------------

int         e_xx;
int         yy;
string      sObject;
string      isn;
double      Line1[];

double   vsl;
double   minvsl;
int      lotsize;
double   minlot;
double   minvslpln;
double   wsp;
double   tradelot;
double   tickpln;
double   ticksize;
double   slippage_entry;
double   slippage_sl;
double   vsl0;      //atr
double   stoplevel;
double   myspread;
double   realvsl;
double   tradeallowed;
string   bisymbol;
string   curr;
string   acccurr;
datetime ttime; 

int TrendMN1;
int TrendW1;
int TrendD1;

int inifinity=100000;
//---------------------------------------------------------------------------------|
//                                                                                 |
//---------------------------------------------------------------------------------|
int OnInit()
{
   sObject = "TradeHelper7 "+Symbol();
   DeleteOwnObjects(sObject);
   SetIndexBuffer(0,Line1); SetIndexLabel(0,"TH7 "); 
   SetIndexStyle(0,DRAW_LINE);
   isn = "TradeHelper7";
   IndicatorShortName(isn);
   bisymbol=Symbol();
   return(INIT_SUCCEEDED);
}

//---------------------------------------------------------------------------------|
//                                                                                 |
//---------------------------------------------------------------------------------|
void OnDeinit(const int reason)
{
   DeleteOwnObjects(sObject);
}

/*
void OnChartEvent(const int id,         // Event identifier   
                  const long& lparam,   // Event parameter of long type 
                  const double& dparam, // Event parameter of double type 
                  const string& sparam) // Event parameter of string type 
{ 
   if(id==CHARTEVENT_OBJECT_CLICK) 
     { 
      Print("The mouse has been clicked on the object with name '"+sparam+"'"); 
      ChartRedraw();
      RefreshRates();
      
     } 

}
*/
//---------------------------------------------------------------------------------|
//                                                                              |
//---------------------------------------------------------------------------------|
int OnCalculate (const int rates_total,      // size of input time series
                 const int prev_calculated,  // bars handled in previous call
                 const datetime& time[],     // Time
                 const double& open[],       // Open
                 const double& high[],       // High
                 const double& low[],        // Low
                 const double& close[],      // Close
                 const long& tick_volume[],  // Tick Volume
                 const long& volume[],       // Real Volume
                 const int& spread[]         // Spread
   )
{   
   DeleteOwnObjects(sObject);

   int buyTopPos;
   double buyTopPrice;
   int buyBottomPos;
   double buyBottomPrice;

   int sellTopPos;
   double sellTopPrice;
   int sellBottomPos;
   double sellBottomPrice;

   SetVariables();

   if (showTable)
   {
      vsl0 = atrWSP*iATR(bisymbol,Period(),MathMin(AtrPeriod,Bars-1),0);
      Calc(vsl0);

      DisplayTable(170,13,1);
   }
   
   DetectTrends();
   DisplayTrends(10,13,1);
   
   datetime tmpdb;
   int      tmppos;
   double   tmppriceU;
   double   tmppriceD;
   int      tmpperiod;
   
   int sellRectanglePos;
   int buyRectanglePos;
   double sellRectanglePriceU;
   double sellRectanglePriceD;
   double buyRectanglePriceU;
   double buyRectanglePriceD;
   double currPrice;
   
   if (ReadOpenPositions()==True)
   {
   }
   else if (Period()==PERIOD_D1)
   {
      currPrice=Ask;
      //szuka prostok¹ty
      sellRectanglePos=inifinity;
      buyRectanglePos=inifinity;
      int i=0;
      while (i <= ObjectsTotal()) {
         if (ObjectType(ObjectName(i))== OBJ_RECTANGLE)
         {
            tmpdb    = ObjectGet(ObjectName(i),OBJPROP_TIME1);
            tmppos   = iBarShift(bisymbol,PERIOD_D1,tmpdb,false);
            tmppriceU = ObjectGet(ObjectName(i),OBJPROP_PRICE1);
            tmppriceD = ObjectGet(ObjectName(i),OBJPROP_PRICE2);
            tmpperiod = ObjectGet(ObjectName(i),OBJPROP_TIMEFRAMES);

            if ((tmpperiod&OBJ_PERIOD_D1) == OBJ_PERIOD_D1  && tmppriceD>currPrice && tmppos<sellRectanglePos)
            {
               sellRectanglePos = tmppos;
               sellRectanglePriceU = tmppriceU;
               sellRectanglePriceD = tmppriceD;
            }
            i++;
         }
         else
         i++;
      }

      i=0;
      while (i <= ObjectsTotal()) {
         if (ObjectType(ObjectName(i))== OBJ_RECTANGLE)
         {
            tmpdb    = ObjectGet(ObjectName(i),OBJPROP_TIME1);
            tmppos   = iBarShift(bisymbol,PERIOD_D1,tmpdb,false);
            tmppriceU = ObjectGet(ObjectName(i),OBJPROP_PRICE1);
            tmppriceD = ObjectGet(ObjectName(i),OBJPROP_PRICE2);
            tmpperiod = ObjectGet(ObjectName(i),OBJPROP_TIMEFRAMES);
               if (Debug==True)
                  Print("sellRectanglePos: "+tmppriceU);            
            if ((tmpperiod&OBJ_PERIOD_D1) == OBJ_PERIOD_D1  && tmppriceU<currPrice && tmppos<buyRectanglePos)
            {
               buyRectanglePos = tmppos;
               buyRectanglePriceU = tmppriceU;
               buyRectanglePriceD = tmppriceD;
            }
            i++;
         }
         else
         i++;
      }
      sellRectanglePriceD=0.999*sellRectanglePriceD;
      buyRectanglePriceU=1.001*buyRectanglePriceU;
      
      if (sellRectanglePos!=inifinity)
      {
         //wyznacza górê i dó³ w prostok¹tach
         i = sellRectanglePos;
         while (IsCrossed(i,PERIOD_D1,sellRectanglePriceU,sellRectanglePriceD)) i--;
         i++;
         sellBottomPrice   = inifinity;
         sellTopPrice      = 0;
         while (i<=sellRectanglePos)
         {
            tmppriceD = MathMin(iOpen(bisymbol,PERIOD_D1,i),iClose(bisymbol,PERIOD_D1,i));
            if (tmppriceD <= sellBottomPrice && tmppriceD>=sellRectanglePriceD &&tmppriceD<=sellRectanglePriceU)
            {
               sellBottomPrice = tmppriceD;
               sellBottomPos = i;
            }
            tmppriceD = iHigh(bisymbol,PERIOD_D1,i);
            if (tmppriceD >= sellTopPrice)
            {
               sellTopPrice = tmppriceD;
               sellTopPos = i;
            }
            i++;
         }
      }
      if (buyRectanglePos!=inifinity)
      {      
         i = buyRectanglePos;
         while (IsCrossed(i,PERIOD_D1,buyRectanglePriceU,buyRectanglePriceD)) i--;
         i++;
         buyBottomPrice   = inifinity;
         buyTopPrice      = 0;
         while (i<=buyRectanglePos)
         {
            tmppriceU = MathMax(iOpen(bisymbol,PERIOD_D1,i),iClose(bisymbol,PERIOD_D1,i));
            if (tmppriceU >= buyTopPrice && tmppriceU>=buyRectanglePriceD &&tmppriceU<=buyRectanglePriceU)
            {
               buyTopPrice = tmppriceU;
               buyTopPos = i;
            }
            tmppriceU = iLow(bisymbol,PERIOD_D1,i);
            if (tmppriceU <= buyBottomPrice)
            {
               buyBottomPrice = tmppriceU;
               buyBottomPos = i;
            }
            i++;
         }
      }
            
      double priceperpixel = (ChartGetDouble(0,CHART_PRICE_MAX,0)-ChartGetDouble(0,CHART_PRICE_MIN,0))/ChartGetInteger(0,CHART_HEIGHT_IN_PIXELS,0);
      string strObjectName;   
         
      
      
      if (sellBottomPos>0)
      {
         strObjectName = sObject+" SellBottomLine";
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_TEXT, 0, Time[sellBottomPos], Low[sellBottomPos]-4*priceperpixel, Time[sellBottomPos-1], Low[sellBottomPos]-4*priceperpixel);
         ObjectSetText(strObjectName,"S",6,"Arial Black",Red);   
   
         strObjectName = sObject+" SellTopLine";
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_TEXT, 0, Time[sellTopPos], sellTopPrice+12*priceperpixel, Time[sellTopPos-1], sellTopPrice+12*priceperpixel);
         ObjectSetText(strObjectName,"S",6,"Arial Black",Red);   
         if (Debug==True)
         {
            yy             = 170;
            e_xx           = 13;   
            e_Corner       = 1;

            WriteInfo("sellBottomPrice",sellBottomPrice);      
            WriteInfo("sellTopPrice",sellTopPrice);      

         }
      }
      if (buyTopPos>0)
      {
         strObjectName = sObject+" BuyTopLine";
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_TEXT, 0, Time[buyTopPos], High[buyTopPos]+12*priceperpixel, Time[buyTopPos-1], High[buyTopPos]+12*priceperpixel);
         ObjectSetText(strObjectName,"B",6,"Arial Black",Red);   
   
         strObjectName = sObject+" BuyBottomLine";
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_TEXT, 0, Time[buyBottomPos], buyBottomPrice-4*priceperpixel, Time[buyBottomPos-1], buyBottomPrice-4*priceperpixel);
         ObjectSetText(strObjectName,"B",6,"Arial Black",Red);   
         if (Debug==True)
         {
            yy             = 170;
            e_xx           = 13;   
            e_Corner       = 1;

            WriteInfo("buyBottomPrice",buyBottomPrice);      
            WriteInfo("buyTopPrice",buyTopPrice); 
         }
      }
   }
   return(rates_total);
}
void DetectTrends()
{
   int i=0;
   datetime tmpdb1,tmpdb2;
   double tmpprice1,tmpprice2;
   TrendMN1 = 0;
   TrendW1 = 0;
   TrendD1 = 0;
   int tmpperiod;
   while (i <= ObjectsTotal()) {
      if (ObjectType(ObjectName(i))== OBJ_TREND)
      {
         tmpdb1    = ObjectGet(ObjectName(i),OBJPROP_TIME1);
         tmpdb2    = ObjectGet(ObjectName(i),OBJPROP_TIME2);
         tmpprice1 = ObjectGet(ObjectName(i),OBJPROP_PRICE1);
         tmpprice2 = ObjectGet(ObjectName(i),OBJPROP_PRICE2);
         tmpperiod = ObjectGet(ObjectName(i),OBJPROP_TIMEFRAMES);
         if ((tmpperiod&OBJ_PERIOD_MN1) == OBJ_PERIOD_MN1)
         {
            if (tmpdb1<tmpdb2 && tmpprice1>tmpprice2) TrendMN1 = -2;
            else if (tmpdb1>tmpdb2 && tmpprice1<tmpprice2) TrendMN1 = -2;
            else TrendMN1=2;
            TrendMN1=isTrendBroken(PERIOD_MN1,tmpdb1,tmpdb2,tmpprice1,tmpprice2,TrendMN1);
         }
         else if ((tmpperiod&OBJ_PERIOD_W1) == OBJ_PERIOD_W1)
         {
            if (tmpdb1<tmpdb2 && tmpprice1>tmpprice2) TrendW1 = -2;
            else if (tmpdb1>tmpdb2 && tmpprice1<tmpprice2) TrendW1 = -2;
            else TrendW1 = 2;
            TrendW1=isTrendBroken(PERIOD_W1,tmpdb1,tmpdb2,tmpprice1,tmpprice2,TrendW1);
         }
         else if ((tmpperiod&OBJ_PERIOD_D1) == OBJ_PERIOD_D1)
         {
            if (tmpdb1<tmpdb2 && tmpprice1>tmpprice2) TrendD1 = -2;
            else if (tmpdb1>tmpdb2 && tmpprice1<tmpprice2) TrendD1 = -2;
            else TrendD1 = 2;
            TrendD1=isTrendBroken(PERIOD_D1,tmpdb1,tmpdb2,tmpprice1,tmpprice2,TrendD1);
         }         
         i++;
      }
      else
      i++;
   }

}

bool IsCrossed(int shift,int period,double levelU,double levelD)
{
   if (iHigh(bisymbol,period,shift)>=levelD && levelU>=iHigh(bisymbol,period,shift)) return (True);
   else if (iLow(bisymbol,period,shift)>=levelD && levelU>=iLow(bisymbol,period,shift)) return (True);
   else if (iHigh(bisymbol,period,shift)>=levelU && levelD>=iLow(bisymbol,period,shift)) return (True);
   else return(False);
}
int isTrendBroken(int PPeriod,datetime dt1,datetime dt2,double price1, double price2,int trend)
{
   int pos1,pos2;
   double per;
   int i;
   pos1 =  iBarShift(bisymbol, PPeriod, dt1,true);
   pos2 =  iBarShift(bisymbol, PPeriod, dt2,true);
   if (pos1-pos2!=0)
   {
      per = MathAbs(price1-price2)/(1.0*(MathAbs(pos1-pos2)));
     
      if (trend==2)
      {
         for (i=1;i<=10;i++)
         if (pos2>=i)
         {
            if (bisymbol=="AUDCAD" && PPeriod==PERIOD_MN1 &&i==1)
            {
               Print(i);
               Print(per*MathAbs(pos2-i)+price2);
               Print(iHigh(bisymbol,PPeriod,i));
            }
            if (iHigh(bisymbol,PPeriod,i)<price2+per*MathAbs(pos2-i))
            {
               trend = 1;
               break;
            }
         }
         else
         {
            if (bisymbol=="AUDCAD" && PPeriod==PERIOD_MN1 && i==1)
            {
               Print(i);
               Print(price2-per*MathAbs(pos2-i));
               Print(iHigh(bisymbol,PPeriod,i));
            }           
            if (iHigh(bisymbol,PPeriod,i)<price2-per*MathAbs(pos2-i))
            {
               trend = 1;
               break;
            }
         }
      }
      else if (trend==-2)
      {
         for (i=1;i<=10;i++)
         if (pos2>=i)
            if ((iLow(bisymbol,PPeriod,i))>price2-per*MathAbs(pos2-i))
            {
               trend = -1;
               break;
            }
         else
            if ((iLow(bisymbol,PPeriod,i))>price2+per*MathAbs(pos2-i))
            {
               trend = -1;
               break;
            }
      }
   }
   return trend;
}


void SetVariables()
{
   double tmp;

   minlot = MarketInfo(bisymbol,MODE_MINLOT);
   lotsize = MarketInfo(bisymbol,MODE_LOTSIZE);
   ttime = MarketInfo(bisymbol,MODE_TIME);
   ticksize = MarketInfo(bisymbol,MODE_TICKSIZE);
   stoplevel = MarketInfo(bisymbol,MODE_STOPLEVEL);
   myspread=Ask-Bid;
   
   if (bisymbol == "BRENT" || bisymbol == "GOLD" || bisymbol == "SILVER" || StringSubstr(bisymbol,0,1)=="#")
      curr = "USD";
   else
      curr = StringSubstr(bisymbol,3,3);
  
   acccurr = AccountCurrency(); 
   
//wsp - kurs

   wsp = -1;
   if (curr=="EUR")
      tmp = 1;
   else
      tmp = iClose("EUR"+curr,PERIOD_D1,0);
   
   if (tmp!=0) 
      wsp = (1.0/tmp)*iClose("EUR"+acccurr,PERIOD_D1,0);
   else
   {
      tmp = iClose("USD"+curr,PERIOD_D1,0);
      if (tmp!=0) 
         wsp = (1.0/tmp)*iClose("USD"+acccurr,PERIOD_D1,0);
      else Print("Brak kursu: "+curr);
   }                        

}
void DisplayTrends(int dx, int dy,int dc)
{
    
   yy             = dy;
   e_xx           = dx;   
   e_Corner       = dc;

   WriteInfo("MN:  "+ DisplayChar(TrendMN1),"");      
   WriteInfo("W1:  "+ DisplayChar(TrendW1),"");      
   WriteInfo("D1:  "+ DisplayChar(TrendD1),"");      
}
string DisplayChar(int trend)
{
   string rr;
   if (trend == -2) rr=CharToString(92);
   else if (trend == -1) rr=CharToString(92)+CharToString(126);
   else if (trend == 2) rr=CharToString(47);
   else if (trend == 1) rr=CharToString(47)+CharToString(126);
   else if (trend == 0) rr=CharToString(61);
   return rr;
   
}

void DisplayTable(int dx, int dy,int dc)
{
    
   yy             = dy;
   e_xx           = dx;   
   e_Corner       = dc;

   WriteInfo("symbol",bisymbol+" "+TimeToStr(ttime,TIME_MINUTES));      
   WriteInfo("atr",DoubleToStr(vsl0,Digits)+" ("+DoubleToStr(atrWSP,1)+")");     
   WriteInfo("curr",curr);      
   WriteInfo("tradelot",DoubleToStr(tradelot,2));            
   WriteInfo("lotsize",lotsize);      
   WriteInfo("minlot",DoubleToStr(minlot,2));      
   WriteInfo("wsp",wsp);      

   yy             = 13;
   e_xx           = e_xx - 160;   

   WriteInfo("TRADE",tradeallowed);      
   WriteInfo("STOPLEVEL",DoubleToStr((MarketInfo(bisymbol,MODE_STOPLEVEL)),0));      
   WriteInfo("Spread",DoubleToStr(myspread,Digits));      
   WriteInfo("minsl"+acccurr,DoubleToStr(minvslpln,2));
   WriteInfo("slip_entry ",IntegerToString(slippageTICK_entry));
   WriteInfo("slip_sl",IntegerToString(slippageTICK_sl));
   WriteInfo("SL"+acccurr,DoubleToStr(SLPLN,2));      

}

void Calc(double vs)
{
   vsl = StrToDouble(DoubleToStr(vs,Digits));

   if (vsl == 0) 
   {
      slippage_entry = 0;
      slippage_sl = 0;
   }
   else
   {
      slippage_entry = ticksize * slippageTICK_entry;
      slippage_sl = ticksize * slippageTICK_sl;
   }
   
   minvsl = (vsl + slippage_entry+slippage_sl)*lotsize*minlot;
   minvslpln = minvsl *wsp;

   if (minvslpln!=0) 
      tradelot = MathFloor((double)SLPLN/minvslpln)*minlot;
   else 
      tradelot = 0;
      
   tickpln = ticksize*wsp*tradelot*lotsize;
   
   realvsl = tradelot*(vsl + slippage_entry+slippage_sl)*lotsize*wsp;

}

bool ReadOpenPositions()
{
   bool result=False;
   int orderpos;
   double R;
   string strObjectName;
   double wholeSL;  
   double wspOP; 
   string currOP;
   double tmp;
   int lotsizeOP;
         string ss;
   
      e_xx           = e_xx - 160;
      
      WriteInfo("","");
      WriteInfo("","");
      WriteInfo("","");
   
   
   
   
     wholeSL = 0; 
     for(int i=0;i<OrdersTotal();i++)
     {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false)        
         break;
      if(OrderSymbol()!=Symbol()) 
         {
            wspOP = -1; 
            acccurr = AccountCurrency();   
            lotsizeOP = MarketInfo(OrderSymbol(),MODE_LOTSIZE);
            if (OrderSymbol() == "GOLD" || OrderSymbol() == "SILVER" || StringSubstr(OrderSymbol(),0,1)=="#")
               currOP = "USD";
            else
               currOP = StringSubstr(OrderSymbol(),3,3);

            if (currOP=="EUR")
               tmp = 1;
            else
               tmp = iClose("EUR"+currOP,PERIOD_D1,0);
            
            if (tmp!=0) 
               wspOP = (1.0/tmp)*iClose("EUR"+acccurr,PERIOD_D1,0);
            else
            {
               tmp = iClose("USD"+currOP,PERIOD_D1,0);
               if (tmp!=0) 
                  wspOP = (1.0/tmp)*iClose("USD"+acccurr,PERIOD_D1,0);
               else Print("Brak kursu: "+currOP);
            }                        

            if(OrderType()==OP_BUY)
               wholeSL = wholeSL + (OrderStopLoss() - OrderOpenPrice())*OrderLots()*lotsizeOP*wspOP;
            if(OrderType()==OP_SELL)               
               wholeSL = wholeSL + (OrderOpenPrice() - OrderStopLoss())*OrderLots()*lotsizeOP*wspOP;
            continue;
         }
      //---- check order type 
      
      if(OrderType()==OP_BUY)
        {         
         wholeSL = wholeSL + (OrderStopLoss() - OrderOpenPrice())*OrderLots()*lotsize*wsp;
         WriteInfo("R_B: "+OrderComment()+"   SL: "+DoubleToStr((OrderStopLoss() - OrderOpenPrice())*OrderLots()*lotsize*wsp,2)+"        A: "+DoubleToStr(OrderProfit(),2),"",Red);     

         orderpos = iBarShift(Symbol(),Period(),OrderOpenTime(),False);
         
         string initsl = OrderComment();
         StringReplace(initsl,":1","");
         StringReplace(initsl,":2","");
         StringReplace(initsl,":3","");
         StringReplace(initsl,":20","");
         //StringReplace(initsl,".",",");
         double initslR = StringToDouble(initsl)/OrderLots()/lotsize/wsp;
         
         R = initslR;
         //R = OrderOpenPrice() - (Low[orderpos+1]-slippageTICK_sl*ticksize);
         
         
         strObjectName = sObject+" RB1_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB2_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+2*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB3_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+3*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB4_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+4*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB5_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+5*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB6_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+6*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB7_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+7*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB8_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+8*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB9_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+9*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB10_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+10*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB11_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+11*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB12_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+12*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB13_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+13*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RB14_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()+14*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);


         strObjectName = sObject+" EntryB"+i;
         ss = StringSetChar(ss, 0, 94);
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_TEXT, 0, Time[orderpos], OrderOpenPrice());
         //ObjectSetText(strObjectName,DoubleToStr(i,0),10,"Arial Black",Red);   
         ObjectSetText(strObjectName,ss,10,"Arial Black",Red);   



         
         result=True;
        }
      if(OrderType()==OP_SELL)
        {
         wholeSL = wholeSL + (OrderOpenPrice() - OrderStopLoss())*OrderLots()*lotsize*wsp;
         WriteInfo("R_S: "+OrderComment()+"   SL: "+DoubleToStr((OrderOpenPrice() - OrderStopLoss())*OrderLots()*lotsize*wsp,2)+"        A: "+DoubleToStr(OrderProfit(),2),"",Red);     

         orderpos = iBarShift(Symbol(),Period(),OrderOpenTime(),False);

         initsl = OrderComment();
         StringReplace(initsl,":1","");
         StringReplace(initsl,":2","");
         StringReplace(initsl,":3","");
         StringReplace(initsl,":20","");
         //StringReplace(initsl,".",",");
         initslR = StringToDouble(initsl)/OrderLots()/lotsize/wsp;
         
         R = initslR;
         //R = (High[orderpos+1]+slippageTICK_sl*ticksize) - OrderOpenPrice();

         strObjectName = sObject+" RS1_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()-R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RS2_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()-2*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RS3_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()-3*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RS4_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()-4*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" RS5_"+i;
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_HLINE, 0, OrderOpenTime(), OrderOpenPrice()-5*R);
         ObjectSet(strObjectName, OBJPROP_STYLE, STYLE_SOLID);
         ObjectSet(strObjectName, OBJPROP_COLOR, Blue);
         ObjectSet(strObjectName, OBJPROP_WIDTH, 0);

         strObjectName = sObject+" EntryS"+i;
         ss = StringSetChar(ss, 0, 118);
         ObjectDelete(strObjectName);
         ObjectCreate(strObjectName, OBJ_TEXT, 0, Time[orderpos], OrderOpenPrice());
         //ObjectSetText(strObjectName,DoubleToStr(i,0),10,"Arial Black",Red);   
         ObjectSetText(strObjectName,ss,10,"Arial Black",Red);   

         result=True;
        }
     }
   if (result == True)
      WriteInfo("Total:   "+DoubleToStr(wholeSL,2),"",Red);

   return (result);
   
}

void WriteInfo(string text1,string text2,color dColor = Black)
{
   int iLabelCorner = 0;
   int xx1;
   int xx2;
   int tmpxx;

   int xx3 = 200;
   int xx4 = 300;
   //int yy0 = 13;
   //int yy1;
   int yStep = 14;
   int iWindow = 0;//WindowFind(isn);

   iLabelCorner = e_Corner;
   //yy = yy0;
   //yy1 = yy;
   xx2 = e_xx;
   xx1 = xx2 + 80;
   
   if (iLabelCorner == 1 ){
     tmpxx = xx1;
     xx1   = xx2;
     xx2   = tmpxx;
   } 
   
//   text1 = "DIFF:";
//   text2 = DoubleToStr(Line1[0],2);
   Write_Label(iWindow, iLabelCorner, sObject, text1, text2, xx2, xx1, yy,dColor); yy = yy + yStep;

}

//####################################################################
//+------------------------------------------------------------------+
//    Write Label
//+------------------------------------------------------------------+
void Write_Label(int iWindow, int iLabelCorner, string sObject, 
                 string text1, string text2, int xx1, int xx2, int yy,color dColor)
{
   int fontSize_Text = 8;
   
   string name1 = sObject + "1" + yy + xx1;
   string name2 = sObject + "2" + yy + xx2;
   SetLabelObject(iWindow, iLabelCorner, name1, text1, fontSize_Text, dColor, xx1, yy);
   SetLabelObject(iWindow, iLabelCorner, name2, text2, fontSize_Text, dColor, xx2, yy);
   return;
}


//+------------------------------------------------------------------+
//    Set Label Object
//+------------------------------------------------------------------+
void SetLabelObject(int iWindow, int iLabelCorner, string sName, string sText, 
                    int fontSize_Text, color dColor, int xx, int yy)
{
   ObjectCreate(sName, OBJ_LABEL, iWindow, 0, 0);
        ObjectSetText(sName,sText,fontSize_Text, "Calibri", dColor);
        ObjectSet(sName, OBJPROP_CORNER, iLabelCorner);
        ObjectSet(sName, OBJPROP_XDISTANCE, xx);
        ObjectSet(sName, OBJPROP_YDISTANCE, yy);
   return;     
}

//+------------------------------------------------------------------+
//    Delete Own Objects
//+------------------------------------------------------------------+
void DeleteOwnObjects(string sObject)
{
   int i=0;
   while (i <= ObjectsTotal()) {
      if (StringFind(ObjectName(i), sObject) >= 0) ObjectDelete(ObjectName(i));
      else
      i++;
   }
   return;
}






