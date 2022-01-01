//+------------------------------------------------------------------+
//|                                                  MACD Sample.mq4 |
//|                   Copyright 2005-2014, MetaQuotes Software Corp. |
//|                                              http://www.mql4.com |
//+------------------------------------------------------------------+
#property copyright   "2005-2014, MetaQuotes Software Corp."
#property link        "http://www.mql4.com"

input double StopLoss      =0.6;
input double TakeProfit    =0.8;
input double Lots          =0.1;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick(void)
  {
   double Signall;
   int ticket;
   double atr;
   double StopLossV,TakeProfitV;
   bool neworder;
   int magic_no;
//---
   Signall = iCustom(NULL,0,"Signall",0,1);
   magic_no = 43251;
   if(Signall==1)
   {
      atr = iATR(NULL,0,20,1);
      StopLossV = StopLoss * atr;
      TakeProfitV = TakeProfit * atr;
      neworder= NoTradesToday(magic_no);

      if(neworder)
      {
         //Print(atr);
         //
         if(AccountFreeMargin()<(1000*Lots))
         {
            Print("We have no money. Free Margin = ",AccountFreeMargin());
            return;
         }
         //--- check for long position (BUY) possibility
         if(Signall==1)
         {
            ticket=OrderSend(Symbol(),OP_BUY,Lots,Ask,3,Bid-StopLossV,Bid+TakeProfitV,"ma12rsi",magic_no,0,Green);
//            if(ticket>0)
//            {
//              if(OrderSelect(ticket,SELECT_BY_TICKET,MODE_TRADES))
//                  Print("BUY order opened : ",OrderOpenPrice());
//                  
//            }
//            else
//            {
//               Print("Error opening BUY order : ",GetLastError());
//              
//            }   
//            return;
         }
      }
   }  
//--- it is important to enter the market correctly, but it is more important to exit it correctly...   
//---
  }
//+------------------------------------------------------------------+


bool NoTradesToday(int magic_no)
  {
   datetime today = iTime(NULL,0,0);

   for(int i=OrdersHistoryTotal()-1; i>=0; i--)
     {
      if(!OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)) continue;
      if(OrderSymbol()      != _Symbol)  continue;
      if(OrderMagicNumber() != magic_no) continue;
      if(OrderOpenTime()    >= today)    return(false);
     }

   for(i=OrdersTotal()-1; i>=0; i--)           
     {
      if(!OrderSelect(i,SELECT_BY_POS))  continue;  
      if(OrderSymbol()      != _Symbol)  continue;  
      if(OrderMagicNumber() != magic_no) continue;  
      if(OrderOpenTime()    >= today)    return(false);
     }

   return(true);
  }