import numpy as np
import pandas as pd
from scipy import stats
import scipy.optimize
from scipy.optimize  import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
from arch import arch_model
import datetime as dt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

import pandas as pd
from pandasql import sqldf

'''
from numpy import mean
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Integer
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from warnings import catch_warnings
from warnings import simplefilter
'''
def opentrades(mode,df,hours):
    df['tradetype'] = 0
    
    if (mode==0):
        df.loc[(df.hour.isin(hours)) & (df.tdi13green_slope>=0),'tradetype'] = 1
        df.loc[(df.hour.isin(hours)) & (df.tdi13green_slope<0),'tradetype'] = -1
    else: 
        df.loc[df.hour.isin(hours),'tradetype'] = mode
    df.loc[df.tradetype!=0,'openprice'] = df.open
    df.loc[df.tradetype!=0,'openindex'] = df.id
    return df

def closetrades(df,hour,stoploss):
    """
    df['closeindex'] = -1
    df['closeprice'] = -1
    
    df['closeindex'] = pd.Series(
        findclose(row.openindex,hour,df)
        for row in df.itertuples()  
    )
    df['closeprice']=pd.merge(df,df[['id','open']], left_on='closeindex', right_on='id', how='left')[['open_y']]
    """
    
    df['sl'] = stoploss
    df['closeindex'] = -1
    df['closeprice'] = -1
    df['closehour'] = -1
    df['slindex'] = -1
    df['slprice'] = -1
    df['profit'] = 0
    
    lastclose = df[df.hour==hour].tail(1).id.values[0] - 1
    i = 0
    while ((len(df[(df.tradetype!=0) & (df.closeindex==-1) & (df.id<=lastclose)])>0) & (i>=-20)):
#         print(i,len(df[(df.tradetype!=0) & (df.closeindex==-1) & (df.id<=lastclose)]))
        df['nextbar_hour'] = df.hour.shift(i)
        df['nextbar_open'] = df.open.shift(i)
        df['nextbar_low'] = df.low.shift(i)
        df['nextbar_high'] = df.high.shift(i)
        df['nextbar_id'] = df.id.shift(i)
        #SL buy
        df.loc[(df.tradetype==1) & (df.nextbar_hour != hour) & (df.closeindex==-1)&(df.nextbar_low-df.openprice<=-stoploss),'slprice'] = df.nextbar_low
        df.loc[(df.tradetype==1) & (df.nextbar_hour != hour) & (df.closeindex==-1)&(df.nextbar_low-df.openprice<=-stoploss),'slindex'] = df.nextbar_id
        df.loc[(df.tradetype==1) & (df.nextbar_hour != hour) & (df.closeindex==-1)&(df.nextbar_low-df.openprice<=-stoploss),'closeprice'] = df.nextbar_low
        df.loc[(df.tradetype==1) & (df.nextbar_hour != hour) & (df.closeindex==-1)&(df.nextbar_low-df.openprice<=-stoploss),'profit'] = -stoploss
        df.loc[(df.tradetype==1) & (df.nextbar_hour != hour) & (df.closeindex==-1)&(df.nextbar_low-df.openprice<=-stoploss),'closeindex'] = df.nextbar_id

        #SL sell
        df.loc[(df.tradetype==-1) & (df.nextbar_hour != hour) & (df.closeindex==-1)&(df.nextbar_high-df.openprice>=stoploss),'slprice'] = df.nextbar_high
        df.loc[(df.tradetype==-1) & (df.nextbar_hour != hour) & (df.closeindex==-1)&(df.nextbar_high-df.openprice>=stoploss),'slindex'] = df.nextbar_id
        df.loc[(df.tradetype==-1) & (df.nextbar_hour != hour) & (df.closeindex==-1)&(df.nextbar_high-df.openprice>=stoploss),'closeprice'] = df.nextbar_high
        df.loc[(df.tradetype==-1) & (df.nextbar_hour != hour) & (df.closeindex==-1)&(df.nextbar_high-df.openprice>=stoploss),'profit'] = -stoploss
        df.loc[(df.tradetype==-1) & (df.nextbar_hour != hour) & (df.closeindex==-1)&(df.nextbar_high-df.openprice>=stoploss),'closeindex'] = df.nextbar_id
        
        df.loc[(df.tradetype!=0) & (df.nextbar_hour == hour) & (df.closeindex==-1),'closeprice'] = df.nextbar_open
        df.loc[(df.tradetype==1) & (df.nextbar_hour == hour) & (df.closeindex==-1),'profit'] = df.nextbar_open-df.openprice
        df.loc[(df.tradetype==-1) & (df.nextbar_hour == hour) & (df.closeindex==-1),'profit'] = -(df.nextbar_open-df.openprice)
        df.loc[(df.tradetype!=0) & (df.nextbar_hour == hour) & (df.closeindex==-1),'closeindex'] = df.nextbar_id
        i-=1
    
    df.loc[(df.closeindex!=-1),'closehour'] = hour

    df['profit'] = df.profit * 10000
    df['profit1'] = np.where(df.profit>=20,1,-1)

    df['sl'] = df.sl * 10000
    df['tdi13habarsize1'] = df.tdi13habarsize1 * 10000
    df['tdi13habarsize2'] = df.tdi13habarsize2 * 10000
    
    
    df = df.drop(['nextbar_hour'],1)
    df = df.drop(['nextbar_open'],1)
    df = df.drop(['nextbar_low'],1)
    df = df.drop(['nextbar_high'],1)
    df = df.drop(['nextbar_id'],1)
    
    return df

def cleartrades(df,save=False):
    
    if (save==True):
        df.to_csv(sep=';',path_or_buf='../Data/trades.csv',date_format="%Y-%m-%d",index = False,na_rep='')
    
    df = df[df.closeindex!=-1]
    df = df.drop(['date'],1)
    df = df.drop(['year'],1)
    df = df.drop(['open'],1)
    df = df.drop(['low'],1)
    df = df.drop(['high'],1)
    df = df.drop(['close'],1)
    df = df.drop(['volume'],1)
    df = df.drop(['openprice'],1)
    df = df.drop(['openindex'],1)
    df = df.drop(['closeindex'],1)
    df = df.drop(['closeprice'],1)
    df = df.drop(['slindex'],1)
    df = df.drop(['slprice'],1)
    df = df.drop(['id'],1)

    df = df.drop(['profit'],1)
    
#
    
#     df = df.drop(['tdi13green1_red1'],1)
#     df = df.drop(['tdi13green2_red2'],1)
#     df = df.drop(['tdi13green_red_change'],1)
#     df = df.drop(['tdi13green_red_slope_change'],1)
#     df = df.drop(['tdi13rsi1'],1)
#     df = df.drop(['tdi13green1'],1)
#     df = df.drop(['tdi13red1'],1)
#     df = df.drop(['tdi13green2'],1)
#     df = df.drop(['tdi13red2'],1)
#     df = df.drop(['tdi13green_red_mul'],1)

#     df = df.drop(['tdi13red_slope'],1)
#     df = df.drop(['tdi13green_slope'],1)

    df = df.drop(['tdi13haclose2'],1)
    df = df.drop(['tdi13haopen2'],1)

    df = df.drop(['tdi13haclose1'],1)
    df = df.drop(['tdi13haopen1'],1)

    df = df.drop(['tdi13rsi'],1)
    df = df.drop(['tdi13red'],1)
    df = df.drop(['tdi13green'],1)
    df = df.drop(['tdi13mid'],1)
    df = df.drop(['tdi13haclose'],1)
    df = df.drop(['tdi13haopen'],1)
    df = df.drop(['tdi13hacolor'],1)
    df = df.drop(['tdi13barnumber'],1)

    
#     df = df[(df.tdi13habarsize1>-20)&(df.tdi13habarsize1<20)]
#     df = df[(df.tdi13habarsize2>-20)&(df.tdi13habarsize2<20)]
    df = df[(df.tdi13habarsize1<-10)|(df.tdi13habarsize1>10)]
    df = df[(df.tdi13habarsize2<-10)|(df.tdi13habarsize2>10)]
    df = df[(df.tradetype>0)]

#     df = df[(df.tdi13red_slope>-2.332) & (df.tdi13red_slope<=0.087)]
#     df = df[(df.tdi13green_slope<=3.437)]
#      df = df[(df.tdi13habarsize1>-2.519)&(df.tdi13habarsize1<=8.613)]
    
    
    
    return df


def preparetrades(masterFrame, trtypes,openhours,closehours,sls):
    first = True
    for trtype in trtypes:
        for closehour in closehours:                
            for sl in sls:                
                df = masterFrame.copy()
                df = opentrades(trtype,df,openhours)
                df = closetrades(df,closehour,sl)
                if (first==True): 
                    trades=df 
                    first = False
                else:
                    trades=trades.append(df)
    return trades


def findclose(openindex,hour,masterFrame):
    closeid = masterFrame[(masterFrame.hour==hour)&(masterFrame.id>openindex)].head(1) 
    if (len(closeid)>0):
         closeid = closeid.id.values[0]
    else: 
        closeid = -1
    return closeid



def stattrades(trades):

#     stats = trades.drop_duplicates(subset = ['hour','closehour','sl'])[['hour','closehour','sl']]
    trades = trades[trades.tradetype!=0]
    statsgb = trades.groupby(['tradetype','hour','closehour','sl'])
    stats = statsgb.size().to_frame(name='counts')
    stats = stats.join(statsgb.agg({'profit': 'mean'}).rename(columns={'profit': 'profit_mean'}))
    stats = stats.join(statsgb.agg({'profit': 'sum'}).rename(columns={'profit': 'profit_sum'}))
    stats = stats.reset_index()
    
    stats['profit_ratio'] = stats.profit_sum/stats.sl
    
    grouping = statsgb
    maxdowns = [find_maxdownseries(i).values() for i in grouping]
    maxdowns_df =  pd.DataFrame(data = maxdowns, index = grouping.groups, columns = ['maxdown']).reset_index()    
    
    stats = stats.merge(maxdowns_df,left_on=['tradetype','hour','closehour','sl'], right_on = ['level_0','level_1','level_2','level_3'], how='left')
    stats['maxdown_ratio'] = stats.maxdown/stats.sl

    stats = stats.drop(['level_0','level_1','level_2','level_3'],1)
    
    return stats

def stathyper(trades,tradetypes,openhours,closehours,sls,bar2froms,bar2tos,bar1froms,bar1tos,gr2froms,gr2tos,gr1froms,gr1tos):
    trades = trades[trades.tradetype!=0]
    
    stats = pd.DataFrame(columns=['tradetype','openhour','closehour','sl','bar2from','bar2to','bar1from','bar1to','gr2from','gr2to','gr1from','gr1to','count','mean','profit_sum','maxdown'])
    for tradetype in tradetypes:                
        print('tradetype',tradetype)
        for openhour in openhours:                
            print('  openhour',openhour)
            for closehour in closehours:                
                print('    closehour',closehour)
                for sl in sls:                
                    print('      sl',sl)
                    for bar2from in bar2froms:
                        print('        bar2from',bar2from)
                        for bar2to in bar2tos:
                            print('          bar2to',bar2to)
                            if (bar2to>bar2from):
                                for bar1from in bar1froms:
                                    print('            bar1from',bar1from)
                                    for bar1to in bar1tos:
                                        print('              bar1to',bar1to)
                                        if (bar1to>bar1from):
                                            for gr2from in gr2froms:
                                                print('                gr2from',gr2from)
                                                for gr2to in gr2tos:
                                                    if(gr2to>gr2from):
                                                        print('                  gr2to',gr2to)
                                                        for gr1from in gr1froms:
                                                            for gr1to in gr1tos:
                                                                if(gr1to>gr1from):
                                                                    df,stats0 = calculatestats(trades,tradetype,openhour,closehour,sl,bar2from,bar2to,bar1from,bar1to,gr2from,gr2to,gr1from,gr1to)
                                                                    stats = stats.append(df,sort=False)
    stats['profit_ratio'] = stats.profit_sum/stats.sl
    stats['maxdown_ratio'] = stats.maxdown/stats.sl
    return stats,stats0

def calculatestats(trades,tradetype,openhour,closehour,sl,bar2from,bar2to,bar1from,bar1to,gr2from,gr2to,gr1from,gr1to):
    stats0 = trades[trades.tradetype==tradetype]
    stats0 = stats0[stats0.hour==openhour]
    stats0 = stats0[stats0.closehour==closehour]
    stats0 = stats0[stats0.sl==sl]
    stats0 = stats0[(stats0.tdi13habarsize2>=bar2from)&(stats0.tdi13habarsize2<=bar2to)]
    stats0 = stats0[(stats0.tdi13habarsize1>=bar1from)&(stats0.tdi13habarsize1<=bar1to)]
    stats0 = stats0[(stats0.tdi13green2_red2>=gr2from)&(stats0.tdi13green2_red2<=gr2to)]
    stats0 = stats0[(stats0.tdi13green1_red1>=gr1from)&(stats0.tdi13green1_red1<=gr1to)]
    pr_c = len(stats0)
    pr_mean = stats0.profit.mean()
    pr_sum = stats0.profit.sum()
    pr_maxdown = (stats0.groupby((stats0['profit'] * stats0['profit'].shift(1) <=0).cumsum())['profit'].cumsum()).min()
    df = pd.DataFrame(data={'tradetype':tradetype,'openhour':openhour,'closehour':closehour,'sl':sl,'bar2from':bar2from,'bar2to':bar2to,'bar1from':bar1from,'bar1to':bar1to,'gr2from':gr2from,'gr2to':gr2to,'gr1from':gr1from,'gr1to':gr1to,'count':pr_c,'mean':pr_mean,'profit_sum':pr_sum,'maxdown':pr_maxdown}, index=[0])
    return df,stats0

def find_maxdownseries(grouping):
    (group_label, df) = grouping 
    maxdownseries = (df.groupby((df['profit'] * df['profit'].shift(1) <=0).cumsum())['profit'].cumsum()).min()
    return({group_label: maxdownseries})

#Bayesian Optimization
    
"""    
search_space = [
                Categorical(alltrades,name ='trades')
                ,Integer(2003,2020, name='year')
                ,Categorical((13,17), name='closehour')
                ,Categorical((10,20,30), name='sl')
               ]
@use_named_args(search_space)
def evaluate_model(**params):
    trades = params['trades']
    iyear= params['year']
    iclosehour= params['closehour']
    isl = params['sl']
    print(iyear,iclosehour,isl)
    
    trades = alltrades
    trades = trades[trades.tradetype!=0]
    trades = trades[trades.year==iyear]
    trades = trades[trades.closehour==iclosehour]
    trades = trades[trades.sl==isl]
    
    statsgb = trades.groupby(['year','closehour','sl'])
    stats = statsgb.size().to_frame(name='counts')
    stats = stats.join(statsgb.agg({'profit': 'sum'}).rename(columns={'profit': 'profit_sum'}))
    stats = stats.reset_index()
    
    if (len(stats)>0): 
        isum = -max(stats.profit_sum)
    else:
        isum = 1000
    print(isum)
    return isum  

with catch_warnings():
      # ignore generated warnings
    simplefilter("ignore")

    result = gp_minimize(evaluate_model, search_space,n_calls=20)
# summarizing finding:
print('Best Accuracy: %.3f' % (result.fun))
print('Best Parameters: year=%d,closehour=%d, sl=%d' % (result.x[1], result.x[2], result.x[3]))
"""    
    
    
    