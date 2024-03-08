import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
# import math
from datetime import timedelta
import itertools
import json 
import matplotlib.pyplot as plt

# from scipy import stats
# import scipy.optimize
# from scipy.optimize  import OptimizeWarning
# import warnings
# import math
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from datetime import datetime
# from arch import arch_model
# import datetime as dt
# from statsmodels.tsa.arima_model import ARIMA
# import statsmodels.api as sm
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler
# from datetime import timedelta

# import pandas as pd
# from pandasql import sqldf

# from numpy import mean
# from sklearn.datasets import make_blobs
# from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier
# from skopt.space import Integer
# from skopt.space import Categorical
# from skopt.utils import use_named_args
# from skopt import gp_minimize
# from warnings import catch_warnings
# from warnings import simplefilter

class holder:
    1
    
def timedump(stamp):
    return    

# from time import time
# global tt
# tt = time()

# def timedump(stamp):
#     global tt
#     print(stamp,time()-tt)
#     tt=time()


    
# ducascopy
# BID EET GMT
# do daty trzeba dodać jeden dzień
# Ostatni w pliku pokazuje aktualny kurs z bieżącego dnia.
# date,open,high,low,close,volume
# 02.08.2003 21:00:00.000,0.65130,0.65130,0.65130,0.65130,0

    
def loaddata_4h(datafile):
    df = pd.read_csv('../Data/'+datafile)
    try:
        df.date=pd.to_datetime(df.date,format='%Y-%m-%d')
    except:
        #df.date=pd.to_datetime(df.date,format='%Y.%m.%d')    
        df.date=pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')    

    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['week'] = pd.DatetimeIndex(df['date']).week
    df['day'] = pd.DatetimeIndex(df['date']).day 
    df['hour'] = pd.DatetimeIndex(df['date']).hour
#     df['weekday'] = df.index.map(lambda v: pd.to_datetime(v).isocalendar()[1])   
    df['weekday'] = df.date.dt.dayofweek
    df.loc[df.hour==2,'hour']=1
    df.loc[df.hour==6,'hour']=5
    df.loc[df.hour==10,'hour']=9
    df.loc[df.hour==14,'hour']=13
    df.loc[df.hour==18,'hour']=17
    df.loc[df.hour==22,'hour']=21
    df=df[-(((df.weekday==5)|(df.weekday==6))&(df.volume==0))] # jedna świeca w niedziele zostaje.
    df.reset_index(inplace = True, drop = True)
    df['id'] = df.index    
    return df

def loaddata_1D_old(datafile):
    df = pd.read_csv('../Data/'+datafile)
    try:
        df.date=pd.to_datetime(df.date,format='%Y-%m-%d')
    except:
        #df.date=pd.to_datetime(df.date,format='%Y.%m.%d')    
        df.date=pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')    

    df.date = df.date + timedelta(days=1) # w oryginale jest godzina 22:00 więc to już następny dzień

    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
#     df['week'] = pd.DatetimeIndex(df['date']).week
#     df['week'] = pd.Int64Index(df.date.isocalendar().week)
    df['week'] = pd.DatetimeIndex(df.date).isocalendar().week
    df['day'] = pd.DatetimeIndex(df['date']).day 
    df['weekday'] = df.date.dt.dayofweek
    df=df[df.volume!=0] 
    df.reset_index(inplace = True, drop = True)
    df['id'] = df.index    
    return df

def loaddata_1W(datafile):
    df = pd.read_csv('../Data/'+datafile)
    df.rename(columns={'Data': 'date', 'Otwarcie': 'open', 'Zamkniecie': 'close', 'Najwyzszy': 'high', 'Najnizszy': 'low', 'Wolumen': 'volume'}, inplace=True)
    try:
        df.date=pd.to_datetime(df.date,format='%Y-%m-%d')
    except:
        df.date=pd.to_datetime(df.date,format='%Y.%m.%d')    

    df['date'] = df.date - timedelta(days=6)
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['week'] = pd.DatetimeIndex(df.date).isocalendar().week
    df['day'] = pd.DatetimeIndex(df['date']).day 
    df['weekday'] = df.date.dt.dayofweek
    df=df[df.volume!=0] 
    df.reset_index(inplace = True, drop = False)
    df['id'] = df.index    
    return df

def loaddata_1D(datafile):
    df = pd.read_csv('../Data/'+datafile)
    df.rename(columns={'Data': 'date', 'Otwarcie': 'open', 'Zamkniecie': 'close', 'Najwyzszy': 'high', 'Najnizszy': 'low', 'Wolumen': 'volume'}, inplace=True)
    try:
        df.date=pd.to_datetime(df.date,format='%Y-%m-%d')
    except:
        df.date=pd.to_datetime(df.date,format='%Y.%m.%d')    

    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['week'] = pd.DatetimeIndex(df.date).isocalendar().week
    df['day'] = pd.DatetimeIndex(df['date']).day 
    df['weekday'] = df.date.dt.dayofweek
    df=df[df.volume!=0] 
    df.reset_index(inplace = True, drop = False)
    df['id'] = df.index    
    return df

def rose(prices,periods, history = 10,ignore = 2, entry_buy_perc = 7, entry_sell_perc = 7, onlytry=0):
    results = holder()
    
    if (onlytry==1): 
        print('08')
        return
    
    df = prices.loc[:,['id','date','close']]

    realtick_up = (100 - ignore)/100 #uptick
    realtick_down = (100 + ignore)/100 #downtick
    
    entry_sell = (100 - entry_sell_perc)/100 #sell
    entry_buy = (100 + entry_buy_perc)/100 #buy
    
    # oznacza 2 gdy close jest większe od poprzednika o realtick (procent)
    df['rose'] = 0
    df.loc[(df.rose==0),'rose'] = 1
    for i in range(1,history+1):
        df['close_prev'] = df['close'].shift(i)
        df.loc[(df.rose==1)&(df.close_prev>df.close),'rose'] = 0
        df.loc[(df.rose==1)&(df.close_prev<=df.close*realtick_up),'rose'] = 2

    # oznacza -2 gdy close jest mniejsz od poprzednika o realtick (procent)
    df.loc[(df.rose==0),'rose'] = -1
    for i in range(1,history+1):
        df['close_prev'] = df['close'].shift(i)
        df.loc[(df.rose==-1)&(df.close_prev<df.close),'rose'] = 0
        df.loc[(df.rose==-1)&(df.close_prev>=df.close*realtick_down),'rose'] = -2

    # uptick downtick
    # dla każdego wiersz szuka najświeższego ticka 2 jak i -2
    # dodatkowy warunek: tick up musi byś wyższy niż aktualne close
    # tick down musi być niżej niż aktualny close
    
    # uptick_any1 uptick_any2
    # downtick_any1 downtick_any2
    # to są ticki bez dodatkowego warunku że uptick musi być wyższy od aktulnego close
    # i że downtick musi być niższy od aktualnego close
    
    history2 = history * 3
    df['uptick'] = 0
    df['uptick_date'] = datetime(1900, 1, 1,0,0,0)
    df['uptick_close'] = 0
    df['uptick_diff'] = 0

    df['downtick'] = 0
    df['downtick_date'] = datetime(1900, 1, 1,0,0,0)
    df['downtick_close'] = 0
    df['downtick_diff'] = 0
    
    df['uptick_a1'] = 0
    df['uptick_a1_close'] = 0
    df['downtick_a1'] = 0
    df['downtick_a1_close'] = 0
    
    df['uptick_a2'] = 0
    df['uptick_a2_close'] = 0
    df['downtick_a2'] = 0
    df['downtick_a2_close'] = 0

    
    for i in range(1,history2+1):
        df['rose_prev'] = df['rose'].shift(i)
        df['date_prev'] = df['date'].shift(i)
        df['close_prev'] = df['close'].shift(i)

        df.loc[(df.uptick==0)&(df.rose_prev==2)&(df.close_prev>df.close),'uptick_date'] =    df.date_prev
        df.loc[(df.uptick==0)&(df.rose_prev==2)&(df.close_prev>df.close),'uptick_close'] =   df.close_prev
        df.loc[(df.uptick==0)&(df.rose_prev==2)&(df.close_prev>df.close),'uptick_diff'] =    (df.close-df.close_prev)/df.close_prev
        df.loc[(df.uptick==0)&(df.rose_prev==2)&(df.close_prev>df.close),'uptick'] =         df.rose_prev

        df.loc[(df.downtick==0)&(df.rose_prev==-2)&(df.close_prev<df.close),'downtick_date'] =   df.date_prev
        df.loc[(df.downtick==0)&(df.rose_prev==-2)&(df.close_prev<df.close),'downtick_close'] =  df.close_prev
        df.loc[(df.downtick==0)&(df.rose_prev==-2)&(df.close_prev<df.close),'downtick_diff'] =   (df.close-df.close_prev)/df.close_prev
        df.loc[(df.downtick==0)&(df.rose_prev==-2)&(df.close_prev<df.close),'downtick'] =        df.rose_prev

        df.loc[(df.uptick_a2==0)&(df.downtick_a1==-2)&(df.rose_prev==2),'uptick_a2_close'] =   df.close_prev
        df.loc[(df.uptick_a2==0)&(df.downtick_a1==-2)&(df.rose_prev==2),'uptick_a2'] =         df.rose_prev
        df.loc[(df.downtick_a2==0)&(df.uptick_a1==2)&(df.rose_prev==-2),'downtick_a2_close'] =   df.close_prev
        df.loc[(df.downtick_a2==0)&(df.uptick_a1==2)&(df.rose_prev==-2),'downtick_a2'] =         df.rose_prev

        df.loc[(df.uptick_a1==0)&(df.rose_prev==2),'uptick_a1_close'] =   df.close_prev
        df.loc[(df.uptick_a1==0)&(df.rose_prev==2),'uptick_a1'] =         df.rose_prev
        df.loc[(df.downtick_a1==0)&(df.rose_prev==-2),'downtick_a1_close'] =   df.close_prev
        df.loc[(df.downtick_a1==0)&(df.rose_prev==-2),'downtick_a1'] =         df.rose_prev

    
    
    df['entry'] = 0
    # wejście gdy ostatnio była górka i spadło poniżej entry level
    df.loc[(df.downtick_date>df.uptick_date)&(df.close>df.downtick_close*entry_buy),'entry'] = 1
    # wyjście gdy ostatnio był dołek i wzrosło powyżej entry level
    df.loc[(df.uptick_date>df.downtick_date)&(df.close<df.uptick_close*entry_sell),'entry'] = -1
    

    '''
    pętla idzie po kolei od początku historii
    szuka tradeup i do niego trade down
    potem kolejnego tradeup i do niego trade down
    zapisuje entry_date, close_date, zysk/stratę w procentach
    '''
    df['open_trade'] = 0
    df['close_trade'] = 0
    
    entry_state = -1
    entry_date = datetime(1900,1,1,0,0,0)
    entry_close = 0
    climate = -1
    df['open_trade_date'] = entry_date
    df['open_trade_close'] = entry_close
    # climate  1 gdy jesteśmy wyżej niż ostatni szczyt - od tej pory można wchodzić
    # climate -1 gdy jestśmy niżej niż ostatni dołek - od tej pory nie można wchodzić
    df['climate'] = 0
    df.loc[(df.close>df.uptick_a2_close)&(df.uptick_a2_close!=0),'climate'] = 1
    df.loc[(df.close<df.downtick_a2_close)&(df.downtick_a2_close!=0),'climate'] = -1
    for index, row in df.iterrows():
        entry = row['entry']
        if (row['climate'] != 0): 
            climate = row['climate']
        if (entry_state == -1) & (entry == 1) & (climate == 1):
            entry_state = 1
            entry_date = row['date']
            entry_close = row['close']
            df.loc[df.id == row['id'],'open_trade'] = row['close']#dowykresu
        if (entry_state == 1) & (entry == -1):
            entry_state = -1
            df.loc[df.id == row['id'],'open_trade_date'] = entry_date
            df.loc[df.id == row['id'],'open_trade_close'] = entry_close
            df.loc[df.id == row['id'],'close_trade'] = row['close']#dowykresu
            
    
    df['profit'] = 0
    df.loc[df.open_trade_close != 0,'profit'] = (df.close - df.open_trade_close)/df.open_trade_close
    
    #do wykresu
    df['rosehigh'] = 0
    df['roselow'] = 0
    df.loc[df.rose == 2,'rosehigh'] = df.close
    df.loc[df.rose == -2,'roselow'] = df.close
    df['climateU'] = 0
    df['climateD'] = 0
    df.loc[df.climate == 1,'climateU'] = df.close
    df.loc[df.climate == -1,'climateD'] = df.close
    
     
    df = df.drop(columns='id')
    df = df.drop(columns='date')
    df = df.drop(columns='rose_prev')
    df = df.drop(columns='date_prev')
    df = df.drop(columns='close_prev')
    df = df.drop(columns='close')
    df = df.drop(columns='uptick')
    df = df.drop(columns='downtick')
    '''
    df = df.drop(columns='low_prev')
    df = df.drop(columns='close_prev')
    '''
    dict = {}
    dict[periods[0]] = df
    results.df = dict
    return results




def rsi(prices, periods):
    """
    Returns a pd.Series with the relative strength index.
    """
    ema = True
    results = holder()
    dict = {}
    for i in range(0,len(periods)):
        period = periods[i]
        rsidf = pd.DataFrame()

        # Make two series: one for lower closes and one for higher closes
        rsidf['close_delta'] = prices['close'].diff()
        rsidf['up'] = rsidf.close_delta.clip(lower=0)
        rsidf['down'] = -1 * rsidf.close_delta.clip(upper=0)

        # Use exponential moving average
        rsidf['ma_up'] = rsidf.up.ewm(com = period-1,adjust=False).mean()
        rsidf['ma_down'] = rsidf.down.ewm(com = period-1,adjust=False).mean()
        rsidf['rsi'] = 100 - (100/(1 + rsidf['ma_up'] / rsidf['ma_down']))

        rsidf['ma_up1'] = 0
        rsidf['ma_down1'] = 0
        rsidf['ma_up1'] = rsidf.ma_up.shift(1)*(1-1/period) # this is faster than for loop
        rsidf['ma_down1'] = rsidf.ma_down.shift(1)*(1-1/period)
        rsidf['rsi1'] = 100 - (100/(1 + rsidf['ma_up1'] / rsidf['ma_down1']))


        rsidf['rsidiff1n'] = rsidf.rsi.diff(1)
        rsidf['rsidiff2n'] = rsidf.rsi.shift(1) - rsidf.rsi.shift(2)
        rsidf['rsidiff3n'] = rsidf.rsi.shift(2) - rsidf.rsi.shift(3)
        rsidf['rsidiff4n'] = rsidf.rsi.shift(3) - rsidf.rsi.shift(4)
        rsidf['rsidiff5n'] = rsidf.rsi.shift(4) - rsidf.rsi.shift(5)

        rsidf['rsidiffseq'] = 0
        rsidf.loc[rsidf.rsidiff1n>=0,'rsidiffseq'] = 1
        rsidf.loc[(rsidf.rsidiff1n>=0) & (rsidf.rsidiff2n>=0),'rsidiffseq'] = 2
        rsidf.loc[(rsidf.rsidiff1n>=0) & (rsidf.rsidiff2n>=0) & (rsidf.rsidiff3n>=0),'rsidiffseq'] = 3
        rsidf.loc[(rsidf.rsidiff1n>=0) & (rsidf.rsidiff2n>=0) & (rsidf.rsidiff3n>=0) & (rsidf.rsidiff4n>=0),'rsidiffseq'] = 4
        rsidf.loc[(rsidf.rsidiff1n>=0) & (rsidf.rsidiff2n>=0) & (rsidf.rsidiff3n>=0) & (rsidf.rsidiff4n>=0) & (rsidf.rsidiff5n>=0),'rsidiffseq'] = 5
        rsidf.loc[rsidf.rsidiff1n<0,'rsidiffseq'] = -1
        rsidf.loc[(rsidf.rsidiff1n<0) & (rsidf.rsidiff2n<0),'rsidiffseq'] = -2
        rsidf.loc[(rsidf.rsidiff1n<0) & (rsidf.rsidiff2n<0) & (rsidf.rsidiff3n<0),'rsidiffseq'] = -3
        rsidf.loc[(rsidf.rsidiff1n<0) & (rsidf.rsidiff2n<0) & (rsidf.rsidiff3n<0) & (rsidf.rsidiff4n<0),'rsidiffseq'] = -4
        rsidf.loc[(rsidf.rsidiff1n<0) & (rsidf.rsidiff2n<0) & (rsidf.rsidiff3n<0) & (rsidf.rsidiff4n<0) &(rsidf.rsidiff5n<0),'rsidiffseq'] = -5

        rsidf['rsi_prev'] = rsidf.rsi.shift(1)
        rsidf['rsidiffseq_prev'] = rsidf.rsidiffseq.shift(1)
#         rsidf['rsidiff1n_prev'] = rsidf.rsidiff1n.shift(1)

        rsidf = rsidf.drop(columns='close_delta')
        rsidf = rsidf.drop(columns='up')
        rsidf = rsidf.drop(columns='down')
        rsidf = rsidf.drop(columns='ma_up')
        rsidf = rsidf.drop(columns='ma_down')
        rsidf = rsidf.drop(columns='ma_up1')
        rsidf = rsidf.drop(columns='ma_down1')

        rsidf = rsidf.drop(columns='rsi')
        rsidf = rsidf.drop(columns='rsidiff1n')
        rsidf = rsidf.drop(columns='rsidiff2n')
        rsidf = rsidf.drop(columns='rsidiff3n')
        rsidf = rsidf.drop(columns='rsidiff4n')
        rsidf = rsidf.drop(columns='rsidiff5n')
        rsidf = rsidf.drop(columns='rsidiffseq')
    
        dict[periods[i]] = rsidf
    results.df = dict
    return results


def tdi(prices,periods):
    greenperiod = 2
    redperiod = 7
    results = holder()
    df = rsi(prices,[periods[0]]).df[periods[0]]
    df['red'] = df.rsi.rolling(redperiod).mean()
    df['green'] = df.rsi.rolling(greenperiod).mean()
    df['green_red'] = df.green - df.red
    df['green_red_delta'] = df.green_red.diff()
    df['green_green'] = df.green.diff()
    df['haclose'] = prices[['close','high','low','open']].mean(axis = 1) 
    df['haopen'] = prices[['close','open']].mean(axis = 1) 
    df['haopen'] = (df.haopen.shift(1) + df.haclose.shift(1)) / 2
    df['hacolor'] = np.where(df.haclose>=df.haopen,1,-1)
    df['barnumber'] = df.groupby((df['hacolor'] != df['hacolor'].shift(1)).cumsum()).cumcount()+1
    df = df.drop(columns='rsi1')

    dict = {}
    dict[periods[0]] = df
    results.df = dict
    return results


def tdi1(prices,periods):
    greenperiod = 2
    redperiod = 7
    midperiod = 31
    results = holder()
    df = rsi(prices,[periods[0]]).df[periods[0]]
    df['red'] = df.rsi.rolling(redperiod).mean()
    df['green'] = df.rsi.rolling(greenperiod).mean()
    df['mid'] = df.rsi.rolling(midperiod).mean()
    df['red1'] = df.rsi1/redperiod + df.rsi.shift(1).rolling(redperiod-1).sum()/redperiod
    df['green1'] = df.rsi1/greenperiod + df.rsi.shift(1).rolling(greenperiod-1).sum()/greenperiod
    df['mid1'] = df.rsi1/midperiod + df.rsi.shift(1).rolling(midperiod-1).sum()/midperiod
    df['green1_red1'] = df.green1 - df.red1
    df['red2'] = df.red.shift(1)
    df['red3'] = df.red.shift(2)
    df['green2'] = df.green.shift(1)
    df['green3'] = df.green.shift(2)
    df['mid2'] = df.mid.shift(1)
    df['green2_red2'] = df.green2 - df.red2
    df['green3_red3'] = df.green3 - df.red3
    df['green_red_change'] = df.green1_red1 - df.green2_red2
    df['green_red_change2'] = df.green2_red2 - df.green3_red3
    df['green_red_mul'] = df.green1_red1 * df.green2_red2
    df['green_red_cross'] = np.where(df.green_red_mul<=0,1,0)
    df['green_red_mul2'] = df.green2_red2 * df.green3_red3
    df['green_red_cross2'] = np.where(df.green_red_mul2<=0,1,0)
    df['red_slope'] = df.red1 - df.red2
    df['red_slope2'] = df.red2 - df.red3
    df['green_slope'] = df.green1 - df.green2
    df['green_slope2'] = df.green2 - df.green3
    df['green_red_slope_change'] = df.green_slope - df.red_slope
    df['green_red_dist'] = (df.green1 + df.green2)/2 - (df.red1 + df.red2)/2
    df['mid_slope'] = df.mid1 - df.mid2

    df['haclose'] = prices[['close','high','low','open']].mean(axis = 1) 
    df['haopen'] = prices[['close','open']].mean(axis = 1) 
    df['haopen'] = (df.haopen.shift(1) + df.haclose.shift(1)) / 2
    df['hacolor'] = np.where(df.haclose>=df.haopen,1,-1)
    df['barnumber'] = df.groupby((df['hacolor'] != df['hacolor'].shift(1)).cumsum()).cumcount()+1
    df['barnumber2'] = df.barnumber.shift(1)
    df['haclose1'] = prices['open']
    df['haopen1'] = df.haopen
    df['habarsize1'] = df.haclose1-df.haopen1
    df['hacolor1'] = np.where(df.haclose1>=df.haopen1,1,-1)
    df['hacolor2'] = df.hacolor.shift(1)
    df['hacolor3'] = df.hacolor.shift(2)
    df['hacolor4'] = df.hacolor.shift(3)
    df['hacolor5'] = df.hacolor.shift(4)
    df['hacolor6'] = df.hacolor.shift(5)
    df['hacolor7'] = df.hacolor.shift(6)
    df['hacolor8'] = df.hacolor.shift(7)
    df['hacolor9'] = df.hacolor.shift(8)
    df['hacolor10'] = df.hacolor.shift(9)

    df['barnumber1'] = df.apply(lambda row : barnumber([row['hacolor1'],row['hacolor2'],row['hacolor3'],row['hacolor4'],row['hacolor5'],row['hacolor6'],row['hacolor7'],row['hacolor8'],row['hacolor9'],row['hacolor10']]), axis = 1)    

    df = df.drop(columns='hacolor2')
    df = df.drop(columns='hacolor3')
    df = df.drop(columns='hacolor4')
    df = df.drop(columns='hacolor5')
    df = df.drop(columns='hacolor6')
    df = df.drop(columns='hacolor7')
    df = df.drop(columns='hacolor8')
    df = df.drop(columns='hacolor9')
    df = df.drop(columns='hacolor10')

    
    df['haopen2'] = df.haopen.shift(1)
    df['haclose2'] = df.haclose.shift(1)
    df['habarsize2'] = df.haclose.shift(1)-df.haopen.shift(1)
    
    """
    df = df.drop(columns='red')
    df = df.drop(columns='green')
    df = df.drop(columns='haclose')
    df = df.drop(columns='haopen')
    df = df.drop(columns='hacolor')
    df = df.drop(columns='barnumber')
    df = df.drop(columns='haclose2')
    df = df.drop(columns='haopen2')

    df = df.drop(columns='haclose1')
    df = df.drop(columns='haopen1')

    df = df.drop(columns='rsi')
    """
    dict = {}
    dict[periods[0]] = df
    results.df = dict
    return results

def barnumber(a):
    first = a[0]
    cumsum = 1
    for i in range(1,len(a)):
        if (a[i]==first): 
            cumsum+=1
        else:
            break
    return cumsum


def priceaction(prices,periods):
    results = holder()
                
    df = prices[['open','high','low','close']]
    df['open_prev']  = prices['open'].shift(1)
    df['high_prev']  = prices['high'].shift(1)
    df['low_prev']   = prices['low'].shift(1)
    df['close_prev'] = prices['close'].shift(1)
    
    df['pa'] = 0
    df.loc[(df.high>=df.high_prev) & (df.low<=df.low_prev) & (df.close>df.open)
           ,'pa'] = 2
    df.loc[(df.high>=df.high_prev) & (df.low<=df.low_prev) & (df.close<=df.open)
           ,'pa'] = -2
    df.loc[(df[['open','close']].max(axis=1)>=df[['open_prev','close_prev']].max(axis=1)) 
           & (df[['open','close']].min(axis=1)<=df[['open_prev','close_prev']].min(axis=1))
           & (df.close>df.open)
           ,'pa'] = 3
    df.loc[(df[['open','close']].max(axis=1)>=df[['open_prev','close_prev']].max(axis=1)) 
           & (df[['open','close']].min(axis=1)<=df[['open_prev','close_prev']].min(axis=1))
           & (df.close<=df.open)
           ,'pa'] = -3
    df.loc[(df[['open','close']].max(axis=1)>=df[['open_prev','close_prev']].max(axis=1)) 
           & (df[['open','close']].min(axis=1)<=df[['open_prev','close_prev']].min(axis=1))
           & (df.high>=df.high_prev) & (df.low<=df.low_prev) & (df.close>df.open)
           ,'pa'] = 4
    df.loc[(df[['open','close']].max(axis=1)>=df[['open_prev','close_prev']].max(axis=1)) 
           & (df[['open','close']].min(axis=1)<=df[['open_prev','close_prev']].min(axis=1))
           & (df.high>=df.high_prev) & (df.low<=df.low_prev) & (df.close<=df.open)
           ,'pa'] = -4
    df['body'] = df.open-df.close
    df['body'] = df.body.abs()
    df['upwick'] = df.high-df[['open','close']].max(axis=1)
    df['downwick'] = df[['open','close']].min(axis=1) - df.low
    df.loc[(df.upwick>=2*df.body)&(df.downwick<=2*df.body),'pa'] = -1
    df.loc[(df.upwick<=2*df.body)&(df.downwick>=2*df.body),'pa'] = 1

    
    df = df.drop(columns='open')
    df = df.drop(columns='high')
    df = df.drop(columns='low')
    df = df.drop(columns='close')
    df = df.drop(columns='open_prev')
    df = df.drop(columns='high_prev')
    df = df.drop(columns='low_prev')
    df = df.drop(columns='close_prev')

    
    dict = {}
    dict[periods[0]] = df
    results.df = dict
    return results

def ma(prices,periods,m1= 5,m2= 10,m3= 20):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; ema1, ema2, signal period
    :return; macd
    '''
    m1 = str(m1)
    m2 = str(m2)
    m3 = str(m3)
    results = holder()
    dict = {}
    for i in range(0,len(periods)):
        madf = pd.DataFrame(index=prices.index)
        madf['SMA'] = prices.close.rolling(window=periods[i]).mean()
        madf['SMA'+m1] = prices.close.rolling(window=int(m1)).mean()
        madf['SMA'+m2] = prices.close.rolling(window=int(m2)).mean()
        madf['SMA'+m3] = prices.close.rolling(window=int(m3)).mean()
        madf['SMAdiff'] = madf.SMA.diff(1)
        madf['SMAdiff2'] = madf.SMA.diff(2)
        madf['SMAdiff2n'] = madf.SMA.shift(1) - madf.SMA.shift(2)
        madf['SMAdiff3'] = madf.SMA.diff(3)
        madf['SMAdiff4'] = madf.SMA.diff(4)
        madf['SMAdiffdiff'] = madf.SMAdiff.diff(1)
        madf['SMAvs'+m1] = madf['SMA'+m1]-madf.SMA
        madf['SMAvs'+m2] = madf['SMA'+m2]-madf.SMA
        madf['SMAvs'+m3] = madf['SMA'+m3]-madf.SMA
        madf['SMA_prev'] = madf.SMA.shift(1)
        madf['SMAdiff_prev'] = madf.SMAdiff.shift(1)
        madf['SMAdiff2_prev'] = madf.SMAdiff2.shift(1)
        madf['SMAdiff2n_prev'] = madf.SMAdiff2n.shift(1)
        madf['SMAdiff3_prev'] = madf.SMAdiff3.shift(1)
        madf['SMAdiff4_prev'] = madf.SMAdiff4.shift(1)
        madf['SMAdiffdiff_prev'] = madf.SMAdiffdiff.shift(1)
        madf['SMAvs'+m1+'_prev'] = madf['SMAvs'+m1].shift(1)
        madf['SMAvs'+m2+'_prev'] = madf['SMAvs'+m2].shift(1)
        madf['SMAvs'+m3+'_prev'] = madf['SMAvs'+m3].shift(1)
        
        madf = madf.drop(columns='SMA')
        madf = madf.drop(columns='SMA'+m1)
        madf = madf.drop(columns='SMA'+m2)
        madf = madf.drop(columns='SMA'+m3)
        madf = madf.drop(columns='SMAdiff')
        madf = madf.drop(columns='SMAdiff2')
        madf = madf.drop(columns='SMAdiff2n')
        madf = madf.drop(columns='SMAdiff3')
        madf = madf.drop(columns='SMAdiff4')
        madf = madf.drop(columns='SMAdiffdiff')
        madf = madf.drop(columns='SMAvs'+m1)
        madf = madf.drop(columns='SMAvs'+m2)
        madf = madf.drop(columns='SMAvs'+m3)
    #     madf['EMA'] = prices.close.ewm(span=periods[0]).mean()
    #     madf['EMAdiff'] = madf.EMA.diff(1)
    #     madf['EMAdiffdiff'] = madf.EMAdiff.diff(1)
    #     madf['EMAclose'] = prices.close - madf.EMA
        dict[periods[i]] = madf.copy()
    results.df = dict
    return results


def ma2(prices,periods,m1= 5,m2= 10,m3= 20):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; ema1, ema2, signal period
    :return; macd
    '''
    m1 = str(m1)
    m2 = str(m2)
    m3 = str(m3)
    results = holder()
    dict = {}
    for i in range(0,len(periods)):
        madf = pd.DataFrame(index=prices.index)
        madf['SMA'] = prices.close.rolling(window=periods[i]).mean()
        madf['SMA'+m1] = prices.close.rolling(window=int(m1)).mean()
        madf['SMA'+m2] = prices.close.rolling(window=int(m2)).mean()
        madf['SMA'+m3] = prices.close.rolling(window=int(m3)).mean()
        madf['SMAdiff1n'] = madf.SMA.diff(1)
        madf['SMAdiff2n'] = madf.SMA.shift(1) - madf.SMA.shift(2)
        madf['SMAdiff3n'] = madf.SMA.shift(2) - madf.SMA.shift(3)
        madf['SMAdiff4n'] = madf.SMA.shift(3) - madf.SMA.shift(4)
        madf['SMAdiff5n'] = madf.SMA.shift(4) - madf.SMA.shift(5)
        madf['SMAdiffdiff'] = madf.SMAdiff1n.diff(1)
        madf['SMAvs'+m1] = madf['SMA'+m1]-madf.SMA
        madf['SMAvs'+m2] = madf['SMA'+m2]-madf.SMA
        madf['SMAvs'+m3] = madf['SMA'+m3]-madf.SMA

        madf['SMA_prev'] = madf.SMA.shift(1)
        madf['SMAdiff1n_prev']   = madf.SMAdiff1n.shift(1)
        madf['SMAdiff2n_prev'] = madf.SMAdiff2n.shift(1)
        madf['SMAdiff3n_prev'] = madf.SMAdiff3n.shift(1)
        madf['SMAdiff4n_prev'] = madf.SMAdiff4n.shift(1)
        madf['SMAdiff5n_prev'] = madf.SMAdiff5n.shift(1)
        madf['SMAdiffdiff_prev'] = madf.SMAdiffdiff.shift(1)
        madf['SMAvs'+m1+'_prev'] = madf['SMAvs'+m1].shift(1)
        madf['SMAvs'+m2+'_prev'] = madf['SMAvs'+m2].shift(1)
        madf['SMAvs'+m3+'_prev'] = madf['SMAvs'+m3].shift(1)
        
        madf['SMAdiffseq_prev'] = 0
        madf.loc[madf.SMAdiff1n_prev>=0,'SMAdiffseq_prev'] = 1
        madf.loc[(madf.SMAdiff1n_prev>=0) & (madf.SMAdiff2n_prev>=0),'SMAdiffseq_prev'] = 2
        madf.loc[(madf.SMAdiff1n_prev>=0) & (madf.SMAdiff2n_prev>=0) & (madf.SMAdiff3n_prev>=0),'SMAdiffseq_prev'] = 3
        madf.loc[(madf.SMAdiff1n_prev>=0)&(madf.SMAdiff2n_prev>=0)&(madf.SMAdiff3n_prev>=0)&(madf.SMAdiff4n_prev>=0),'SMAdiffseq_prev'] = 4
        madf.loc[(madf.SMAdiff1n_prev>=0)&(madf.SMAdiff2n_prev>=0)&(madf.SMAdiff3n_prev>=0)&(madf.SMAdiff4n_prev>=0)& (madf.SMAdiff5n_prev>=0),'SMAdiffseq_prev'] = 5
        madf.loc[madf.SMAdiff1n_prev<0,'SMAdiffseq_prev'] = -1
        madf.loc[(madf.SMAdiff1n_prev<0) & (madf.SMAdiff2n_prev<0),'SMAdiffseq_prev'] = -2
        madf.loc[(madf.SMAdiff1n_prev<0) & (madf.SMAdiff2n_prev<0) & (madf.SMAdiff3n_prev<0),'SMAdiffseq_prev'] = -3
        madf.loc[(madf.SMAdiff1n_prev<0)&(madf.SMAdiff2n_prev<0)&(madf.SMAdiff3n_prev<0)&(madf.SMAdiff4n_prev<0),'SMAdiffseq_prev'] = -4
        madf.loc[(madf.SMAdiff1n_prev<0)&(madf.SMAdiff2n_prev<0)&(madf.SMAdiff3n_prev<0)&(madf.SMAdiff4n_prev<0)& (madf.SMAdiff5n_prev<0),'SMAdiffseq_prev'] = -5

        madf = madf.drop(columns='SMA')
        madf = madf.drop(columns='SMA'+m1)
        madf = madf.drop(columns='SMA'+m2)
        madf = madf.drop(columns='SMA'+m3)
        madf = madf.drop(columns='SMAdiff1n')
        madf = madf.drop(columns='SMAdiff2n')
        madf = madf.drop(columns='SMAdiff3n')
        madf = madf.drop(columns='SMAdiff4n')
        madf = madf.drop(columns='SMAdiff5n')
        madf = madf.drop(columns='SMAdiffdiff')
        madf = madf.drop(columns='SMAvs'+m1)
        madf = madf.drop(columns='SMAvs'+m2)
        madf = madf.drop(columns='SMAvs'+m3)
        madf = madf.drop(columns='SMAdiff1n_prev')
        madf = madf.drop(columns='SMAdiff2n_prev')
        madf = madf.drop(columns='SMAdiff3n_prev')
        madf = madf.drop(columns='SMAdiff4n_prev')
        madf = madf.drop(columns='SMAdiff5n_prev')
    #     madf['EMA'] = prices.close.ewm(span=periods[0]).mean()
    #     madf['EMAdiff'] = madf.EMA.diff(1)
    #     madf['EMAdiffdiff'] = madf.EMAdiff.diff(1)
    #     madf['EMAclose'] = prices.close - madf.EMA
        dict[periods[i]] = madf.copy()
    results.df = dict
    return results

def ma3(prices,periods,m1= 3,m2= 5,m3= 7):
    m1 = str(m1)
    m2 = str(m2)
    m3 = str(m3)
    results = holder()
    dict = {}
    for i in range(0,len(periods)):
        madf = pd.DataFrame(index=prices.index)
        madf['SMA'] = prices.close.rolling(window=periods[i]).mean()
        madf['SMA'+m1] = prices.close.rolling(window=int(m1)).mean()
        madf['SMA'+m2] = prices.close.rolling(window=int(m2)).mean()
        madf['SMA'+m3] = prices.close.rolling(window=int(m3)).mean()
        madf['SMAdiff1n'] = madf.SMA.diff(1)
        madf['SMAdiffdiff'] = madf.SMAdiff1n.diff(1)
        madf['SMAvs'+m1] = madf['SMA'+m1]-madf.SMA
        madf['SMAvs'+m2] = madf['SMA'+m2]-madf.SMA
        madf['SMAvs'+m3] = madf['SMA'+m3]-madf.SMA

        madf['SMAdiffseq'] = 0
        madf.loc[(madf.SMAdiff1n>=0) & (madf.SMAdiffdiff>=0),'SMAdiffseq'] = 1
        madf.loc[(madf.SMAdiff1n>=0) & (madf.SMAdiffdiff< 0),'SMAdiffseq'] = 2
        madf.loc[(madf.SMAdiff1n< 0) & (madf.SMAdiffdiff< 0),'SMAdiffseq'] = 3
        madf.loc[(madf.SMAdiff1n< 0) & (madf.SMAdiffdiff>=0),'SMAdiffseq'] = 4
#         madf.loc[(madf.SMAdiff1n>0) & (madf.SMAdiffdiff>0),'SMAdiffseq'] = 1
#         madf.loc[(madf.SMAdiff1n>0) & (madf.SMAdiffdiff<0),'SMAdiffseq'] = 2
#         madf.loc[(madf.SMAdiff1n<0) & (madf.SMAdiffdiff<0),'SMAdiffseq'] = 3
#         madf.loc[(madf.SMAdiff1n<0) & (madf.SMAdiffdiff>0),'SMAdiffseq'] = 4

        madf['SMA_prev'] = madf.SMA.shift(1)
        madf['SMAdiffseq_prev'] = madf['SMAdiffseq'].shift(1)
        madf['SMAvs'+m1+'_prev'] = madf['SMAvs'+m1].shift(1)
        madf['SMAvs'+m2+'_prev'] = madf['SMAvs'+m2].shift(1)
        madf['SMAvs'+m3+'_prev'] = madf['SMAvs'+m3].shift(1)

        madf = madf.drop(columns='SMA')
        madf = madf.drop(columns='SMA'+m1)
        madf = madf.drop(columns='SMA'+m2)
        madf = madf.drop(columns='SMA'+m3)
        madf = madf.drop(columns='SMAdiff1n')
        madf = madf.drop(columns='SMAdiffdiff')
        madf = madf.drop(columns='SMAdiffseq')
        madf = madf.drop(columns='SMAvs'+m1)
        madf = madf.drop(columns='SMAvs'+m2)
        madf = madf.drop(columns='SMAvs'+m3)

        dict[periods[i]] = madf.copy()
    results.df = dict
    return results

def atr(prices, periods):
    results = holder()
    dict = {}
    for i in range(0, len(periods)):
        resdf = pd.DataFrame(index=prices.index)
        resdf0 = pd.DataFrame(index=prices.index)
        resdf0['tr1'] = prices['high'] - prices['low']
        resdf0['tr2'] = abs (prices['high'] - prices['close'].shift())
        resdf0['tr3'] = abs (prices['low'] - prices['close'].shift())
        resdf['tr'] = resdf0.max(axis=1)
        resdf['atr'] = resdf.tr.rolling(periods[i]).mean()
        resdf['atr_prev'] = resdf.atr.shift(1)
        resdf = resdf.drop(columns='atr')
        resdf = resdf.drop(columns='tr')
        dict[periods[i]] = resdf.copy()
    
    results.df = dict
    return results

def srs(prices, periods):
    results = holder()
    dict = {}
    resdf = prices[['id','open','high','low','close']]

    resdf['close_prev'] = prices.close.shift(1)
    resdf['close_next'] = prices.close.shift(-1)
    resdf['high_next'] = prices.high.shift(-1)
    resdf['low_next'] = prices.low.shift(-1)
    resdf['close_max'] = prices.close.cummax()
    resdf['close_min'] = prices.close.cummin()
    
    resdf['horn'] = 0
    resdf['horn_v'] = 0
    resdf.loc[(resdf.close>=resdf.close_next) & (resdf.close>resdf.close_prev), 'horn'] = 1
#     resdf.loc[(resdf.close>=resdf.close_next) & (resdf.close>resdf.open), 'horn'] = 1
    resdf.loc[(resdf.close<=resdf.close_next) & (resdf.close<resdf.close_prev), 'horn'] = -1
#     resdf.loc[(resdf.close<=resdf.close_next) & (resdf.close<resdf.open), 'horn'] = -1
    resdf.loc[resdf.horn==1 ,'horn_v'] = resdf[['high','high_next']].max(axis=1)
    resdf.loc[resdf.horn==-1,'horn_v'] = resdf[['low' ,'low_next' ]].min(axis=1)
    
    resdf['horn_broke_id'] = -1
    resdf['sr_broke_id'] = -1
    resdf['sr_broke'] = 0    

# przełamanie hornow        
    nonbroken_horns0 = len(resdf[((resdf.horn==1)|(resdf.horn==-1))&(resdf.horn_broke_id==-1)])
    print('nonbroken_horns0:',nonbroken_horns0)
    i = -1
    resdf['nextbar_open'] = resdf.open.shift(i)
    while ((len(resdf[((resdf.horn==1)|(resdf.horn==-1)) & (resdf.horn_broke_id==-1) & (resdf.nextbar_open>0)])>0) & (i>-len(resdf))):
        resdf['nextbar_open'] = resdf.open.shift(i)
        resdf['nextbar_id_1'] = resdf.id.shift(i+1)
        resdf.loc[(resdf.horn==1) & (resdf.nextbar_open>resdf.horn_v) & (resdf.horn_broke_id==-1),'horn_broke_id'] = resdf.nextbar_id_1
        resdf.loc[(resdf.horn==-1) & (resdf.nextbar_open<resdf.horn_v) & (resdf.horn_broke_id==-1),'horn_broke_id'] = resdf.nextbar_id_1
        i-=1
    print('breakhorn_i:',i+1)
    nonbroken_horns1 = len(resdf[((resdf.horn==1)|(resdf.horn==-1))&(resdf.horn_broke_id==-1)])
    print('nonbroken_horns1:',nonbroken_horns1)

#     alltimeshigh/low
    resdf.loc[(resdf.horn==1)&(resdf.horn_v>resdf.close_max),'horn'] = 10
    resdf.loc[(resdf.horn==-1)&(resdf.horn_v<resdf.close_min),'horn'] = -10

    resdf = resdf.drop(columns='close_prev')
    resdf = resdf.drop(columns='close_next')
    resdf = resdf.drop(columns='high_next')
    resdf = resdf.drop(columns='low_next')
    resdf = resdf.drop(columns='close_max')
    resdf = resdf.drop(columns='close_min')
    resdf = resdf.drop(columns='close')
    resdf = resdf.drop(columns='high')
    resdf = resdf.drop(columns='low')
    resdf = resdf.drop(columns='horn_v')
    resdf = resdf.drop(columns='nextbar_open')
    resdf = resdf.drop(columns='nextbar_id_1')
    
    srs0 = 0
    srs1 = len(resdf[(resdf.horn==10)|(resdf.horn==-10)])
    print('srs:',srs1)
    maxiter = resdf[((resdf.horn==1)|(resdf.horn==-1))&(resdf.horn_broke_id!=-1)]
    maxiter = (maxiter.horn_broke_id-maxiter.id).max()
    while(srs1>srs0):
# propagate horn_broke to sr_broke        
        xxx = resdf[['id']].merge(resdf[(resdf.horn==10) | (resdf.horn==-10)][['id','horn','horn_broke_id']],
                                                 left_on=['id'],suffixes=('', '_y'), 
                                                 right_on = ['horn_broke_id'], how='left')
#         xxx.to_csv(sep=';',path_or_buf='../Data/xxx1.csv',date_format="%Y-%m-%d",index = False,na_rep='')
        statsgb = xxx.groupby(['id'])
        stats = statsgb.size().to_frame(name='xx')
        stats = stats.join(statsgb.agg({'id_y': 'min'}))
        stats = stats.join(statsgb.agg({'horn': 'min'}))
        xxx = stats.reset_index().set_index('id')
#         xxx.to_csv(sep=';',path_or_buf='../Data/xxx2.csv',date_format="%Y-%m-%d",index = False,na_rep='')
        resdf['sr_broke']    = xxx['horn'].fillna(resdf['sr_broke'])    
        resdf['sr_broke_id'] = xxx['id_y'].fillna(resdf['sr_broke_id'])    

#     z hd zrobic się ss gdy się znajdzie po hd jakiś broke=rr, a z hu->rr gdy broke=ss
        i = -1
        print('maxiter_0:',maxiter)
        while (i>=-maxiter):
            resdf['nextbar_sr_broke'] = resdf.sr_broke.shift(i)
            resdf['nextbar_sr_broke_id'] = resdf.sr_broke_id.shift(i)
            resdf['nextbar_id'] = resdf.id.shift(i)
            resdf.loc[(resdf.horn==1) & (resdf.nextbar_sr_broke==-10) &(resdf.nextbar_sr_broke_id<=resdf.id) &(resdf.nextbar_id<=resdf.horn_broke_id),'horn'] = 10
            resdf.loc[(resdf.horn==-1) & (resdf.nextbar_sr_broke==10) &(resdf.nextbar_sr_broke_id<=resdf.id) &(resdf.nextbar_id<=resdf.horn_broke_id),'horn'] = -10
            i-=1
            maxiter = resdf[((resdf.horn==1)|(resdf.horn==-1))&(resdf.horn_broke_id!=-1)]
            maxiter = (maxiter.horn_broke_id-maxiter.id).max()
            
        print('newsr_i:',i+1)
        srs0 = srs1
        srs1 = len(resdf[(resdf.horn==10)|(resdf.horn==-10)])
        print('srs:',srs1)

    
    resdf = resdf.drop(columns='id')
    resdf = resdf.drop(columns='nextbar_sr_broke')
    resdf = resdf.drop(columns='nextbar_sr_broke_id')
    resdf = resdf.drop(columns='nextbar_id')
    resdf = resdf.drop(columns='open')
    resdf = resdf.drop(columns='horn_broke_id')
    resdf = resdf.drop(columns='horn')
    resdf = resdf.drop(columns='sr_broke_id')
    
    
    resdf['sr_broke_prev'] = resdf.sr_broke.shift(1)
    i = 1
    while ((len(resdf[(resdf.sr_broke_prev==0)])>0) & (i<len(resdf))):
        resdf['prevbar_sr_broke_prev'] = resdf.sr_broke_prev.shift(i)
        resdf.loc[(resdf.sr_broke_prev==0)&(resdf.prevbar_sr_broke_prev!=0),'sr_broke_prev'] = resdf.prevbar_sr_broke_prev
        i+=1
    print('broke prop:',i-1)
    
    resdf = resdf.drop(columns='prevbar_sr_broke_prev')
    
    dict[periods[0]] = resdf
    results.df = dict
    return results

#Momentum
def momentum(prices,periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; list of periods for which to compute the function [3,5,10,...]
    :return; momentum indicator
    '''    

    results = holder()
    dict = {}
    for i in range(0, len(periods)):
        resdf = pd.DataFrame(index=prices.index)
        resdf['close'] = prices.close.diff(periods=periods[i])
        resdf['direction'] = -1
        resdf.loc[prices.close>prices.open,'direction'] = 1
        resdf['close_prev'] = resdf.close.shift(1)
        resdf['direction_prev'] = resdf.direction.shift(1)
        resdf = resdf.drop(columns='close')
        resdf = resdf.drop(columns='direction')
        
        dict[periods[i]] = resdf
    
    results.df = dict
    return results

def macd(prices,periods):
    '''
    :param prices; dataframe of OHLC currency data
    :param periods; ema1, ema2, signal period
    :return; macd
    '''
    results = holder()
    dict = {}
    
    EMA1 = prices.close.ewm(span=periods[0]).mean()
    EMA2 = prices.close.ewm(span=periods[1]).mean()
    macddf = pd.DataFrame(index=prices.index)
    macddf['MACD'] = EMA1-EMA2
    macddf['SigMACD'] = macddf.MACD.ewm(span=periods[2]).mean()
    macddf['HistMACD'] = macddf['MACD'] - macddf['SigMACD']
    macddf['HistMACD_prev'] = macddf.HistMACD.shift(1)
    macddf = macddf.drop(columns='MACD')
    macddf = macddf.drop(columns='SigMACD')
    macddf = macddf.drop(columns='HistMACD')
    dict[periods[0]] = macddf
    results.df = dict
    return results














#-----------------------------------------------------------------------------------------
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
    
    
    df = df.drop(columns='nextbar_hour')
    df = df.drop(columns='nextbar_open')
    df = df.drop(columns='nextbar_low')
    df = df.drop(columns='nextbar_high')
    df = df.drop(columns='nextbar_id')
    
    return df

def cleartrades(df,save=False):
    
    if (save==True):
        df.to_csv(sep=';',path_or_buf='../Data/trades.csv',date_format="%Y-%m-%d",index = False,na_rep='',float_format='%.5f')
    
    df = df[df.closeindex!=-1]
    df = df.drop(columns='date')
#     df = df.drop(columns='year')
#     df = df.drop(columns='month')
    df = df.drop(columns='day')
    df = df.drop(columns='weekday')
    df = df.drop(columns='open')
    df = df.drop(columns='low')
    df = df.drop(columns='high')
    df = df.drop(columns='close')
    df = df.drop(columns='volume')
    df = df.drop(columns='openprice')
    df = df.drop(columns='openindex')
    df = df.drop(columns='closeindex')
    df = df.drop(columns='closeprice')
    df = df.drop(columns='slindex')
    df = df.drop(columns='slprice')
    df = df.drop(columns='id')

#     df = df.drop(columns='profit')
    
#
    
#     df = df.drop(columns='tdi13green1_red1')
#     df = df.drop(columns='tdi13green2_red2')
#     df = df.drop(columns='tdi13green_red_change')
#     df = df.drop(columns='tdi13green_red_slope_change')
#     df = df.drop(columns='tdi13rsi1')
#     df = df.drop(columns='tdi13green1')
#     df = df.drop(columns='tdi13red1')
#     df = df.drop(columns='tdi13green2')
#     df = df.drop(columns='tdi13red2')
#     df = df.drop(columns='tdi13green_red_mul')

#     df = df.drop(columns='tdi13red_slope')
#     df = df.drop(columns='tdi13green_slope')

    df = df.drop(columns='tdi13haclose2')
    df = df.drop(columns='tdi13haopen2')

    df = df.drop(columns='tdi13haclose1')
    df = df.drop(columns='tdi13haopen1')

    df = df.drop(columns='tdi13rsi')
    df = df.drop(columns='tdi13red')
    df = df.drop(columns='tdi13green')
    df = df.drop(columns='tdi13mid')
    df = df.drop(columns='tdi13haclose')
    df = df.drop(columns='tdi13haopen')
    df = df.drop(columns='tdi13hacolor')
    df = df.drop(columns='tdi13barnumber')

    
#     df = df[(df.tdi13habarsize1>-20)&(df.tdi13habarsize1<20)]
#     df = df[(df.tdi13habarsize2>-20)&(df.tdi13habarsize2<20)]
#     df = df[(df.tdi13habarsize1<-10)|(df.tdi13habarsize1>10)]
#     df = df[(df.tdi13habarsize2<-10)|(df.tdi13habarsize2>10)]
#     df = df[(df.tradetype>0)]

#     df = df[(df.tdi13red_slope>-2.332) & (df.tdi13red_slope<=0.087)]
#     df = df[(df.tdi13green_slope<=3.437)]
#     df = df[(df.tdi13habarsize1>-2.519)&(df.tdi13habarsize1<=8.613)]
    
    
    
    return df


def preparetrades(masterFrame, trtypes,openhours,closehours,sls,yearfrom,yearto):
    first = True
    masterFrame = masterFrame[(masterFrame.year>=yearfrom)&(masterFrame.year<=yearto)]
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

    stats = stats.drop(columns=['level_0','level_1','level_2','level_3'])
    
    return stats

def stathyperparams(trades,params):
    starttime = datetime.now()

    trades = trades[trades.tradetype!=0]
    seq = {}
    stats = pd.DataFrame()
    seq['execs'] = 0
    seq['locs'] = 0
    seq['allexecs'] = 0
    alldays = len(trades.drop_duplicates(['year','month','day']))
    seq['mintrades'] = alldays/25 #once a month
    seq['dryrun'] = True
    stats = execstats_tradetype(trades,stats,params,seq)
    print('allexecs: ',seq['allexecs'])
#index=range(10000),
    stats = pd.DataFrame(columns=['tradetype','openhour','closehour','sl',
                                  'count','countup','countdown','updown_ratio',
                                  'monthsup','monthsdown',
                                  'profit_sum','profit_ratio',
                                  'maxdown','maxdown_ratio',
                                  'bar2from','bar2to','bar1from','bar1to',
                                  'gr2from','gr2to','gr1from','gr1to',
                                  'rslopefrom','rslopeto','gslopefrom','gslopeto',
                                  'grcfrom','grcto',
                                  'redfrom','redto',
                                  'barnofrom','barnoto',
                                  'crossfrom','crossto'
                                 ])
    seq['dryrun'] = False
    seq['starttime'] = datetime.now()
    seq['lastrun'] = datetime.now()
    
    stats = execstats_tradetype(trades,stats,params,seq)

    #scalanie
    statsgb = stats.groupby(['tradetype',
                             'openhour',
                             'closehour',
                             'sl',
                             'count',
                             'countup',
                             'countdown',
                             'monthsup',
                             'monthsdown',
                             'profit_sum',
                             'maxdown'
                            ])
    stats = statsgb.size().to_frame(name='xx')
    stats = stats.join(statsgb.agg({'bar2from': 'min','bar2to': 'max'}))
    stats = stats.join(statsgb.agg({'bar1from': 'min','bar1to': 'max'}))
    stats = stats.join(statsgb.agg({'gr2from': 'min','gr2to': 'max'}))
    stats = stats.join(statsgb.agg({'gr1from': 'min','gr1to': 'max'}))
    stats = stats.join(statsgb.agg({'rslopefrom': 'min','rslopeto': 'max'}))
    stats = stats.join(statsgb.agg({'gslopefrom': 'min','gslopeto': 'max'}))
    stats = stats.join(statsgb.agg({'grcfrom': 'min','grcto': 'max'}))
    stats = stats.join(statsgb.agg({'redfrom': 'min','redto': 'max'}))
    stats = stats.join(statsgb.agg({'barnofrom': 'min','barnoto': 'max'}))
    stats = stats.join(statsgb.agg({'crossfrom': 'min','crossto': 'max'}))
    stats = stats.reset_index()    
    
    
    stats['profit_ratio'] = stats.profit_sum/stats.sl
    stats['maxdown_ratio'] = stats.maxdown/stats.sl
    stats['updown_ratio'] = (stats.countup*1.0)-(stats.countdown)
    stats['mm_ratio'] = (stats.monthsup*1.0)-(stats.monthsdown)
    top = 500
    stats0 = stats.sort_values("count",ascending=False).head(top)
    stats0 = stats0.append(stats.sort_values("profit_sum",ascending=False).head(top))
    stats0 = stats0.append(stats.sort_values("profit_ratio",ascending=False).head(top))
    stats0 = stats0.append(stats.sort_values("maxdown_ratio",ascending=True).head(top))
    stats0 = stats0.append(stats.sort_values("updown_ratio",ascending=False).head(top))
    stats0 = stats0.append(stats.sort_values("monthsup",ascending=False).head(top))
    stats0 = stats0.append(stats.sort_values("monthsdown",ascending=True).head(top))
    stats0 = stats0.append(stats.sort_values("mm_ratio",ascending=False).head(top))
    stats0 = stats0.drop_duplicates()
    
    stats0 = stats0[['tradetype','openhour','closehour','sl','count','countup','countdown','updown_ratio',
                     'monthsup','monthsdown','mm_ratio','profit_sum','profit_ratio','maxdown','maxdown_ratio','xx',
                     'bar2from','bar2to','bar1from','bar1to','gr2from','gr2to','gr1from','gr1to',
                     'rslopefrom','rslopeto','gslopefrom','gslopeto','grcfrom','grcto',
                     'redfrom','redto','barnofrom','barnoto','crossfrom','crossto']]
    
    
    stats0.to_csv(sep=';',
                  path_or_buf='../Data/stats_'+str(params['filename'])+'.csv',
                  date_format="%Y-%m-%d",index = False,na_rep='',float_format='%.3f')
    endtime = datetime.now()
    print('finish:        ',str(datetime.now()))
    print('duration:      ',str(endtime - starttime))
    return stats0


def execstats_tradetype(trades,stats,params,seq):
    for tradetype in params['tradetypes']:
        seq['tradetype'] = tradetype
        stats = execstats_openclosehour(trades,stats,params,seq)
    return stats

def execstats_openclosehour(trades,stats,params,seq):
    for openhour in params['openhours']:
        for closehour in params['closehours']:                
            seq['openhour'] = openhour
            seq['closehour'] = closehour
            stats = execstats_sl(trades,stats,params,seq)
    return stats

def execstats_sl(trades,stats,params,seq):
    for sl in params['sls']:                
        seq['sl'] = sl
        stats = execstats_bar2(trades,stats,params,seq)
    return stats

def execstats_bar2(trades,stats,params,seq):
    for bar2from in params['bar2froms']:
        for bar2to in params['bar2tos']:
            if (bar2to>bar2from):
                seq['bar2from'] = bar2from
                seq['bar2to'] = bar2to
                stats = execstats_bar1(trades,stats,params,seq)
    return stats

def execstats_bar1(trades,stats,params,seq):
    for bar1from in params['bar1froms']:
        for bar1to in params['bar1tos']:
            if (bar1to>bar1from):
                seq['bar1from'] = bar1from
                seq['bar1to'] = bar1to
                stats = execstats_gr12(trades,stats,params,seq)
    return stats

def execstats_gr12(trades,stats,params,seq):
    for gr2from in params['gr2froms']:
        for gr2to in params['gr2tos']:
            if(gr2to>gr2from):
                for gr1from in params['gr1froms']:
                    for gr1to in params['gr1tos']:
                        if(gr1to>gr1from):
                            seq['gr2from'] = gr2from
                            seq['gr2to'] = gr2to
                            seq['gr1from'] = gr1from
                            seq['gr1to'] = gr1to
                            stats = execstats_grslopes(trades,stats,params,seq)
    return stats

def execstats_grslopes(trades,stats,params,seq):
    for gslopefrom in params['gslopefroms']:
        for gslopeto in params['gslopetos']:
            if(gslopeto>gslopefrom):
                for rslopefrom in params['rslopefroms']:
                    for rslopeto in params['rslopetos']:
                        if(rslopeto>rslopefrom):
                            seq['gslopefrom'] = gslopefrom
                            seq['gslopeto'] = gslopeto
                            seq['rslopefrom'] = rslopefrom
                            seq['rslopeto'] = rslopeto
                            stats = execstats_grc(trades,stats,params,seq)
    return stats

def execstats_grc(trades,stats,params,seq):
    for grcfrom in params['grcfroms']:
        for grcto in params['grctos']:
            if(grcto>grcfrom):
                seq['grcfrom'] = grcfrom
                seq['grcto'] = grcto
                stats = execstats_red(trades,stats,params,seq)
    return stats

def execstats_red(trades,stats,params,seq):
    for redfrom in params['redfroms']:
        for redto in params['redtos']:
            if(redto>redfrom):
                seq['redfrom'] = redfrom
                seq['redto'] = redto
                stats = execstats_barno(trades,stats,params,seq)
    return stats

def execstats_barno(trades,stats,params,seq):
    for barnofrom in params['barnofroms']:
        for barnoto in params['barnotos']:
            if(barnoto>barnofrom):
                seq['barnofrom'] = barnofrom
                seq['barnoto'] = barnoto
                stats = execstats_cross(trades,stats,params,seq)
    return stats

def execstats_cross(trades,stats,params,seq):
    for crossfrom in params['crossfroms']:
        for crossto in params['crosstos']:
            if(crossto>crossfrom):
                seq['crossfrom'] = crossfrom
                seq['crossto'] = crossto
                stats = execstats(trades,stats,params,seq)
    return stats


def execstats(trades,stats,params,seq):

    if (seq['dryrun']):
        seq['allexecs'] = seq['allexecs'] + 1
    else:
        seq['execs'] = seq['execs'] + 1
        df = calculatestats(trades,seq['mintrades'],seq['tradetype'],
                        seq['openhour'],seq['closehour'],
                        seq['sl'],
                        seq['bar2from'],seq['bar2to'],seq['bar1from'],seq['bar1to'],
                        seq['gr2from'],seq['gr2to'],seq['gr1from'],seq['gr1to'], 
                        seq['gslopefrom'],seq['gslopeto'],seq['rslopefrom'],seq['rslopeto'],
                        seq['grcfrom'],seq['grcto'],
                        seq['redfrom'],seq['redto'],
                        seq['barnofrom'],seq['barnoto'],
                        seq['crossfrom'],seq['crossto']                      
                       )
        if (not df is None):
            seq['locs'] = seq['locs'] + 1
#             stats.loc[seq['locs']] = df
            stats = stats.append(df, ignore_index=True)
            
        if ((seq['execs'] % 1000)==0):
            progress = (1.0*seq['execs']/seq['allexecs'])
            print('____progress:  ', "{:.4f}".format(progress*100.00))
            elapsedtime = datetime.now() - seq['starttime']
            print('elapsed:       ',str(elapsedtime))
            print('last run:      ',str(datetime.now() - seq['lastrun']))
            seq['lastrun'] = datetime.now()
            remainingtime = (elapsedtime.total_seconds()*(1-progress))/progress
            print('remaining:     ',timedelta(seconds=remainingtime))
            print('estimated end: ',str(datetime.now()+timedelta(seconds=remainingtime)))
        
    return stats
                               

def calculatestats(trades,mintrades,tradetype,openhour,closehour,sl,
                   bar2from,bar2to,bar1from,bar1to,
                   gr2from,gr2to,gr1from,gr1to,
                   gslopefrom,gslopeto,rslopefrom,rslopeto,
                   grcfrom,grcto,
                   redfrom,redto,
                   barnofrom,barnoto,
                   crossfrom,crossto
                  ):
    stats0 = trades[(trades.tradetype==tradetype)&
                    (trades.hour==openhour)&
                    (trades.closehour==closehour)&
                    (trades.sl==sl)&
                    (trades.tdi13habarsize2>=bar2from)&(trades.tdi13habarsize2<bar2to)&
                    (trades.tdi13habarsize1>=bar1from)&(trades.tdi13habarsize1<bar1to)&
                    (trades.tdi13green2_red2>=gr2from)&(trades.tdi13green2_red2<gr2to)&
                    (trades.tdi13green1_red1>=gr1from)&(trades.tdi13green1_red1<gr1to)&
                    (trades.tdi13green_slope>=gslopefrom)&(trades.tdi13green_slope<gslopeto)&
                    (trades.tdi13red_slope>=rslopefrom)&(trades.tdi13red_slope<rslopeto)&
                    (trades.tdi13green_red_change>=grcfrom)&(trades.tdi13green_red_change<grcto)&
                    (trades.tdi13red1>=redfrom)&(trades.tdi13red1<redto)&
                    (trades.tdi13barnumber1>=barnofrom)&(trades.tdi13barnumber1<barnoto)&
                    (trades.tdi13green_red_cross>=crossfrom)&(trades.tdi13green_red_cross<crossto)
                   ]
    pr_c = len(stats0)
    pr_sum = stats0.profit.sum()

    if ((pr_c>=mintrades) and (pr_sum>0)):
        pr_maxdown = (stats0.groupby((stats0['profit'] * stats0['profit'].shift(1) <=0).cumsum())['profit'].cumsum()).min()
        if (pr_maxdown>0):
            pr_maxdown = 0

        pr_c_u = len(stats0[stats0.profit>=0])
        pr_c_d = len(stats0[stats0.profit<0])            
        yearmonth = stats0.groupby(['year','month'])['profit'].sum().reset_index()
        monthsup = len(yearmonth[yearmonth.profit>0])
        monthsdown = len(yearmonth[yearmonth.profit<0])
        
        df = {'tradetype':tradetype,'openhour':openhour,'closehour':closehour,'sl':sl,
              'bar2from':bar2from,'bar2to':bar2to,'bar1from':bar1from,'bar1to':bar1to,
              'gr2from':gr2from,'gr2to':gr2to,'gr1from':gr1from,'gr1to':gr1to,
              'rslopefrom':rslopefrom,'rslopeto':rslopeto,
              'gslopefrom':gslopefrom,'gslopeto':gslopeto,
              'grcfrom':grcfrom,'grcto':grcto,
              'count':pr_c,'countup':pr_c_u,'countdown':pr_c_d,'profit_sum':pr_sum,
              'maxdown':pr_maxdown,'monthsup':monthsup,'monthsdown':monthsdown,
              'redfrom':redfrom,'redto':redto,
              'barnofrom':barnofrom,'barnoto':barnoto,
              'crossfrom':crossfrom,'crossto':crossto
             }
    else:
        df = None
    return df

def find_maxdownseries(grouping):
    (group_label, df) = grouping 
    maxdownseries = (df.groupby((df['profit'] * df['profit'].shift(1) <=0).cumsum())['profit'].cumsum()).min()
    return({group_label: maxdownseries})


def stathyperparams2(trades,params,conf):
    '''
    mode 0 - ranges combinations      - [0,[-1000,0,1000],[-1000,0,1000]] -> <-1000-0) <-1000-1000)  <0-1000)
    mode 1 - elem combinations        - [1,[1,2,3]]                       -> (1)  (2)  (3)  (1,2)  (1,3)  (2,3)  (1,2,3)
    mode 2 - elem one from list       - [2,[0.3,0.4,0.5,0.6,0.7,0.8,0.9]] -> (0.3)  (0.4)   (0.5)  etc
    mode 3 - ranges selected          - [3,[a,b,c],[d,e,f]]               -> <a,d)  <b,e)  <c,f)
    '''
    starttime = datetime.now()

    
    trades = trades[trades.tradetype!=0]
    tradecolumns = ['year','month','day','profit','slrisk']
    for key in params.keys():
        tradecolumns = np.append(tradecolumns,key)
    trades = trades[tradecolumns]    
    seq = {}
    stats = []
    fx={}
    seq['execs'] = 0
    seq['allexecs'] = 0
    alldays = len(trades.drop_duplicates(['year','month','day']))
    seq['mintrades'] = alldays/25 #once a month
    seq['dryrun'] = True
    stats = execstats2_r(trades,stats,params,seq,fx)
    print('allexecs: ',seq['allexecs'])

    statscolumns = ['ii','c','cu','cd','cc',
                                  'mu','md','mm',
                                  'p_sm','r',
                                  'maxp','maxd2','d','rd','rd2','avgsl','fx'
                    ]
    groupbycolumns = ['c',
                             'cu',
                             'cd',
                             'mu',
                             'md',
                             'p_sm',
                             'maxp','maxd2','avgsl'
                            ]

    for key in params.keys():
        statscolumns = np.append(statscolumns,key+'from')
        mode = params[key][0]
        if ( (mode == 0) or (mode == 3)):
            statscolumns = np.append(statscolumns,key+'to')
        else:
            groupbycolumns = np.append(groupbycolumns,key+'from')
    
    
    seq['dryrun'] = False
    seq['starttime'] = datetime.now()
    seq['lastrun'] = datetime.now()
    stats = []
    fx={}
    stats = execstats2_r(trades,stats,params,seq,fx)
    stats = pd.DataFrame(stats)

#     stats.to_csv(sep=';',path_or_buf='../Data/stats00.csv',date_format="%Y-%m-%d",index = False,na_rep='',float_format='%.3f')
    if (len(stats)==0):
        print('--no profitable strategy')
    else:
        #scalanie
#         statsgb = stats.groupby(by=list(groupbycolumns))
#         stats = statsgb.size().to_frame(name='xx')

#         for key in params.keys():
#             mode = params[key][0]
#             if ( (mode == 0) or (mode == 3)):
#                 stats = stats.join(statsgb.agg({key+'from': 'min',key+'to': 'max'}))
#         stats = stats.reset_index()    
        stats.drop_duplicates(subset=groupbycolumns,inplace=True, keep='last')

        stats['cc'] = stats.cu-stats.cd
        stats['mm'] = stats.mu-stats.md
        stats['r']  = stats.p_sm/stats.avgsl
        stats['d']  = stats.maxd2/stats.avgsl
        stats['rd'] = -1*stats.p_sm/stats.maxd2
        stats['rd2'] = -1*stats.maxp/stats.maxd2
        
        top = 6000
        stats0 = stats.sort_values("rd2",ascending=False).head(top)
#         stats0 = stats0.append(stats.sort_values("p_sm",ascending=False).head(top))
#         stats0 = stats0.append(stats.sort_values("c",ascending=False).head(top))
#         stats0 = stats0.append(stats.sort_values("maxd2",ascending=True).head(top))
#         stats0 = stats0.append(stats.sort_values("cc",ascending=False).head(top))
#         stats0 = stats0.append(stats.sort_values("mu",ascending=False).head(top))
#         stats0 = stats0.append(stats.sort_values("md",ascending=True).head(top))
#         stats0 = stats0.append(stats.sort_values("mm",ascending=False).head(top))
#         stats0 = stats0.append(stats.sort_values("r",ascending=False).head(top))
#         stats0 = stats0.append(stats.sort_values("d",ascending=True).head(top))
#         stats0 = stats0.append(stats.sort_values("rd",ascending=False).head(top))
#         stats0 = stats0.append(stats.sort_values("maxp",ascending=False).head(top))
#         stats0 = stats0.drop_duplicates()
        
        stats0 = stats0[statscolumns]
        
#         stats0 = stats[stats.maxp>3][statscolumns]

        stats0['fn']=conf['filename']
        stats0.to_csv(sep=';',
                      path_or_buf='../Data/stats_v2_'+str(conf['filename'])+'.csv',
                      date_format="%Y-%m-%d",index = False,na_rep='',float_format='%.3f')
    endtime = datetime.now()
    print('finish:        ',str(datetime.now()))
    print('duration:      ',str(endtime - starttime))
    return 

def execstats2_r(trades,stats,params,seq,fx={},cursor=0):
            
    if (cursor<len(params)):
        key = list(params.keys())[cursor]
        imode = params[key][0]
        if (imode == 0):
            for ifrom in params[key][1]:
                for ito in params[key][2]:
                    if (ito>ifrom):
                        fx[key] = [imode,ifrom,ito]
                        if (not seq['dryrun']):
                            cond = (trades[key].values>=fx[key][1])&(trades[key].values<fx[key][2])#
                            trades1 = trades[cond]#
                        else:
                            trades1 = trades
                        stats = execstats2_r(trades1,stats,params,seq,fx,cursor+1)
        elif (imode == 1):
            froms = params[key][1]
            a = []
            for L in range(1, len(froms)+1):
                for subset in itertools.combinations(froms, L):
                    a.append(subset)
            for ifrom in a:
                fx[key] = [imode,ifrom]
                if (not seq['dryrun']):
                    cond = trades[key].isin(fx[key][1])#
                    trades1 = trades[cond]#
                else:
                    trades1 = trades
                stats = execstats2_r(trades1,stats,params,seq,fx,cursor+1)
        elif (imode == 2):
            for ifrom in params[key][1]:
                fx[key] = [imode,ifrom]
                if (not seq['dryrun']):
                    cond = trades[key].values==fx[key][1]#
                    trades1 = trades[cond]#
                else:
                    trades1 = trades
                stats = execstats2_r(trades1,stats,params,seq,fx,cursor+1)
        elif (imode == 3):
            for ii in range(0, len(params[key][1])):
                ifrom = params[key][1][ii]
                ito   = params[key][2][ii]
                if (ito>ifrom):
                    fx[key] = [imode,ifrom,ito]
                    if (not seq['dryrun']):
                        cond = (trades[key].values>=fx[key][1])&(trades[key].values<fx[key][2])#
                        trades1 = trades[cond]#
                    else:
                        trades1 = trades
                    stats = execstats2_r(trades1,stats,params,seq,fx,cursor+1)
                
    else:
        stats = execstats2(trades,stats,params,seq,fx)
    return stats
    
def execstats2(trades,stats,params,seq,fx):
    
    if (seq['dryrun']):
        seq['allexecs'] = seq['allexecs'] + 1
    else:
        seq['execs'] = seq['execs'] + 1
        df = calculatestats2(trades,params,seq,fx)                  
        if (not df is None):
            timedump('10')
            stats.append(df)
            timedump('11')
            
        if ((datetime.now() - seq['lastrun']).total_seconds()>30):
            progress = (1.0*seq['execs']/seq['allexecs'])
            print('____progress:  ', "{:.2f}".format(progress*100.00),'%')
            elapsedtime = datetime.now() - seq['starttime']
            print('elapsed:       ',str(elapsedtime))
            print('last run:      ',str(datetime.now() - seq['lastrun']))
            seq['lastrun'] = datetime.now()
            remainingtime = (elapsedtime.total_seconds()*(1-progress))/progress
            print('remaining:     ',timedelta(seconds=remainingtime))
            print('estimated end: ',str(datetime.now()+timedelta(seconds=remainingtime)))
        
    return stats
# tutu
def calculatestats2(stats0,params,seq,fx):
    timedump('1')
    pr_c = len(stats0)
#     avgsl = stats0.sl_val.mean()
    avgsl = stats0.slrisk.mean()
    pr_sum = stats0.profit.sum()
    
    cumsum        = stats0.profit.cumsum()
    cumsumcummax  = cumsum.cummax()
    cumsum_cummax = cumsum-cumsumcummax
    pr_maxdown2   = cumsum_cummax.min()

    cumsumcummin  = cumsum.cummin()
    cumsum_cummin = cumsum-cumsumcummin
    pr_maxp       = cumsum_cummin.max()
    timedump('2')
        

    if ((pr_c>=seq['mintrades']) and (pr_maxp>-pr_maxdown2)):
        pr_c_u = len(stats0[stats0.profit>=0])
        pr_c_d = len(stats0[stats0.profit<0])            
#         yearmonth = stats0.groupby(['year','month'])['profit'].sum().reset_index()
#         monthsup = len(yearmonth[yearmonth.profit>0])
#         monthsdown = len(yearmonth[yearmonth.profit<0])
        monthsup = 0
        monthsdown = 0
        timedump('3')
        df = {'ii':seq['execs'],'c':pr_c,'cu':pr_c_u,'cd':pr_c_d,'p_sm':pr_sum,
              'maxp':pr_maxp,'maxd2':pr_maxdown2,'mu':monthsup,'md':monthsdown,'avgsl':avgsl,'fx':json.dumps(fx)
             }
        for kk in params.keys():
            imode = fx[kk][0]
            if ((imode == 0) or (imode == 3)):
                df[kk+'from'] = fx[kk][1]
                df[kk+'to'] = fx[kk][2]
            elif (imode == 1):
                ifrom = fx[kk][1]
                str1 = ','.join(str(e) for e in ifrom)
                df[kk+'from'] = str1
            elif (imode == 2):
                ifrom = fx[kk][1]
                df[kk+'from'] = str(ifrom)
        
    else:
        df = None
    timedump('4')    
    return df

def printfx(fx):
    for kk in fx.keys():
        imode = fx[kk][0]
        if ((imode == 0) or (imode == 3)):
            print(kk.rjust(20),str(fx[kk][1]).rjust(6),str(fx[kk][2]).rjust(6))
        else:
            print(kk.rjust(20),str(fx[kk][1]).rjust(6))

def calcandplot(trades,fxs):
    conditions = None   
    stats0 = pd.DataFrame()
    i=1
    for f in fxs:
        if ('fx' in f):
            break
        else:
            jj = fxs
            fxs = []
            for j in jj:
                j1 = {'ii':0,'fx':json.loads(j.replace('""', '\"'))}
                fxs.append(j1)                
            break
        
    for f in fxs:
        fx = f['fx']
        print('ii',f['ii'])
        printfx(fx)
        cond = prepareconditions(trades,fx)
        stats0 = calcfx(trades,cond)
        if (i<len(fxs)):
            print('--------------OR--------------')
        if conditions is None:
            conditions = cond
        else:
            conditions = conditions | cond            
        i+=1
    
    if (len(fxs)>1):
        print('--------------=--------------')
        stats0 = calcfx(trades,conditions)
    if (len(fxs)>0):
        plottrades(trades,stats0)
    
    return stats0

def prepareconditions(trades,fx):
    conditions = None            
    for kk in fx.keys():
        imode = fx[kk][0]
        if ((imode == 0) or (imode == 3)):
            cond = (trades[kk].values>=fx[kk][1])&(trades[kk].values<fx[kk][2])
        elif (imode == 1):
            cond = trades[kk].isin(fx[kk][1])
        elif (imode == 2):
            cond = trades[kk].values==fx[kk][1]
        if conditions is None:
            conditions = cond
        else:
            conditions = conditions & cond            

    return conditions

def plottrades(trades,stats0):
    ct = trades[['date','close']]
    ct = ct.drop_duplicates()

    ct = ct.merge(stats0[['date','profit']],left_on=['date'],right_on=['date'],how='left')
    ct.profit=ct.profit.fillna(0)
    ct['cumprofit']        = ct.profit.cumsum()
    
    xx = max(abs(ct.cumprofit.max()),abs(ct.cumprofit.min()))/ct.close.max()
    xx = (ct.cumprofit.max()-ct.cumprofit.min())/(ct.close.max()-ct.close.min())
    ct.close = ct.close*xx    
    offset = ct.cumprofit.min() - ct.close.min()
    ct.close = ct.close + offset    
    x = np.array(ct.date)
    y = np.array(ct.cumprofit)
    c = np.array(ct.close)
    plt.figure(figsize=(20,10))
    plt.plot(x,y)
    plt.plot(x,c)
    plt.show()
    return

def calcfx(trades,conditions):
    stats0 = trades[conditions] 
    stats0 = stats0.sort_values("date",ascending=True)
    seq = {}
    params = {}
    seq['mintrades'] = 0
    seq['execs'] = 0
    df = calculatestats2(stats0,{},seq,{})
    if (not df is None):
        df['r']=df['p_sm']/df['avgsl']
        df['d']=df['maxd2']/df['avgsl']
        df['rd2']=-1*df['maxp']/df['maxd2']
        print(df)
    else:
        print('{}')
        print('no data for filter')
        print('{}')
    return stats0


        
        
def statsall(trades):
    df = trades[trades.tradetype!=0]
    df = df.reset_index()
    
    allcolumns = df.columns
    staticcolumns = ['profit','profit1','hour','closehour','tradetype','index','year','month']
    
    dfcolumns = allcolumns
    for cc in staticcolumns:
        dfcolumns = np.delete(dfcolumns, np.where(dfcolumns == cc))    
    
#     dfcolumns = ['tdi13barnumber1','tdi13green_red_cross']
    
    ftcolumns = []
    for cc in dfcolumns:
        df[cc + '_f'] = df[cc]
        df[cc + '_t'] = df[cc]
#         ftcolumns.append(cc)
        ftcolumns.append(cc + '_f')
        ftcolumns.append(cc + '_t')
    
#     df = df[ftcolumns]
    
    df['id'] = df.index    

    '''
    low = .05
    high = .95
    quant_df = df[dfcolumns].quantile([low, high])
    for cc in dfcolumns:
        df[cc + '_f'] = np.where(df[cc + '_f']<quant_df.loc[low,cc],-1000,df[cc + '_f'])
        df[cc + '_t'] = np.where(df[cc + '_f']<quant_df.loc[low,cc],quant_df.loc[low,cc],df[cc + '_t'])
        df[cc + '_t'] = np.where(df[cc + '_t']>quant_df.loc[high,cc],1000,df[cc + '_t'])    
        df[cc + '_f'] = np.where(df[cc + '_t']>quant_df.loc[high,cc],quant_df.loc[high,cc],df[cc + '_f'])    
    '''
# test   
    '''
    df = pd.DataFrame({'A' : [2,4,3,8,5],'B' : [5,8,6,4,7]})
    dfcolumns = df.columns
    for cc in dfcolumns:
        df[cc + '_f'] = df[cc]
        df[cc + '_t'] = df[cc]
    dfcolumns1 = df.columns
    df['id'] = df.index        
    '''
# test        
        
    mul = 1
    for cc in dfcolumns:
        df[cc + '_f'] = np.floor(df[cc + '_f']/mul)*mul
        df[cc + '_t'] = np.where(np.ceil(df[cc + '_t']/mul)*mul==df[cc + '_t'],df[cc + '_t']+mul,np.ceil(df[cc + '_t']/mul)*mul)        
        
#     df.to_csv(sep=';',path_or_buf='../Data/trades_ranged_b_'+str(mul)+'.csv',date_format="%Y-%m-%d",index = False,na_rep='')
        
    onemore=1
    while onemore>0:
        starttime = datetime.now()
        lastrun   = datetime.now()
        onemore=0
        print('_x')
        for i in range(1,len(df)):
            df['ii'] = df.id.shift(-i)
            df['mm'] = 1
            df['dif'] = 0
            for cc in dfcolumns:
                df[cc + '_f_s'] = df[cc + '_f'].shift(-i)
                df[cc + '_t_s'] = df[cc + '_t'].shift(-i)
                df['mm'] = np.where(isoverlap(df[cc + '_f'],df[cc + '_t'],df[cc + '_f_s'],df[cc + '_t_s']),df.mm,0)
                df['dif'] = np.where((df[cc + '_f']!=df[cc + '_f_s'])|(df[cc + '_t']!=df[cc + '_t_s']),1,df.dif)
#             display(df)

            df['m'] = df.mm * df.dif
#             if (df.m.sum()>50):
#                 df.to_csv(sep=';',path_or_buf='../Data/trades_ranged_b_'+str(mul)+'_'+str(i)+'.csv',date_format="%Y-%m-%d",index = False,na_rep='')
            if (df.m.sum()>0):
#                 print(df.m.sum())
                for cc in dfcolumns:
                    df.loc[df.m==1,cc + '_f'] = df[[cc + '_f',cc + '_f_s']].min(axis=1)
                    df.loc[df.m==1,cc + '_t'] = df[[cc + '_t',cc + '_t_s']].max(axis=1)
                    df[cc + '_f'] = df[['id',cc + '_f',cc + '_t']].merge(df[df.m==1][['ii',cc + '_f',cc + '_t']],
                                             left_on=['id'],suffixes=('', '_y'), 
                                             right_on = ['ii'], how='left')[cc + '_f_y'].fillna(df[cc + '_f'])
                    df[cc + '_t'] = df[['id',cc + '_f',cc + '_t']].merge(df[df.m==1][['ii',cc + '_f',cc + '_t']],
                                             left_on=['id'],suffixes=('', '_y'), 
                                             right_on = ['ii'], how='left')[cc + '_t_y'].fillna(df[cc + '_t'])
#             display(df)

            if ((i% np.ceil(len(df)/10))==0):
                progress = (1.0*i/len(df))
                print('____progress:  ', "{:.1f}".format(progress*100.00))
                print('onemore:       ',onemore)
                elapsedtime = datetime.now() - starttime
                print('elapsed:       ',str(elapsedtime))
                print('last run:      ',str(datetime.now() - lastrun))
                lastrun   = datetime.now()
                remainingtime = (elapsedtime.total_seconds()*(1-progress))/progress
                print('remaining:     ',timedelta(seconds=remainingtime))
                print('estimated end: ',str(datetime.now()+timedelta(seconds=remainingtime)))
                
            if onemore==0:
                onemore = df.m.sum()
    
    df = df[np.concatenate((staticcolumns, ftcolumns))]
    df.to_csv(sep=';',path_or_buf='../Data/trades_ranged_'+str(mul)+'.csv',date_format="%Y-%m-%d",index = False,na_rep='')
    
    groupcolumns = ['hour','closehour','tradetype','year','month']
    groupcolumns = np.append(groupcolumns, ftcolumns)
    statsgb = df.groupby(by=list(groupcolumns))
    stats = statsgb.size().to_frame(name='xx')
    stats = stats.join(statsgb.agg({'profit': 'sum'}))
    stats = stats.reset_index()   
    stats['monthsup']   = np.where(stats.profit>0,1,0)
    stats['monthsdown'] = np.where(stats.profit<0,1,0)

    groupcolumns = ['hour','closehour','tradetype']
    groupcolumns = np.append(groupcolumns, ftcolumns)
    statsgb = stats.groupby(by=list(groupcolumns))
    stats = statsgb.size().to_frame(name='yy')
    stats = stats.join(statsgb.agg({'xx': 'sum'}).rename(columns={'xx': 'count'}))
    stats = stats.join(statsgb.agg({'profit': 'sum'}).rename(columns={'profit': 'profit'}))
    stats = stats.join(statsgb.agg({'monthsup': 'sum'}))
    stats = stats.join(statsgb.agg({'monthsdown': 'sum'}))
    stats = stats.reset_index()   
    
    stats.to_csv(sep=';',path_or_buf='../Data/stats_trades_ranged_'+str(mul)+'.csv',date_format="%Y-%m-%d",index = False,na_rep='')
    
    
    return df,stats

def isoverlap(A,B,AA,BB):
    slip = 0
    return (
            (((AA>=A)&(AA<=B+slip))|((BB>=A-slip)&(BB<=B)))
            |
            (((A>=AA)&(A<=BB+slip))|((B>=AA-slip)&(B<=BB)))
    )


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




def runtrades_v0_4h_0(alltrades):
    tradetypes = [1]
    openhours = [5]
    closehours = [13]
    sls = [10]
    bar2froms = [-1000,-20,-10,0,10,20,1000]
    bar2tos = [-1000,-20,-10,0,10,20,1000]
    bar1froms = [-1000,-20,-10,0,10,20,1000]
    bar1tos =[-1000,-20,-10,0,10,20,1000]
    bar2froms = [-1000,1000]
    bar2tos = [-1000,1000]
    bar1froms = [-1000,1000]
    bar1tos =[-1000,1000]
    gr2froms = [-1000,-8,-3,0,3,8,1000]
    gr2tos = [-1000,-8,-3,0,3,8,1000]
    gr1froms = [-1000,-8,-3,0,3,8,1000]
    gr1tos = [-1000,-8,-3,0,3,8,1000]
    gr2froms = [-1000,-10,-8,-6,-4,-2,0,2,4,6,8,10,1000]
    gr2tos = [-1000,-10,-8,-6,-4,-2,0,2,4,6,8,10,1000]
    gr1froms = [-1000,-10,-8,-6,-4,-2,0,2,4,6,8,10,1000]
    gr1tos = [-1000,-10,-8,-6,-4,-2,0,2,4,6,8,10,1000]
    starttime = datetime.now()
    stats = stathyper(alltrades,tradetypes,openhours,closehours,sls,bar2froms,bar2tos,bar1froms,bar1tos,gr2froms,gr2tos,gr1froms,gr1tos)
    endtime = datetime.now()
    print(str(endtime - starttime))
    return stats


def runtrades_v1_4h_0(alltrades):
    params = {}

    params['tradetypes'] = [1]
    params['openhours'] = [5]
    params['closehours'] = [13]
    params['sls'] = [10]

    params['bar2froms'] = [-1000]
    params['bar2tos'] = [1000]

    params['bar1froms'] = [-1000]
    params['bar1tos'] =[1000]

    params['gr2froms'] = [-1000]
    params['gr2tos'] = [1000]

    params['gr1froms'] = [-1000]
    params['gr1tos'] = [1000]

    params['rslopefroms'] = [-1000,-3,0,3,1000]
    params['rslopetos'] = [-1000,-3,0,3,1000]

    params['gslopefroms'] = [-1000,-8,-3,0,3,8,1000]
    params['gslopetos'] = [-1000,-8,-3,0,3,8,1000]

    params['grcfroms'] = [-1000]
    params['grctos'] = [1000]

    params['redfroms'] = [0,100]
    params['redtos'] = [0,100]

    params['barnofroms'] = [1,100]
    params['barnotos'] = [1,100]

    params['crossfroms'] = [0,1]
    params['crosstos'] = [0,1]

    params['filename'] = '2015_2021_5_13_10'

    stats = stathyperparams(alltrades,params)
    return stats

def runtrades_v1_4h_1(alltrades):
    params = {}

    params['tradetypes'] = [1]
    params['openhours'] = [5]
    params['closehours'] = [13]
    params['sls'] = [30]

    params['bar2froms'] = [-1000,-8,0,8,1000]
    params['bar2tos'] = [-1000,-8,0,8,1000]
    params['bar2froms'] = [-1000]
    params['bar2tos'] = [1000]

    params['bar1froms'] = [-1000,-8,0,8,1000]
    params['bar1tos'] =[-1000,-8,0,8,1000]
    # params['bar1froms'] = [-1000]
    # params['bar1tos'] =[1000]

    params['gr2froms'] = [-1000]
    params['gr2tos'] = [1000]

    params['gr1froms'] = [-1000,0,1000]
    params['gr1tos'] = [-1000,0,1000]
    # params['gr1froms'] = [-1000]
    # params['gr1tos'] = [1000]

    params['rslopefroms'] = [-1000,-3,0,3,1000]
    params['rslopetos'] = [-1000,-3,0,3,1000]

    params['gslopefroms'] = [-1000,-8,-3,0,3,8,1000]
    params['gslopetos'] = [-1000,-8,-3,0,3,8,1000]

    params['grcfroms'] = [-1000]
    params['grctos'] = [1000]

    params['redfroms'] = [0,40,60,100]
    params['redtos'] = [0,40,60,100]
    # params['redfroms'] = [0,100]
    # params['redtos'] = [0,100]

    params['barnofroms'] = [1]
    params['barnotos'] = [2,3,4,100]
    # params['barnofroms'] = [1,100]
    # params['barnotos'] = [1,100]

    params['crossfroms'] = [0,1]
    params['crosstos'] = [1,2]

    params['filename'] = '2015_2021_5_13_30'

    stats = stathyperparams(alltrades,params)
    return stats


def runtrades_v2_4h_0(alltrades):
    conf   = {}
    params = {}

    params['tradetype'] = [[1],[2]]
    params['hour'] = [[5],[6]]
    params['closehour'] = [[13],[14]]
    params['sl'] = [[10],[11]]

    params['tdi13habarsize2'] = [[-1000],[1000]]

    params['tdi13habarsize1'] = [[-1000],[1000]]

    params['tdi13green2_red2'] = [[-1000],[1000]]

    params['tdi13green1_red1'] = [[-1000],[1000]]

    params['tdi13red_slope'] = [[-1000,-3,0,3,1000],[-1000,-3,0,3,1000]]

    params['tdi13green_slope'] = [[-1000,-8,-3,0,3,8,1000],[-1000,-8,-3,0,3,8,1000]]

    params['tdi13green_red_change'] = [[-1000],[1000]]

    params['tdi13red1'] = [[0,100],[0,100]]

    params['tdi13barnumber1'] = [[1,100],[2,100]]

    params['tdi13green_red_cross'] = [[0,1],[1,2]]

    conf['filename'] = '2015_2021_5_13_10'

    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runtrades_v2_4h_1(alltrades):
    conf   = {}
    params = {}

    params['tradetype'] = [[1],[1.1]]
    params['hour'] = [[5],[5.1]]
    params['closehour'] = [[13],[13.1]]
    params['sl'] = [[30],[30.1]]

    params['tdi13habarsize2'] = [[-1000],[1000]]

    params['tdi13habarsize1'] = [[-1000,-8,0,8,1000],[-1000,-8,0,8,1000]]

    params['tdi13green2_red2'] = [[-1000],[1000]]

    params['tdi13green1_red1'] = [[-1000,0,1000],[-1000,0,1000]]

    params['tdi13red_slope'] = [[-1000,-3,0,3,1000],[-1000,-3,0,3,1000]]

    params['tdi13green_slope'] = [[-1000,-8,-3,0,3,8,1000],[-1000,-8,-3,0,3,8,1000]]

    params['tdi13green_red_change'] = [[-1000],[1000]]

    params['tdi13red1'] = [[0,40,60,100],[0,40,60,100]]

    params['tdi13barnumber1'] = [[1],[2,3,4,100]]

    params['tdi13green_red_cross'] = [[0,1],[1,2]]

    conf['filename'] = '2015_2021_5_13_30'

    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runtrades_v2_4h_2(alltrades,tt=1,oh=5,ch=13,sl=10):
    conf   = {}
    params = {}

#     conf['tradetype'] = tt
#     conf['openhour'] = oh
#     conf['closehour'] = ch
#     conf['sl'] = sl
    params['tradetype'] = [[tt],[tt+0.1]]
    params['hour'] = [[oh],[oh+0.1]]
    params['closehour'] = [[ch],[ch+0.1]]
    params['sl'] = [[sl],[sl+0.1]]

#     params['tdi13habarsize2'] = [[-1000,-8,0,8,1000],[-1000,-8,0,8,1000]]

#     params['tdi13habarsize1'] = [[-1000],[1000]]

#     params['tdi13green2_red2'] = [[-1000,0,1000],[-1000,0,1000]]

#     params['tdi13green1_red1'] = [[-1000],[1000]]

    params['tdi13red_slope2'] = [[-1000,-5,-3,-2,-1,0,1,2,3,5,1000],[-1000,-5,-3,-2,-1,0,1,2,3,5,1000]]

    params['tdi13green_slope2'] = [[-1000,-8,-5,-3,-2,-1,0,1,2,3,5,8,1000],[-1000,-8,-5,-3,-2,-1,0,1,2,3,5,8,1000]]

    params['tdi13green_red_change2'] = [[-1000,0,1000],[-1000,0,1000]]

    params['tdi13red2'] = [[0,30,40,50,60,100],[0,30,40,50,60,100]]

    params['tdi13barnumber2'] = [[1,100],[2,3,4,100]]

#     params['tdi13green_red_cross2'] = [[0,1],[1,2]]

    conf['filename'] = '2015_2021_'+str(tt)+str(oh)+str(ch)+str(sl)
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats



def preparetrades_pa(masterFrame, trtypes, sls, yearfrom, yearto, slips):
    first = True
    masterFrame = masterFrame[(masterFrame.year>=yearfrom)&(masterFrame.year<=yearto)]
    for trtype in trtypes:
        for sl in sls:                
            for slip in slips:                
                df = masterFrame.copy()
                df = opentrades_pa(trtype,df,slip)
                df = closetrades_pa(df,sl)
                if (first==True): 
                    trades=df 
                    first = False
                else:
                    trades=trades.append(df)
    return trades

def opentrades_pa(mode,df,slip):
    df['tradetype'] = 0
    df['slip'] = slip
    
    df['pa_prev'] = df.pa1pa.shift(1)
    df['close_prev'] = df.close.shift(1)
    if (mode==0):
        df.loc[(df.pa_prev>0) & (df.high>=df.close_prev+slip),'tradetype'] = 1
        df.loc[(df.pa_prev<0) & (df.low<=df.close_prev-slip),'tradetype'] = -1
    elif (mode==1): 
        df.loc[(df.pa_prev>0) & (df.high>=df.close_prev+slip),'tradetype'] = mode
    elif (mode==-1): 
        df.loc[(df.pa_prev<0) & (df.low<=df.close_prev-slip),'tradetype'] = mode
    df.loc[df.tradetype==1,'openprice'] = df.close_prev+slip
    df.loc[df.tradetype==-1,'openprice'] = df.close_prev-slip
    df.loc[df.tradetype!=0,'openindex'] = df.id
    return df

def closetrades_pa(df,stoploss):
    df['sl'] = stoploss
    df['closeindex'] = -1
    df['closeprice'] = -1
#     df['closehour'] = 0
    df['slindex'] = -1
    df['slprice'] = -1
    df['profit'] = 0
    
    lastcloseU = df[df.pa1pa>0].tail(1).id.values[0] - 1
    lastcloseD = df[df.pa1pa<0].tail(1).id.values[0] - 1
    lastclose = min(lastcloseU,lastcloseD)
    i = 0
    while ((len(df[(df.tradetype!=0) & (df.closeindex==-1) & (df.id<=lastclose)])>0) & (i>=-20)):
        df['nextbar_pa'] = df.pa1pa.shift(i)
        df['nextbar_open'] = df.open.shift(i)
        df['nextbar_close'] = df.close.shift(i)
        df['nextbar_low'] = df.low.shift(i)
        df['nextbar_high'] = df.high.shift(i)
        df['nextbar_id'] = df.id.shift(i)
        #SL buy
        df.loc[(df.tradetype==1) & (df.closeindex==-1)&(df.nextbar_low-df.openprice<=-stoploss),'slprice'] = df.nextbar_low
        df.loc[(df.tradetype==1) & (df.closeindex==-1)&(df.nextbar_low-df.openprice<=-stoploss),'slindex'] = df.nextbar_id
        df.loc[(df.tradetype==1) & (df.closeindex==-1)&(df.nextbar_low-df.openprice<=-stoploss),'closeprice'] = df.openprice-stoploss
#         df.loc[(df.tradetype==1) & (df.closeindex==-1)&(df.nextbar_low-df.openprice<=-stoploss),'profit'] = -stoploss
        df.loc[(df.tradetype==1) & (df.closeindex==-1)&(df.nextbar_low-df.openprice<=-stoploss),'closeindex'] = df.nextbar_id

        #SL sell
        df.loc[(df.tradetype==-1) & (df.closeindex==-1)&(df.nextbar_high-df.openprice>=stoploss),'slprice'] = df.nextbar_high
        df.loc[(df.tradetype==-1) & (df.closeindex==-1)&(df.nextbar_high-df.openprice>=stoploss),'slindex'] = df.nextbar_id
        df.loc[(df.tradetype==-1) & (df.closeindex==-1)&(df.nextbar_high-df.openprice>=stoploss),'closeprice'] = df.openprice+stoploss
#         df.loc[(df.tradetype==-1) & (df.closeindex==-1)&(df.nextbar_high-df.openprice>=stoploss),'profit'] = -stoploss
        df.loc[(df.tradetype==-1) & (df.closeindex==-1)&(df.nextbar_high-df.openprice>=stoploss),'closeindex'] = df.nextbar_id
        
        #close buy
        df.loc[(df.tradetype>0) & (df.nextbar_pa < 0) & (df.closeindex==-1),'closeprice'] = df.nextbar_close
        df.loc[(df.tradetype>0) & (df.nextbar_pa < 0) & (df.closeindex==-1),'closeindex'] = df.nextbar_id
        #close sell
        df.loc[(df.tradetype<0) & (df.nextbar_pa > 0) & (df.closeindex==-1),'closeprice'] = df.nextbar_close
        df.loc[(df.tradetype<0) & (df.nextbar_pa > 0) & (df.closeindex==-1),'closeindex'] = df.nextbar_id

#         df.loc[(df.tradetype==1) & (df.nextbar_hour == hour) & (df.closeindex==-1),'profit'] = df.nextbar_open-df.openprice
#         df.loc[(df.tradetype==-1) & (df.nextbar_hour == hour) & (df.closeindex==-1),'profit'] = -(df.nextbar_open-df.openprice)
        i-=1
    
    print(i)
    df.loc[(df.tradetype==1) & (df.closeindex!=-1),'profit'] = df.closeprice - df.openprice
    df.loc[(df.tradetype==-1) & (df.closeindex!=-1),'profit'] = df.openprice - df.closeprice
#     df.loc[(df.closeindex!=-1),'closehour'] = hour

    df['profit'] = df.profit * 10000
    df['profit1'] = np.where(df.profit>=20,1,-1)

    df['sl'] = df.sl * 10000
    
    df = df.drop(columns='nextbar_pa')
    df = df.drop(columns='nextbar_close')
    df = df.drop(columns='nextbar_open')
    df = df.drop(columns='nextbar_low')
    df = df.drop(columns='nextbar_high')
    df = df.drop(columns='nextbar_id')
    
    return df

def cleartrades_pa(df,save=False):
    
    if (save==True):
        df.to_csv(sep=';',path_or_buf='../Data/trades.csv',date_format="%Y-%m-%d",index = False,na_rep='',float_format='%.5f')
    
    df = df[df.closeindex!=-1]
    df = df.drop(columns='date')
#     df = df.drop(columns='year')
#     df = df.drop(columns='month')
    df = df.drop(columns='day')
    df = df.drop(columns='weekday')
    df = df.drop(columns='open')
    df = df.drop(columns='low')
    df = df.drop(columns='high')
    df = df.drop(columns='close')
    df = df.drop(columns='volume')
    df = df.drop(columns='openprice')
    df = df.drop(columns='openindex')
    df = df.drop(columns='closeindex')
    df = df.drop(columns='closeprice')
    df = df.drop(columns='slindex')
    df = df.drop(columns='slprice')
    df = df.drop(columns='id')
    return df

def stattrades_pa(trades):
#     stats = trades.drop_duplicates(subset = ['hour','closehour','sl'])[['hour','closehour','sl']]
    trades = trades[trades.tradetype!=0]
    trades['profitup'] = 0
    trades['profitdown'] = 0
    trades.loc[trades.profit>=0,'profitup'] = 1
    trades.loc[trades.profit<0,'profitdown'] = 1
    
    statsgb = trades.groupby(['tradetype','weekday','pa_prev','sl','slip'])
    stats = statsgb.size().to_frame(name='counts')
    stats = stats.join(statsgb.agg({'profit': 'mean'}).rename(columns={'profit': 'profit_mean'}))
    stats = stats.join(statsgb.agg({'profit': 'sum'}).rename(columns={'profit': 'profit_sum'}))
    stats = stats.join(statsgb.agg({'profitup': 'sum'}).rename(columns={'profitup': 'countup'}))
    stats = stats.join(statsgb.agg({'profitdown': 'sum'}).rename(columns={'profitdown': 'countdown'}))
    stats = stats.reset_index()
    
    stats['profit_ratio'] = stats.profit_sum/stats.sl
    
    grouping = statsgb
    maxdowns = [find_maxdownseries(i).values() for i in grouping]
    maxdowns_df =  pd.DataFrame(data = maxdowns, index = grouping.groups, columns = ['maxdown']).reset_index()    
    
    stats = stats.merge(maxdowns_df,left_on=['tradetype','weekday','pa_prev','sl','slip'], right_on = ['level_0','level_1','level_2','level_3','level_4'], how='left')
    stats['maxdown_ratio'] = stats.maxdown/stats.sl

    stats = stats.drop(columns=['level_0','level_1','level_2','level_3','level_4'])
    
    return stats

def runstats_pa_v1(alltrades):
    conf   = {}
    params = {}

    params['tradetype'] = [0,[1],[1.1]]
    params['weekday'] = [0,[0,1,2,3,4],[0,1,2,3,4,5]]
    params['sl'] = [0,[10,20,30,40,50,60],[10,20,30,40,50,60,70]]
    params['slip'] = [0,[0,0.0005,0.001,0.0015,0.002],[0,0.0005,0.001,0.0015,0.002,0.003]]
    params['pa_prev'] =[0,[1,2,3,4],[1,2,3,4,5]]
    conf['filename'] = 'pa_2015_2021_'
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats


def runstats_pa_v3(alltrades):
    conf   = {}
    params = {}

    params['tradetype'] = [1,[1]]
    params['weekday'] = [1,[0,1,2,3,4]]
    params['sl'] = [2,[10,20,30,40,50,60]]
    params['slip'] = [2,[0,0.0005,0.001,0.0015,0.002]]
    params['pa_prev'] =[1,[1,2,3,4]]
    conf['filename'] = 'pa_2015_2021_3_'
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats



def preparetrades_brut_tsl0(masterFrame, trtypes, sls, tps, tsls, yearfrom, yearto,atr=''):
    if (tsls==[]):
        return preparetrades_brut_tp(masterFrame, trtypes, sls, tps, yearfrom, yearto,atr)
    first = True
    masterFrame = masterFrame[(masterFrame.year>=yearfrom)&(masterFrame.year<=yearto)]
    for trtype in trtypes:
        for sl in sls:                
            for tp in tps:
                if (tp==100):
                    df = masterFrame.copy()
                    df = opentrades_brut(trtype,df)
                    df = closetrades_tsl(df,sl,tp,0,atr)
                    if (first==True): 
                        trades=df
                        first = False
                    else:
                        trades=trades.append(df)
                else:
                    for tsl in tsls:                
                        df = masterFrame.copy()
                        df = opentrades_brut(trtype,df)
                        df = closetrades_tsl(df,sl,tp,tsl,atr)
                        if (first==True): 
                            trades=df
                            first = False
                        else:
                            trades=trades.append(df)

    return trades

def preparetrades_brut_tsl(masterFrame, trtypes, sls, tps, tsls, yearfrom, yearto,atr=''):
    if (tsls==[]):
        return preparetrades_brut_tp(masterFrame, trtypes, sls, tps, yearfrom, yearto,atr)
    masterFrame = masterFrame[(masterFrame.year>=yearfrom)&(masterFrame.year<=yearto)]
    trades = []
    for trtype in trtypes:
        for sl in sls:                
            for tp in tps:
                if (tp==100):
                    df = masterFrame.copy()
                    df = opentrades_brut(trtype,df)
                    df = closetrades_tsl(df,sl,tp,0,atr)
                    trades.append(df)
                else:
                    for tsl in tsls:                
                        df = masterFrame.copy()
                        df = opentrades_brut(trtype,df)
                        df = closetrades_tsl(df,sl,tp,tsl,atr)
                        trades.append(df)

    trades1 = pd.concat(trades, ignore_index=True)
    return trades1


def preparetrades_brut_tp(masterFrame, trtypes, sls, tps, yearfrom, yearto,atr=''):
    first = True
    masterFrame = masterFrame[(masterFrame.year>=yearfrom)&(masterFrame.year<=yearto)]
    for trtype in trtypes:
        for sl in sls:                
            for tp in tps:                
                df = masterFrame.copy()
                df = opentrades_brut(trtype,df)
                df = closetrades_tp(df,sl,tp,atr)
                if (first==True): 
                    trades=df 
                    first = False
                else:
                    trades=trades.append(df)
    return trades

def opentrades_brut(mode,df):
    df['tradetype'] = mode
    df.loc[df.tradetype==1,'openprice'] = df.openASK
    df.loc[df.tradetype==-1,'openprice'] = df.open
    df.loc[df.tradetype!=0,'openindex'] = df.id
    return df


def closetrades_tsl(df,stoploss,takeprofit,trailsl,atr=''):
    tpr = 0.5
    lotsize = 100000
    risk = 100
    
    if (stoploss<=0.09):
        df['sl'] = stoploss   * 10000
        df['tp'] = takeprofit * 10000
        df['tsl'] = trailsl   * 10000
        df['sl_val'] = stoploss
        df['tp_val'] = takeprofit
        df['tsl_val'] = trailsl
    else:
        df['sl'] = stoploss
        df['tp'] = takeprofit
        df['tsl'] = trailsl
        df['sl_val'] = stoploss   * df[atr]
        df['tp_val'] = takeprofit * df[atr]
        df['tsl_val'] = trailsl   * df[atr]
    
    df.loc[df.tradetype==1,'stoploss'] = df.openprice - df.sl_val
    df.loc[df.tradetype==-1,'stoploss'] = df.openprice + df.sl_val
    df.loc[df.tradetype==1,'takeprofit'] = df.openprice + df.tp_val
    df.loc[df.tradetype==-1,'takeprofit'] = df.openprice - df.tp_val

    df['closeindex'] = -1
    df['tpcloseindex'] = -1
    df['closeprice'] = -1
    df['tpcloseprice'] = -1
    df['slindex'] = -1
    df['slprice'] = -1
    df['profit'] = 0
    df['tsize'] = risk/(lotsize*df.sl_val)
    df.tsize = np.floor(df.tsize*100)/100.0
#     df.tsize = 0.1     # no size scaling
    
    i = 0
    df['nextbar_id'] = df.id.shift(i)
    while ((len(df[(df.tradetype!=0) & (df.closeindex<0) & (df.nextbar_id>0)])>0) ):#& (i>=-500)):
        df['nextbar_open'] = df.open.shift(i)
        df['nextbar_close'] = df.close.shift(i)
        df['nextbar_low'] = df.low.shift(i)
        df['nextbar_high'] = df.high.shift(i)
        df['nextbar_openASK'] = df.openASK.shift(i)
        df['nextbar_closeASK'] = df.closeASK.shift(i)
        df['nextbar_lowASK'] = df.lowASK.shift(i)
        df['nextbar_highASK'] = df.highASK.shift(i)
        df['nextbar_id'] = df.id.shift(i)
        #SL buy
        df.loc[(df.tradetype==1) & (df.closeindex<0)&(df.nextbar_low<=df.stoploss),  'slprice']    = df.nextbar_low
        df.loc[(df.tradetype==1) & (df.closeindex<0)&(df.nextbar_low<=df.stoploss),  'slindex']    = df.nextbar_id
        df.loc[(df.tradetype==1) & (df.closeindex<0)&(df.nextbar_low<=df.stoploss),  'closeprice'] = df.stoploss
        df.loc[(df.tradetype==1) & (df.closeindex==-1)&(df.nextbar_low<=df.stoploss),'profit']     = df.stoploss - df.openprice
        df.loc[(df.tradetype==1) & (df.closeindex==-2)&(df.nextbar_low<=df.stoploss),'profit']     = df.profit + ((df.stoploss - df.openprice) * (1-tpr))
        df.loc[(df.tradetype==1) & (df.closeindex<0)&(df.nextbar_low<=df.stoploss),  'closeindex'] = df.nextbar_id

        #SL sell
        df.loc[(df.tradetype==-1) & (df.closeindex<0)&(df.nextbar_highASK>=df.stoploss),  'slprice']    = df.nextbar_highASK
        df.loc[(df.tradetype==-1) & (df.closeindex<0)&(df.nextbar_highASK>=df.stoploss),  'slindex']    = df.nextbar_id
        df.loc[(df.tradetype==-1) & (df.closeindex<0)&(df.nextbar_highASK>=df.stoploss),  'closeprice'] = df.stoploss
        df.loc[(df.tradetype==-1) & (df.closeindex==-1)&(df.nextbar_highASK>=df.stoploss),'profit']     = df.openprice - df.stoploss
        df.loc[(df.tradetype==-1) & (df.closeindex==-2)&(df.nextbar_highASK>=df.stoploss),'profit']     = df.profit + ((df.openprice - df.stoploss) * (1-tpr))
        df.loc[(df.tradetype==-1) & (df.closeindex<0)&(df.nextbar_highASK>=df.stoploss),  'closeindex'] = df.nextbar_id
        
        #TP buy
        df.loc[(df.tradetype==1)  & (df.closeindex==-1) & (df.nextbar_high>=df.takeprofit),'tpcloseprice'] = df.nextbar_high
        df.loc[(df.tradetype==1)  & (df.closeindex==-1) & (df.nextbar_high>=df.takeprofit),'tpcloseindex'] = df.nextbar_id
        if (trailsl>0):
            df.loc[(df.tradetype==1)  & (df.closeindex==-1) & (df.nextbar_high>=df.takeprofit),'profit']     = df.tp_val * tpr
            df.loc[(df.tradetype==1)  & (df.closeindex==-1) & (df.nextbar_high>=df.takeprofit),'closeindex'] = -2
        else:
            df.loc[(df.tradetype==1)  & (df.closeindex==-1) & (df.nextbar_high>=df.takeprofit),'closeprice'] = df.takeprofit
            df.loc[(df.tradetype==1)  & (df.closeindex==-1) & (df.nextbar_high>=df.takeprofit),'profit']     = df.tp_val
            df.loc[(df.tradetype==1)  & (df.closeindex==-1) & (df.nextbar_high>=df.takeprofit),'closeindex'] = df.nextbar_id
            
        #TP sell
        df.loc[(df.tradetype==-1) & (df.closeindex==-1) & (df.nextbar_lowASK <=df.takeprofit),'tpcloseprice'] = df.nextbar_lowASK
        df.loc[(df.tradetype==-1) & (df.closeindex==-1) & (df.nextbar_lowASK <=df.takeprofit),'tpcloseindex'] = df.nextbar_id
        if (trailsl>0):
            df.loc[(df.tradetype==-1) & (df.closeindex==-1) & (df.nextbar_lowASK <=df.takeprofit),'profit']     = df.tp_val * tpr
            df.loc[(df.tradetype==-1) & (df.closeindex==-1) & (df.nextbar_lowASK <=df.takeprofit),'closeindex'] = -2
        else:
            df.loc[(df.tradetype==-1) & (df.closeindex==-1) & (df.nextbar_lowASK <=df.takeprofit),'closeprice'] = df.takeprofit
            df.loc[(df.tradetype==-1) & (df.closeindex==-1) & (df.nextbar_lowASK <=df.takeprofit),'profit']     = df.tp_val
            df.loc[(df.tradetype==-1) & (df.closeindex==-1) & (df.nextbar_lowASK <=df.takeprofit),'closeindex'] = df.nextbar_id
            
        
        #update SL - trailing SL
        if (takeprofit==100):#podnoś SL po każdnym barze
            df.loc[(df.tradetype==1)&(df.closeindex<0)&(df.nextbar_closeASK-df.sl_val > df.stoploss),'stoploss']=df.nextbar_closeASK - df.sl_val
            df.loc[(df.tradetype==-1)&(df.closeindex<0)&(df.nextbar_close+df.sl_val<df.stoploss),'stoploss'] =df.nextbar_close + df.sl_val
        elif (trailsl>0):#podnoś SL tylko gdy już był profit
            df.loc[(df.tradetype==1)&(df.closeindex==-2)&(df.nextbar_closeASK-df.tsl_val>df.stoploss),'stoploss']=df.nextbar_closeASK - df.tsl_val
            df.loc[(df.tradetype==-1)&(df.closeindex==-2)&(df.nextbar_close+df.tsl_val<df.stoploss),'stoploss']= df.nextbar_close + df.tsl_val
        
        i-=1
    
    print(stoploss,':',takeprofit,':',trailsl,':',i,' open:',len(df[(df.tradetype!=0) & (df.closeindex<0)]))
#     df['sl_val'] = df.sl_val * 10000
#     df['tp_val'] = df.tp_val * 10000
#     df['tsl_val'] = df.tsl_val * 10000

    df['slrisk'] = df.sl_val * df.tsize * lotsize
    df['profit'] = df.profit * df.tsize * lotsize

    df = df.drop(columns='nextbar_close')
    df = df.drop(columns='nextbar_open')
    df = df.drop(columns='nextbar_low')
    df = df.drop(columns='nextbar_high')
    df = df.drop(columns='nextbar_closeASK')
    df = df.drop(columns='nextbar_openASK')
    df = df.drop(columns='nextbar_lowASK')
    df = df.drop(columns='nextbar_highASK')
    df = df.drop(columns='nextbar_id')
    
    
#     df = df.drop(columns='sl_val')
#     df = df.drop(columns='tp_val')
#     df = df.drop(columns='tsl_val')
#     df = df.drop(columns='takeprofit')
    
    return df

def closetrades_tp(df,stoploss,takeprofit,atr=''):
    if (stoploss<=0.09):
        df['sl'] = stoploss   * 10000
        df['tp'] = takeprofit * 10000
        df['sl_val'] = stoploss
        df['tp_val'] = takeprofit
    else:
        df['sl'] = stoploss
        df['tp'] = takeprofit
        df['sl_val'] = df.sl * df[atr]
        df['tp_val'] = df.tp * df[atr]
        
    df.loc[df.tradetype==1,'stoploss'] = df.openprice - df.sl_val
    df.loc[df.tradetype==-1,'stoploss'] = df.openprice + df.sl_val
    df.loc[df.tradetype==1,'takeprofit'] = df.openprice + df.tp_val
    df.loc[df.tradetype==-1,'takeprofit'] = df.openprice - df.tp_val
    df['closeindex'] = -1
    df['tpcloseindex'] = -1
    df['closeprice'] = -1
    df['tpcloseprice'] = -1
#     df['closehour'] = 0
    df['slindex'] = -1
    df['slprice'] = -1
    df['profit'] = 0
    
    i = 0
    df['nextbar_id'] = df.id.shift(i)
    while ((len(df[(df.tradetype!=0) & (df.closeindex<0) & (df.nextbar_id>0)])>0) ):#& (i>=-500)):
#     while ((len(df[(df.tradetype!=0) & (df.closeindex==-1) & (df.id<=lastclose)])>0) & (i>=-500)):
        df['nextbar_open'] = df.open.shift(i)
        df['nextbar_close'] = df.close.shift(i)
        df['nextbar_low'] = df.low.shift(i)
        df['nextbar_high'] = df.high.shift(i)
        df['nextbar_id'] = df.id.shift(i)
        #SL buy
        df.loc[(df.tradetype==1) & (df.closeindex<0)&(df.nextbar_low<=df.stoploss),'slprice'] = df.nextbar_low
        df.loc[(df.tradetype==1) & (df.closeindex<0)&(df.nextbar_low<=df.stoploss),'slindex'] = df.nextbar_id
        df.loc[(df.tradetype==1) & (df.closeindex<0)&(df.nextbar_low<=df.stoploss),'closeprice'] = df.stoploss
        df.loc[(df.tradetype==1) & (df.closeindex<0)&(df.nextbar_low<=df.stoploss),'profit'] = df.stoploss - df.openprice
        df.loc[(df.tradetype==1) & (df.closeindex<0)&(df.nextbar_low<=df.stoploss),'closeindex'] = df.nextbar_id

        #SL sell
        df.loc[(df.tradetype==-1) & (df.closeindex<0)&(df.nextbar_high>=df.stoploss),'slprice'] = df.nextbar_high
        df.loc[(df.tradetype==-1) & (df.closeindex<0)&(df.nextbar_high>=df.stoploss),'slindex'] = df.nextbar_id
        df.loc[(df.tradetype==-1) & (df.closeindex<0)&(df.nextbar_high>=df.stoploss),'closeprice'] = df.stoploss
        df.loc[(df.tradetype==-1) & (df.closeindex<0)&(df.nextbar_high>=df.stoploss),'profit'] = df.openprice - df.stoploss
        df.loc[(df.tradetype==-1) & (df.closeindex<0)&(df.nextbar_high>=df.stoploss),'closeindex'] = df.nextbar_id
        
        #TP buy
        df.loc[(df.tradetype==1)  & (df.closeindex==-1) & (df.nextbar_high>=df.takeprofit),'tpcloseprice'] = df.nextbar_high
        df.loc[(df.tradetype==1)  & (df.closeindex==-1) & (df.nextbar_high>=df.takeprofit),'tpcloseindex'] = df.nextbar_id
        df.loc[(df.tradetype==1)  & (df.closeindex==-1) & (df.nextbar_high>=df.takeprofit),'closeprice'] = df.takeprofit
        df.loc[(df.tradetype==1)  & (df.closeindex==-1) & (df.nextbar_high>=df.takeprofit),'profit'] = df.takeprofit - df.openprice
        df.loc[(df.tradetype==1)  & (df.closeindex==-1) & (df.nextbar_high>=df.takeprofit),'closeindex'] = df.nextbar_id
        #TP sell
        df.loc[(df.tradetype==-1) & (df.closeindex==-1) & (df.nextbar_low <=df.takeprofit),'tpcloseprice'] = df.nextbar_low
        df.loc[(df.tradetype==-1) & (df.closeindex==-1) & (df.nextbar_low <=df.takeprofit),'tpcloseindex'] = df.nextbar_id
        df.loc[(df.tradetype==-1) & (df.closeindex==-1) & (df.nextbar_low <=df.takeprofit),'closeprice'] = df.takeprofit
        df.loc[(df.tradetype==-1) & (df.closeindex==-1) & (df.nextbar_low <=df.takeprofit),'profit'] = df.openprice - df.takeprofit
        df.loc[(df.tradetype==-1) & (df.closeindex==-1) & (df.nextbar_low <=df.takeprofit),'closeindex'] = df.nextbar_id
        
        i-=1
    
    print(stoploss,':',takeprofit,':',i)

    df['sl_val'] = df.sl_val * 10000
    df['tp_val'] = df.tp_val * 10000
    
    
    df['profit'] = df.profit * 10000
    df['profit1'] = np.where(df.profit>=20,1,-1)

    df = df.drop(columns='nextbar_close')
    df = df.drop(columns='nextbar_open')
    df = df.drop(columns='nextbar_low')
    df = df.drop(columns='nextbar_high')
    df = df.drop(columns='nextbar_id')
#     df = df.drop(columns='sl_val')
#     df = df.drop(columns='tp_val')
#     df = df.drop(columns='takeprofit')
    
    return df




def cleartrades_brut(df,save=False,stamp=''):
    
    if (save==True):
        df.to_csv(sep=';',path_or_buf='../Data/trades'+stamp+'.csv',date_format="%Y-%m-%d",index = False,na_rep='',float_format='%.5f')
    
    df = df[df.closeindex!=-1]
    df = df.drop(columns='date')
#     df = df.drop(columns='year')
#     df = df.drop(columns='month')
    df = df.drop(columns='day')
    df = df.drop(columns='weekday')
    df = df.drop(columns='open')
    df = df.drop(columns='low')
    df = df.drop(columns='high')
    df = df.drop(columns='close')
    df = df.drop(columns='volume')
    df = df.drop(columns='openprice')
    df = df.drop(columns='openindex')
    df = df.drop(columns='closeindex')
    df = df.drop(columns='closeprice')
    df = df.drop(columns='slindex')
    df = df.drop(columns='slprice')
    df = df.drop(columns='id')
    print('trades_len: ',len(df))
    return df


def runstats_brut_v3(alltrades):
    conf   = {}
    params = {}

    params['tradetype'] = [1,[1]]
    params['sl'] = [2,[10,20,30,40,50,60]]
    params['tp'] = [2,[10,20,30,40,50,60]]
#     params['tsl'] = [2,[10,20,30,40,50,60]]
    conf['filename'] = 'brut_2015_2021_3_'
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats


def runstats_ma_v3(alltrades):
    conf   = {}
    params = {}

    params['tradetype'] = [1,[1]]
    params['sl'] = [2,[10,20,30,40,50,60]]
    params['tp'] = [2,[10,20,30,40,50,60]]
    params['tsl'] = [2,[10,20,30,40,50,60]]
    params['ma5SMAdiff_prev']      = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAclose_prev']     = [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_2015_2021_1_'
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v4(alltrades,a,b,c,d):
    conf   = {}
    params = {}

    params['tradetype'] = [1,[1]]
    params['sl'] = [2,[10,20,30,40,50,60]]
    params['tp'] = [2,[10,20,30,40,50,60]]
#     params['tsl'] = [2,[10,20,30,40,50,60]]
    params[a]      = [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  = [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  = [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']      = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAclose_prev']     = [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAdiff2_prev']     = [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_2015_2021_1_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v5(alltrades,a,b,c,d):
    conf   = {}
    params = {}

    params['tradetype'] = [1,[1]]
    params['sl'] = [2,[0.1,0.2,0.3,0.4,0.5,0.6]]
    params['tp'] = [2,[0.1,0.2,0.3,0.4,0.5,0.6]]
#     params['sl'] = [2,[0.6]]
#     params['tp'] = [2,[0.6]]
#     params['tsl'] = [2,[10,20,30,40,50,60]]
    params[a]      = [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  = [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  = [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']      = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAclose_prev']     = [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAdiff2_prev']     = [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_2015_2021_1_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats


def runstats_ma_v6(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [1,[1]]
    params['sl'] =             [2,[0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3]]
    params['tp'] =             [2,[0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3]]
    params[atr] =              [3,[-1000,0.005,0.0075,0.01,0.015,0.02],[0.005,0.0075,0.01,0.015,0.02,1000]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
#    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
#    params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2015_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v7(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [1,[-1]]
    params['sl'] =             [2,[30,40,50,60,70,80,100,120,140]]
    params['tp'] =             [2,[30,40,50,60,70,80,100,120,140]]
    params[atr] =              [3,[-1000,0.0075,0.01,0.015],[0.0075,0.01,0.015,1000]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
#    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
#    params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_2015_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v8(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [1,[1]]
    params['sl'] =             [2,[30,40,50,60,70,80,100,120,140]]
    params['tp'] =             [2,[30,40,50,60,70,80,100,120,140]]
    params[atr] =              [3,[-1000,-1000,-1000,-1000,-1000],[0.006,0.007,0.008,0.009,0.01]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
#    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
#    params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_2015_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v9(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [1,[-1]]
    params['sl'] =             [2,[0.5,0.6,0.7,0.8,0.9]]
    params['tp'] =             [2,[0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3]]
    params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v10(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [1,[1]]
    params['sl'] =             [2,[0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
    params['tp'] =             [2,[0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]]
    params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v11(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [1,[1]]
    params['sl'] =             [2,[0.4,0.5,0.6,0.7,0.8,0.9]]
    params['tp'] =             [2,[0.6,0.7,0.8,0.9,1,1.1,1.2]]
    params[atr] =              [3,[-1000,0.006,0.007,0.008,0.009,0.01,0.0125,0.015],[0.006,0.007,0.008,0.009,0.01,0.0125,0.015,1000]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats


def runstats_ma_v12(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[1]]
    params['sl'] =             [2,[0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
    params['tp'] =             [2,[0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]]
    params['srs0sr_broke_prev'] = [3,[-10,10,-100],[0,100,100]]
    params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v13(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [3,[-1,1,-100],[0,100,100]]
    params['sl'] =             [2,[0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
    params['tp'] =             [2,[0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]]
    params[atr] =              [3,[-1000],[0.015]]
#     params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v14(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[1]]
    params['sl'] =             [2,[0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
    params['tp'] =             [2,[0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]]
    params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v15(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[1]]
    params['sl'] =             [2,[0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
    params['tp'] =             [2,[0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]]
#     params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats
def runstats_ma_v16(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[1]]
    params['sl'] =             [2,[30,40,50,60,70,80,90,100]]
    params['tp'] =             [2,[60,70,80,90,100,110,120,130,140,150]]
#     params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v17(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[-1]]
    params['sl'] =             [2,[0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
    params['tp'] =             [2,[0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]]
#     params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v18(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[-1]]
    params['sl'] =             [2,[30,40,50,60,70,80,90,100]]
    params['tp'] =             [2,[60,70,80,90,100,110,120,130,140,150]]
#     params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v20(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[1]]
    params['sl'] =             [2,[1]]
    params['tp'] =             [2,[10]]
    params['tsl'] =            [2,[2]]
#     params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v21(alltrades,a,b,c,d,atr='atr140atr_prev'):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[1]]
    params['sl'] =             [2,[1,1.2,1.4,1.6,1.8,2,2.2]]
    params['tp'] =             [2,[1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2]]
    params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v22(alltrades,a,b,c,d,atr='atr140atr_prev',sl=[],tp=[]):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[1]]
    params['sl'] =             [2,sl]
    params['tp'] =             [2,tp]
    params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_frac_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_ma_v23(alltrades,a,b,c,d,atr='atr140atr_prev',sl=[],tp=[],tsl=[]):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[1]]
    params['sl'] =             [2,sl]
    params['tp'] =             [2,tp]
    params['tsl'] =            [2,tsl]
    params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
#     params[d]     =            [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAclose_prev']=[0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_23_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats    


def runstats24(alltrades,ma1,atrperiod,sl,tp,tsl):
    ma2 = '5'
    stats1 = runstats_ma_v24(alltrades,'ma'+ma1+'SMAdiff_prev','ma'+ma1+'SMAdiffdiff_prev','ma'+ma1+'SMAdiff2_prev','ma'+ma1+'SMAvs'+ma2+'_prev','ma'+ma2+'SMAdiff_prev','ma'+ma2+'SMAdiffdiff_prev','ma'+ma2+'SMAdiff2_prev',atrperiod,sl,tp,tsl)
    ma2 = '10'
    stats1 = runstats_ma_v24(alltrades,'ma'+ma1+'SMAdiff_prev','ma'+ma1+'SMAdiffdiff_prev','ma'+ma1+'SMAdiff2_prev','ma'+ma1+'SMAvs'+ma2+'_prev','ma'+ma2+'SMAdiff_prev','ma'+ma2+'SMAdiffdiff_prev','ma'+ma2+'SMAdiff2_prev',atrperiod,sl,tp,tsl)
    ma2 = '20'
    stats1 = runstats_ma_v24(alltrades,'ma'+ma1+'SMAdiff_prev','ma'+ma1+'SMAdiffdiff_prev','ma'+ma1+'SMAdiff2_prev','ma'+ma1+'SMAvs'+ma2+'_prev','ma'+ma2+'SMAdiff_prev','ma'+ma2+'SMAdiffdiff_prev','ma'+ma2+'SMAdiff2_prev',atrperiod,sl,tp,tsl)
    return stats1

def runstats_ma_v24(alltrades,a,b,c,d,aa,bb,cc,atr='atr140atr_prev',sl=[],tp=[],tsl=[]):
    conf   = {}
    params = {}

    params['tradetype'] = [2,[1]]
    params['sl'] =        [2,sl]
    params['tp'] =        [2,tp]
    params[atr]  =        [3,[-1000],[0.015]]
    params[a]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[d]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[aa]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[bb]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[cc]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] =    'ma_24_2003_2021_1_'+atr+'_'+d
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats


def runstats_ma_v25(alltrades,atr='atr140atr_prev',sl=[],tp=[],tsl=[]):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[1]]
    params['sl'] =             [2,sl]
    params['tp'] =             [2,tp]
    if (tsl!=[]):
        params['tsl'] =            [2,tsl]
    params[atr] =              [3,[-1000],[0.015]]
    params['momentum7close_prev']      =               [0,[-1000,0,1000],[-1000,0,1000]]
    params['momentum7direction_prev']  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params['macd12HistMACD_prev']      =               [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_25_2003_2021_1_'+atr
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats   



def runstats_ma_v26(alltrades,a,b,c,d,atr='atr140atr_prev',sl=[],tp=[],tsl=[]):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[1]]
    params['sl'] =             [2,sl]
    params['tp'] =             [2,tp]
    params['tsl'] =            [2,tsl]
    params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[d]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
#     params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_26_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats    

def runstats_ma_v27(alltrades,a,b,c,d,atr='atr140atr_prev',sl=[],tp=[],tsl=[]):
    conf   = {}
    params = {}

    params['tradetype'] =      [2,[1]]
    params['sl'] =             [2,sl]
    params['tp'] =             [2,tp]
#     params['tsl'] =            [2,tsl]
    params[atr] =              [3,[-1000],[0.015]]
    params[a]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params[d]  =               [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiff_prev']= [0,[-1000,0,1000],[-1000,0,1000]]
    params['ma5SMAdiffdiff_prev']  = [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] = 'ma_27_2003_2021_1_'+atr+'_'+a
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats    

def runstats28(alltrades,ma1,atrperiod,sl,tp,tsl):
    ma2 = '5'
    stats1 = runstats_ma_v28(alltrades,'ma'+ma1+'SMAdiff_prev','ma'+ma1+'SMAdiffdiff_prev','ma'+ma1+'SMAdiff2_prev','ma'+ma1+'SMAvs'+ma2+'_prev','ma'+ma2+'SMAdiff_prev','ma'+ma2+'SMAdiffdiff_prev','ma'+ma2+'SMAdiff2_prev',atrperiod,sl,tp,tsl)
    ma2 = '10'
    stats1 = runstats_ma_v28(alltrades,'ma'+ma1+'SMAdiff_prev','ma'+ma1+'SMAdiffdiff_prev','ma'+ma1+'SMAdiff2_prev','ma'+ma1+'SMAvs'+ma2+'_prev','ma'+ma2+'SMAdiff_prev','ma'+ma2+'SMAdiffdiff_prev','ma'+ma2+'SMAdiff2_prev',atrperiod,sl,tp,tsl)
    ma2 = '20'
    stats1 = runstats_ma_v28(alltrades,'ma'+ma1+'SMAdiff_prev','ma'+ma1+'SMAdiffdiff_prev','ma'+ma1+'SMAdiff2_prev','ma'+ma1+'SMAvs'+ma2+'_prev','ma'+ma2+'SMAdiff_prev','ma'+ma2+'SMAdiffdiff_prev','ma'+ma2+'SMAdiff2_prev',atrperiod,sl,tp,tsl)
    return stats1

def runstats_ma_v28(alltrades,a,b,c,d,aa,bb,cc,atr='atr140atr_prev',sl=[],tp=[],tsl=[]):
    conf   = {}
    params = {}

    params['tradetype'] = [2,[1]]
    params['sl'] =        [2,sl]
    params['tp'] =        [2,tp]
    params['tsl'] =       [2,tsl]
    params[atr]  =        [3,[-1000],[0.015]]
    params[a]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[d]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[aa]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[bb]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[cc]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] =    'ma_28_2003_2021_1_'+atr+'_'+d
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats


def runstats29(alltrades,ma1,atrperiod,sl,tp,tsl):
    ma2 = '5'
    stats1 = runstats_ma_v29(alltrades,'ma'+ma1+'SMAdiff_prev','ma'+ma1+'SMAdiffdiff_prev','ma'+ma1+'SMAdiff2_prev','ma'+ma1+'SMAvs'+ma2+'_prev','ma'+ma2+'SMAdiff_prev','ma'+ma2+'SMAdiffdiff_prev','ma'+ma2+'SMAdiff2_prev',atrperiod,sl,tp,tsl)
    return stats1

def runstats_ma_v29(alltrades,a,b,c,d,aa,bb,cc,atr='atr140atr_prev',sl=[],tp=[],tsl=[]):
    conf   = {}
    params = {}

    params['tradetype'] = [2,[1]]
    params['sl'] =        [2,sl]
    params['tp'] =        [2,tp]
    params['tsl'] =       [2,tsl]
    params[atr]  =        [3,[-1000],[0.015]]
    params[a]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[d]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[aa]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[bb]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[cc]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] =    'ma_29_2003_2021_1_'+atr+'_'+d
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats

def runstats_test1(alltrades,ma1,atrperiod,sl,tp,tsl):
    ma2 = '5'
    stats1 = runstats_ma_test1(alltrades,'ma'+ma1+'SMAdiff_prev','ma'+ma1+'SMAdiffdiff_prev','ma'+ma1+'SMAdiff2_prev','ma'+ma1+'SMAvs'+ma2+'_prev','ma'+ma2+'SMAdiff_prev','ma'+ma2+'SMAdiffdiff_prev','ma'+ma2+'SMAdiff2_prev',atrperiod,sl,tp,tsl)
    return stats1

def runstats_ma_test1(alltrades,a,b,c,d,aa,bb,cc,atr='atr140atr_prev',sl=[],tp=[],tsl=[]):
    conf   = {}
    params = {}

    params['tradetype'] = [2,[1]]
    params['sl'] =        [2,sl]
    params['tp'] =        [2,tp]
    params['tsl'] =       [2,tsl]
    params[atr]  =        [3,[-1000],[0.015]]
    params[a]    =        [3,[-1000],[1000]]
    params[b]    =        [3,[-1000],[1000]]
    params[c]    =        [3,[0],[1000]]
    params[d]    =        [3,[0],[1000]]
    params[aa]   =        [3,[-1000],[1000]]
    params[bb]   =        [3,[-1000],[0]]
    params[cc]   =        [3,[-1000],[1000]]
    conf['filename'] =    'ma_t1_2003_2021_1_'+atr+'_'+d
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return stats


def runstats30(alltrades,ma1,atrperiod,sl,tp,tsl):
    ma2 = '5'
    runstats_ma_v30(alltrades,'ma'+ma1+'SMAdiff_prev','ma'+ma1+'SMAdiffdiff_prev','ma'+ma1+'SMAdiff2_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev','ma'+ma2+'SMAdiff_prev','ma'+ma2+'SMAdiffdiff_prev','ma'+ma2+'SMAdiff2_prev',atrperiod,sl,tp,tsl,'x1')
    runstats_ma_v30(alltrades,'ma'+ma1+'SMAdiff_prev','ma'+ma1+'SMAdiffdiff_prev','ma'+ma1+'SMAdiff2_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev','ma'+ma2+'SMAdiff_prev','ma'+ma2+'SMAdiffdiff_prev','ma'+ma2+'SMAdiff2_prev',atrperiod,sl,[100],[0],'x2')
    return 

def runstats_ma_v30(alltrades,a,b,c,d,aa,bb,cc,atr='atr140atr_prev',sl=[],tp=[],tsl=[],ff=''):
    conf   = {}
    params = {}

    params['tradetype'] = [2,[1]]
    params['sl'] =        [2,sl]
    params['tp'] =        [2,tp]
    params['tsl'] =       [2,tsl]
    params[atr]  =        [3,[-1000],[0.015]]
    params[a]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[d]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[aa]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[bb]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[cc]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] =    'ma_30_2003_2021_1_'+atr+'_'+d+ff
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return 

def runstats31(alltrades,ma1,atrperiod,sl,tp,tsl):
    ma2 = '10'
    runstats_ma_v31(alltrades,'ma'+ma1+'SMAdiff_prev','ma'+ma1+'SMAdiffdiff_prev','ma'+ma1+'SMAdiff2_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev','ma'+ma2+'SMAdiff_prev','ma'+ma2+'SMAdiffdiff_prev','ma'+ma2+'SMAdiff2_prev',atrperiod,sl,tp,tsl,'x1')
    runstats_ma_v31(alltrades,'ma'+ma1+'SMAdiff_prev','ma'+ma1+'SMAdiffdiff_prev','ma'+ma1+'SMAdiff2_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev','ma'+ma2+'SMAdiff_prev','ma'+ma2+'SMAdiffdiff_prev','ma'+ma2+'SMAdiff2_prev',atrperiod,sl,[100],[0],'x2')
    return 

def runstats_ma_v31(alltrades,a,b,c,d,aa,bb,cc,atr='atr140atr_prev',sl=[],tp=[],tsl=[],ff=''):
    conf   = {}
    params = {}

    params['tradetype'] = [2,[1]]
    params['sl'] =        [2,sl]
    params['tp'] =        [2,tp]
    params['tsl'] =       [2,tsl]
    params[atr]  =        [3,[-1000],[0.015]]
    params[a]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[b]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[c]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[d]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[aa]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[bb]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[cc]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] =    'ma_31_2003_2021_1_'+atr+'_'+d+ff
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return 





def runstats32(alltrades,ma1,atrperiod,sl,tp,tsl):
    ma2 = '5'
#     runstats_ma_v32(alltrades,'ma'+ma1+'SMAdiffseq_prev', 'ma'+ma1+'SMAdiffdiff_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev', 'ma'+ma2+'SMAdiffseq_prev', 'ma'+ma2+'SMAdiffdiff_prev', atrperiod,sl,tp,tsl,'x1')
    runstats_ma_v32(alltrades,'ma'+ma1+'SMAdiffseq_prev','ma'+ma1+'SMAdiffdiff_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev', 'ma'+ma2+'SMAdiffseq_prev', 'ma'+ma2+'SMAdiffdiff_prev', atrperiod,sl,[100],[0],'x2')
    return 

def runstats_ma_v32(alltrades,a,b,sv,aa,bb,atr='atr140atr_prev',sl=[],tp=[],tsl=[],ff=''):
    conf   = {}
    params = {}

    params['tradetype'] = [2,[1]]
    params['sl'] =        [2,sl]
    params['tp'] =        [2,tp]
    params['tsl'] =       [2,tsl]
    params[atr]  =        [3,[-1000],[0.015]]
    params[a]    =        [0,[-1000,-1,1,2,3,4,5],[1000]]
    params[b]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[sv]    =       [0,[-1000,0,1000],[-1000,0,1000]]
    params[aa]   =        [0,[-1000,-1,1,2,3,4,5],[1000]]
    params[bb]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] =    'ma_32_2003_2021_1_'+atr+'_'+sv+ff
    print(conf['filename'])
    stats = stathyperparams2(alltrades,params,conf)
    return 


def runstats33(alltrades,ma1,atrperiod,sl,tp,tsl):
    ma2 = '5'
    runstats_ma_v33(alltrades,'ma'+ma1+'SMAdiffseq_prev', 'ma'+ma1+'SMAdiffdiff_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev', 'ma'+ma2+'SMAdiffseq_prev', 'ma'+ma2+'SMAdiffdiff_prev', atrperiod,sl,tp,tsl,'x1')
    runstats_ma_v33(alltrades,'ma'+ma1+'SMAdiffseq_prev','ma'+ma1+'SMAdiffdiff_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev', 'ma'+ma2+'SMAdiffseq_prev', 'ma'+ma2+'SMAdiffdiff_prev', atrperiod,sl,[100],[0],'x2')
    return 

def runstats_ma_v33(alltrades,a,b,sv,aa,bb,atr='atr140atr_prev',sl=[],tp=[],tsl=[],ff=''):
    conf   = {}
    params = {}

    params['tradetype'] = [2,[1]]
    params['sl'] =        [2,sl]
    params['tp'] =        [2,tp]
    params['tsl'] =       [2,tsl]
    params[atr]  =        [3,[-1000],[0.015]]
    params[a]    =        [0,[-5,1,2,3,4,5],[1,2,3,4,5,1000]]
    params[b]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[sv]    =       [0,[-1000,0,1000],[-1000,0,1000]]
    params[aa]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[bb]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] =    'ma_33_2003_2021_1_'+atr+'_'+sv+ff
    print(conf['filename'])
    stathyperparams2(alltrades,params,conf)
    return 

def runstats34(alltrades,ma1,atrperiod,sl,tp,tsl):
    ma2 = '3'
    runstats_ma_v34(alltrades,'ma'+ma1+'SMAdiffseq_prev', 'ma'+ma1+'SMAdiffdiff_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev', 'ma'+ma2+'SMAdiffseq_prev', 'ma'+ma2+'SMAdiffdiff_prev', atrperiod,sl,tp,tsl,'x1')
    ma2 = '5'
    runstats_ma_v34(alltrades,'ma'+ma1+'SMAdiffseq_prev', 'ma'+ma1+'SMAdiffdiff_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev', 'ma'+ma2+'SMAdiffseq_prev', 'ma'+ma2+'SMAdiffdiff_prev', atrperiod,sl,tp,tsl,'x1')
    ma2 = '7'
    runstats_ma_v34(alltrades,'ma'+ma1+'SMAdiffseq_prev', 'ma'+ma1+'SMAdiffdiff_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev', 'ma'+ma2+'SMAdiffseq_prev', 'ma'+ma2+'SMAdiffdiff_prev', atrperiod,sl,tp,tsl,'x1')

    return 

def runstats_ma_v34(alltrades,a,b,sv,aa,bb,atr='atr140atr_prev',sl=[],tp=[],tsl=[],ff=''):
    conf   = {}
    params = {}

    params['tradetype'] = [2,[1]]
    params['sl'] =        [2,sl]
    params['tp'] =        [2,tp]
    params['tsl'] =       [2,tsl]
    params[atr]  =        [3,[-1000],[0.015]]
    params[a]    =        [0,[-5,1,2,3,4,5],[1,2,3,4,5,1000]]
    params[b]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[sv]    =       [0,[-1000,0,1000],[-1000,0,1000]]
    params[aa]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[bb]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] =    'ma_34_2003_2021_1_'+atr+'_'+sv+ff
    print(conf['filename'])
    stathyperparams2(alltrades,params,conf)
    return 

def runstats35(alltrades,ma1,atrperiod,sl,tp,tsl):
    ma2 = '3'
    runstats_ma_v35(alltrades,'ma'+ma1+'SMAdiffseq_prev', 'ma'+ma1+'SMAdiffdiff_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev', 'ma'+ma2+'SMAdiffseq_prev', 'ma'+ma2+'SMAdiffdiff_prev', atrperiod,sl,tp,tsl,'x1')
    ma2 = '5'
    runstats_ma_v35(alltrades,'ma'+ma1+'SMAdiffseq_prev', 'ma'+ma1+'SMAdiffdiff_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev', 'ma'+ma2+'SMAdiffseq_prev', 'ma'+ma2+'SMAdiffdiff_prev', atrperiod,sl,tp,tsl,'x1')
    ma2 = '7'
    runstats_ma_v35(alltrades,'ma'+ma1+'SMAdiffseq_prev', 'ma'+ma1+'SMAdiffdiff_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev', 'ma'+ma2+'SMAdiffseq_prev', 'ma'+ma2+'SMAdiffdiff_prev', atrperiod,sl,tp,tsl,'x1')

    return 

def runstats_ma_v35(alltrades,a,b,sv,aa,bb,atr='atr140atr_prev',sl=[],tp=[],tsl=[],ff=''):
    conf   = {}
    params = {}

    params['tradetype'] = [2,[1]]
    params['sl'] =        [2,sl]
    params['tp'] =        [2,tp]
    params['tsl'] =       [2,tsl]
    params[atr]  =        [3,[-1000],[0.015]]
    params[a]    =        [0,[-5,1,2,3,4,5],[1,2,3,4,5,1000]]
    params[b]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[sv]    =       [0,[-1000,0,1000],[-1000,0,1000]]
    params[aa]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[bb]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] =    'ma_35_2003_2021_1_'+atr+'_'+sv+ff
    print(conf['filename'])
    stathyperparams2(alltrades,params,conf)
    return 


def runstats_ma_v36(alltrades,a,b,atr='atr140atr_prev',sl=[],tp=[],tsl=[],ff=''):
    conf   = {}
    params = {}

    params['tradetype'] = [2,[1]]
    params['sl'] =        [2,sl]
    params['tp'] =        [2,tp]
    params['tsl'] =       [2,tsl]
    params[atr]  =        [3,[-1000],[0.015]]
    params[a]    =        [0,[-5,1,2,3,4,5],[1,2,3,4,5,1000]]
    params[b]    =        [0,[0,10,20,30,40,50,60,70,80,90],[10,20,30,40,50,60,70,80,90,100]]
#     params[b]    =        [3,[0],[30]]
    conf['filename'] =    'ma_36_2003_2021_1_'+atr+'_'+ff
    print(conf['filename'])
    stathyperparams2(alltrades,params,conf)
    return 

def runstats40(alltrades,ma1,atrperiod,sl,tp,tsl,ma2=5):
    ma2 = str(ma2)
    runstats_ma_v40(alltrades,'ma'+ma1+'SMAdiffseq_prev', 'ma'+ma1+'SMAdiffdiff_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev', 'ma'+ma2+'SMAdiffseq_prev', 'ma'+ma2+'SMAdiffdiff_prev', atrperiod,sl,tp,tsl,'x1')
    runstats_ma_v40(alltrades,'ma'+ma1+'SMAdiffseq_prev', 'ma'+ma1+'SMAdiffdiff_prev', 'ma'+ma1+'SMAvs'+ma2+'_prev', 'ma'+ma2+'SMAdiffseq_prev', 'ma'+ma2+'SMAdiffdiff_prev', atrperiod,sl,[100],[0],'x2')

    return 

def runstats_ma_v40(alltrades,a,b,sv,aa,bb,atr='atr140atr_prev',sl=[],tp=[],tsl=[],ff=''):
    conf   = {}
    params = {}

    params['tradetype'] = [2,[1]]
    params['sl'] =        [2,sl]
    params['tp'] =        [2,tp]
    params['tsl'] =       [2,tsl]
    params[atr]  =        [3,[-1000],[0.015]]
    params[a]    =        [0,[-5,1,2,3,4,5],[1,2,3,4,5,1000]]
    params[b]    =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[sv]    =       [0,[-1000,0,1000],[-1000,0,1000]]
    params[aa]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    params[bb]   =        [0,[-1000,0,1000],[-1000,0,1000]]
    conf['filename'] =    'ma_40_2003_2021_1_'+atr+'_'+sv+ff
    print(conf['filename'])
    stathyperparams2(alltrades,params,conf)
    return 



