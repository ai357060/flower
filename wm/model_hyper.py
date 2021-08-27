import pandas as pd
import numpy as np
import time
import importlib
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

import model_collection
from model_collection import *

pd.options.display.max_columns = None

def runhypermodel(fn, fver, featsel='pca',featcount=[5,15,25],models=[['rf','svc','mlp'],['rf','svc','mlp'],['rf','svc','mlp']],testone=False,has_y=False,testsize=0.25):

    masterframe = loaddata_master('../Data/'+fn+'_'+fver+'.csv')

#     atr = masterframe.atr14tr.mean()
#     print(atr)
    atr = calculate_atr(masterframe)
    if has_y == False:
        print(atr)
        prepare_y(masterframe,atr)

    #pre-prepare data
    orygframe = masterframe.copy()
    masterframe = masterframe.drop(['volume'],1)
    masterframe = masterframe.drop(['fulldate'],1)
    masterframe = masterframe.drop(['year'],1)
# tu można usunąć z masterframe: fulldate	year	month	day	open	high	low	close
    try:
        masterframe = masterframe.drop(['month'],1)
    except:
        pass
    try:
        masterframe = masterframe.drop(['day'],1)
    except:
        pass
    masterframe = masterframe.drop(['open'],1)
    masterframe = masterframe.drop(['high'],1)
    masterframe = masterframe.drop(['low'],1)
    masterframe = masterframe.drop(['close'],1)
    
#     masterframe[['open','high','low','close']] = normalize_together(masterframe[['open','high','low','close']])
    
    masterframe.dropna(inplace=True)

    # split data
    # masterframe = masterframe[-3600:] ###testowo
    masterframe = masterframe.drop(masterframe[masterframe.y==-1].index,axis=0)

    
    #Scale
#     Dla v10 nie trzeba skalować to już skalowanie jest w danych
#     if fver.startswith('v10'):
#         print('sc_10')
#         # wszystko już wyskalowane
#     elif fver.startswith('v14'):    
#         print('sc_14')
#         columns=masterframe.filter(regex='^hD|^hW|^hM').columns
#         masterframe[columns] = normalize_together(masterframe[columns])
#     else:
#         print('sc_16')
#         columns=masterframe.filter(regex='^hD|^hW|^hM').columns
#         masterframe[columns] = normalize_together(masterframe[columns])
    
    
    X_df = masterframe.iloc[:, :-1] # tu było 2:-1 aby pominąć fulldate i year, ale teraz powyżej usuwam te kolumny plus inne
    y_df = masterframe.iloc[:, -1] 
    featurenames = masterframe.iloc[:, :-1].columns.values

    X = X_df.values
    y = y_df.values
    y = y.astype('int')
    X = X.astype('float')
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=testsize, shuffle = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None, test_size=testsize, shuffle = False)
    
# balance classes
    from imblearn.over_sampling import SMOTE
#    from imblearn.under_sampling import RandomUnderSampler
#    from imblearn.over_sampling import RandomOverSampler
    # from imblearn.over_sampling import ADASYN
    # from imblearn.over_sampling import BorderlineSMOTE
    # from imblearn.over_sampling import SVMSMOTE
    sm = SMOTE(random_state=27,sampling_strategy='minority')
#    sm = RandomUnderSampler(random_state=27,sampling_strategy='majority')######
#    sm = RandomOverSampler(random_state=27)
    # sm = ADASYN(random_state=27)
    # sm = BorderlineSMOTE(random_state=27)
    # sm = SVMSMOTE(random_state=27)
    X_train, y_train = sm.fit_sample(X_train, y_train)

    # odtworzenie masterframe dla NN
    traindf = pd.DataFrame(X_train, columns=X_df.columns)
    traindf[masterframe.columns[-1]] = y_train
    masterframeN = traindf.sort_values(by=['id'])


#    X_test, y_test = sm.fit_sample(X_test, y_test)   # undersample test
    testdf = pd.DataFrame(X_test, columns=X_df.columns)
    testdf[masterframe.columns[-1]] = y_test
#     testdf = testdf.sort_values(by=['id'])
    masterframeN = pd.concat([masterframeN,testdf])
    
    masterframe = masterframeN

    
# tu można usunąć z Xtrain i Xtest: id!!!
    Xintex = X_test[:,0]     # tylko kolumna id
    X_test = X_test[:,1:]    # wszystkie oprócz kolumny id
    X_train = X_train[:,1:]  # wszystkie oprócz kolumny id
    
    
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)      
    
#     X_train_sc = X_train
#     X_test_sc = X_test

    
    modeltype = {}
    # featcount = []
#     testone=False
    # featsel = 'rf'
    # featsel = 'svc'
    # featsel = 'pca'
    # featsel = 'all'
    # featcount.append(5);modeltype[5]   = ['rf','svc','mlp']
    # featcount.append(10);modeltype[10] = ['rf','svc','mlp']
    # featcount.append(15);modeltype[15] = ['rf','svc','mlp']
    # featcount.append(20);modeltype[20] = ['rf','svc','mlp']
    # featcount.append(25);modeltype[25] = ['rf','svc','mlp']    
    
    model = []
    for i in range(0,len(featcount)):
        modeltype[featcount[i]] = models[i]
    
    if featsel != 'all':
        for i in featcount:
            if (('rf' in modeltype[i]) | ('svc' in modeltype[i]) | ('mlp' in modeltype[i])):
                print('FEATSEL:'+featsel+str(i)+'________________________________________________________________________________________')

                if featsel == 'rf':
                    select = RFE(RandomForestClassifier(n_estimators=100,random_state=2,n_jobs=1),n_features_to_select=i)
                    select.fit(X_train, y_train)
                    X_train_rfe= select.transform(X_train) 
                    X_test_rfe= select.transform(X_test)
                    X_train_sc_rfe= select.transform(X_train_sc)  
                    X_test_sc_rfe= select.transform(X_test_sc)    

                if featsel == 'svc':
                    select = RFE(SVC(kernel='linear')            ,n_features_to_select=i)
                    select.fit(X_train_sc, y_train)
                    X_train_rfe= select.transform(X_train) 
                    X_test_rfe= select.transform(X_test)
                    X_train_sc_rfe= select.transform(X_train_sc)  
                    X_test_sc_rfe= select.transform(X_test_sc)  

                if featsel == 'pca':
                    select = PCA(n_components=i, whiten=True, random_state=2)
                    select.fit(X_train)
                    X_train_rfe= select.transform(X_train) 
                    X_test_rfe= select.transform(X_test)
                    X_train_sc_rfe= X_train_rfe
                    X_test_sc_rfe= X_test_rfe

            #     select = PCA(n_components=i, whiten=False, random_state=2)
            #     select.fit(X_train_sc)
            #     X_train_rfe= select.transform(X_train) 
            #     X_test_rfe= select.transform(X_test)
            #     X_train_sc_rfe= select.transform(X_train_sc)
            #     X_test_sc_rfe= select.transform(X_test_sc)
            #     featsel = 'pca_nw'

                if testone == True:
                    featsel = 'temp_'+featsel

                # visualize the selected features:
                #mask = select.get_support()
                #plt.matshow(mask.reshape(1, -1), cmap='gray_r')
                #plt.xlabel("Sample index")
                #print(X_df.iloc[:2,mask])
                #print("Test score: {:.3f}".format(select.score(X_test_sc, y_test)))
                #print("Test score: {:.3f}".format(select.score(X_test, y_test)))

                #lin_resdf = ExamineLogisticRegression(orygframe,Xintex,X_train_rfe, y_train,X_test_rfe, y_test,featurenames,testone=False,plot=False)
                #lin_resdf.to_csv(sep=';',path_or_buf='Resu/'+fver+'_'+featsel+str(i)+'_LogisticRegression'+str(int(time.time()))+'.csv',date_format="%Y-%m-%d",index = False)

                #lin_resdf = ExamineLinearSVC(orygframe,Xintex,X_train_rfe, y_train,X_test_rfe, y_test,featurenames,testone=False,plot=False)
                #lin_resdf.to_csv(sep=';',path_or_buf='Resu/'+fver+'_'+featsel+str(i)+'_LinearSVC'+str(int(time.time()))+'.csv',date_format="%Y-%m-%d",index = False)


                if 'rf' in modeltype[i]:
                    print('FEATSEL:'+featsel+str(i)+'_model_rf_________________________________________________________________________')
                    forest_resdf = ExamineRandomForest(orygframe, Xintex, X_train_rfe, y_train,X_test_rfe, y_test, featurenames,atr, testone=testone, plot=False, automaxfeat=True)
                    forest_resdf.to_csv(sep=';', path_or_buf='../Resu/'+fver+'_'+featsel+str(i)+'_RandomForest'+str(int(time.time()))+'.csv', date_format="%Y-%m-%d", index = False)

                if 'svc' in modeltype[i]:
                    print('FEATSEL:'+featsel+str(i)+'_model_svc_________________________________________________________________________')
                    svc_resdf = ExamineSVC(orygframe, Xintex, X_train_sc_rfe, y_train, X_test_sc_rfe, y_test,featurenames,atr, testone=testone, plot=False)
                    svc_resdf.to_csv(sep=';', path_or_buf='../Resu/'+fver+'_'+featsel+str(i)+'_SVC'+str(int(time.time()))+'.csv', date_format="%Y-%m-%d", index = False)

                if 'mlp' in modeltype[i]:
                    print('FEATSEL:'+featsel+str(i)+'_model_mlp_________________________________________________________________________')
                    mlp_resdf = ExamineMLP(orygframe, Xintex, X_train_sc_rfe, y_train, X_test_sc_rfe, y_test, featurenames,atr, testone=testone, plot=False)
                    mlp_resdf.to_csv(sep=';', path_or_buf='../Resu/'+fver+'_'+featsel+str(i)+'_MLP'+str(int(time.time()))+'.csv', date_format="%Y-%m-%d", index = False)

                    
        print('FEATSEL________________finished________________________________________________________________________________')
            
        
    if featsel == 'all':
        if 'rf' in modeltype[0]:
            print('ALL_____________rf___________________________________________________________________________')
            featsel = 'all_RandomForest' if testone == False else 'temp_all_RandomForest'
            forest_resdf = ExamineRandomForest(orygframe,Xintex,X_train, y_train,X_test, y_test,featurenames,atr,testone=testone,plot=False,automaxfeat=False)
            forest_resdf.to_csv(sep=';',path_or_buf='../Resu/'+fver+'_'+featsel+str(int(time.time()))+'.csv',date_format="%Y-%m-%d",index = False)

        if 'svc' in modeltype[0]:
            print('ALL_____________svc___________________________________________________________________________')
            featsel = 'all_SVC' if testone == False else 'temp_all_SVC'
            svc_resdf = ExamineSVC(orygframe,Xintex,X_train_sc, y_train,X_test_sc, y_test,featurenames,atr,testone=testone,plot=False)
            svc_resdf.to_csv(sep=';',path_or_buf='../Resu/'+fver+'_'+featsel+str(int(time.time()))+'.csv',date_format="%Y-%m-%d",index = False)

        if 'mlp' in modeltype[0]:
            print('ALL_____________mlp___________________________________________________________________________')
            featsel = 'all_MLP' if testone == False else 'temp_all_MLP'
            mlp_resdf = ExamineMLP(orygframe,Xintex,X_train_sc, y_train,X_test_sc, y_test,featurenames,atr,testone=testone,plot=False)
            mlp_resdf.to_csv(sep=';',path_or_buf='../Resu/'+fver+'_'+featsel+str(int(time.time()))+'.csv',date_format="%Y-%m-%d",index = False)
        
        if 'nn' in modeltype[0]:
            print('ALL_____________nn____________________________________________________________________________')
            featsel = 'all_NN' if testone == False else 'temp_all_NN'
            nn_resdf,model = ExamineNN(orygframe, Xintex, masterframe, featurenames,atr,testone=testone, plot=False)
            nn_resdf.to_csv(sep=';',path_or_buf='../Resu/'+fver+'_'+featsel+str(int(time.time()))+'.csv',date_format="%Y-%m-%d",index = False)
            
            
        print('ALL________________finished________________________________________________________________________________')
    
    return orygframe, masterframe, model

def calculate_atr(prices):

    resdf = pd.DataFrame(index=prices.index)
    resdf0 = pd.DataFrame(index=prices.index)
    resdf0['tr1'] = prices['high'] - prices['low']
    resdf0['tr2'] = abs (prices['high'] - prices['close'].shift())
    resdf0['tr3'] = abs (prices['low'] - prices['close'].shift())
    resdf['tr'] = resdf0.max(axis=1)
    atr = resdf.tr.mean()
    return atr

def prepare_y1(masterframe, atr):
    
    # Prepare Y
    Rtp=1
    Rsl=1
    masterframe['y'] = -1
#     n = 5  #distanse infinity
    i = 0
    while i < len(masterframe) - 1:   
        j = 1
        yy = False
        while j <= len(masterframe) - i - 1:
            #if (masterframe.low.iloc[i+j] < masterframe.close.iloc[i]-Rsl*atr):
            if (masterframe.low.iloc[i+j] < masterframe.low.iloc[i]-Rsl*atr):
                yy = False
                break
            #if (masterframe.high.iloc[i+j] > masterframe.close.iloc[i]+Rtp*atr):
            if (masterframe.high.iloc[i+j] > masterframe.high.iloc[i]+Rtp*atr):
                yy = True
                break
            j = j + 1

        if yy == True:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 1            
            #masterframe.iloc[i+1:i+j+1,masterframe.columns.get_loc('y')]=0      #nochain
            #i = i + j                                                           #nochain 
            i = i + 1   #chain

        else:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 0
            i = i + 1
    return

def prepare_y2(masterframe, atr):    
    # Prepare Y
    Rtp=1
    Rsl=1
    masterframe['y'] = -1
    n = 5  # distanse 5
    i = 0
    while i < len(masterframe) - n:   
        j = 1
        yy = False
        while j <= n:
            #if (masterframe.low.iloc[i+j] < masterframe.close.iloc[i]-Rsl*atr):
            if (masterframe.low.iloc[i+j] < masterframe.low.iloc[i]-Rsl*atr):
                yy = False
                break
            #if (masterframe.high.iloc[i+j] > masterframe.close.iloc[i]+Rtp*atr):
            if (masterframe.high.iloc[i+j] > masterframe.high.iloc[i]+Rtp*atr):
                yy = True
                break
            j = j + 1

        if yy == True:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 1            
            #masterframe.iloc[i+1:i+j+1,masterframe.columns.get_loc('y')]=0      #nochain
            #i = i + j                                                           #nochain 
            i = i + 1   #chain

        else:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 0
            i = i + 1
    return
    
def prepare_y3(masterframe, atr):      
    # Prepare Y
    Rtp=0
    Rsl=0
    masterframe['y'] = -1
    n = 1 # distanse 1
    i = 0
    while i < len(masterframe) - n:   
        j = 1
        yy = False
        while j <= n:
            if (masterframe.high.iloc[i+j] > masterframe.high.iloc[i]):
                yy = True
                break
            j = j + 1

        if yy == True:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 1            
            #masterframe.iloc[i+1:i+j+1,masterframe.columns.get_loc('y')]=0      #nochain
            #i = i + j                                                           #nochain 
            i = i + 1   #chain

        else:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 0
            i = i + 1
    return

def prepare_y4(masterframe, atr):      
    # Prepare Y
    Rtp=1
    masterframe['y'] = -1
    n = 1  # distanse
    i = 0
    while i < len(masterframe) - n:   
        j = 1
        yy = False
        while j <= n:
            if (masterframe.high.iloc[i+j] > masterframe.high.iloc[i]+Rtp*atr):
                yy = True
                break
            j = j + 1

        if yy == True:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 1            
            #masterframe.iloc[i+1:i+j+1,masterframe.columns.get_loc('y')]=0      #nochain
            #i = i + j                                                           #nochain 
            i = i + 1   #chain

        else:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 0
            i = i + 1
    return

# close>high_1atr dist1 
def prepare_y_(masterframe, atr):      
    # Prepare Y
    Rtp=1
    masterframe['y'] = -1
    n = 1  # distanse
    i = 0
    while i < len(masterframe) - n:   
        j = 1
        yy = False
        while j <= n:
            if (masterframe.close.iloc[i+j] > masterframe.high.iloc[i]+Rtp*atr):
                yy = True
                break
            j = j + 1

        if yy == True:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 1            
            #masterframe.iloc[i+1:i+j+1,masterframe.columns.get_loc('y')]=0      #nochain
            #i = i + j                                                           #nochain 
            i = i + 1   #chain

        else:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 0
            i = i + 1
    return

def prepare_y(masterframe, atr,dist = 1,Rtp = 1,mode='c'):      
    # Rtp krotnosc atr
    # dist - distanse
    # mode h - high; każde inne to c (close)
    
    masterframe['y'] = -1
    i = 0
    while i < len(masterframe) - dist:   
        j = 1
        yy = False
        while j <= dist:
            if (mode=='h'):
                ref = masterframe.high.iloc[i+j]
            else:
                ref = masterframe.close.iloc[i+j]
            if (ref > masterframe.high.iloc[i]+Rtp*atr):
                yy = True
                break
            j = j + 1

        if yy == True:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 1            
            #masterframe.iloc[i+1:i+j+1,masterframe.columns.get_loc('y')]=0      #nochain
            #i = i + j                                                           #nochain 
            i = i + 1   #chain

        else:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 0
            i = i + 1
    return



def dumpdatawithy(datafile,dist = 1,Rtp = 1,mode = 'c'):

    masterframe = loaddata_master('../Data/'+datafile+'.csv')

    atr = calculate_atr(masterframe)

    prepare_y(masterframe,atr,dist,Rtp,mode)
    
    masterframe = masterframe.drop(['id'],1)
    
    masterframe.to_csv(sep=';',path_or_buf='../Data/y_'+str(dist)+'_'+str(Rtp)+'_'+mode+'_'+datafile+'.csv',date_format="%Y-%m-%d",index = False,na_rep='')
    
    return