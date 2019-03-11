#%%
import os
os.chdir(r'D:\Learningfile\DC竞赛\2018甜橙杯数据建模大赛\开放数据_甜橙杯数据建模_中国电信（补充）')
import pandas as pd
import numpy as np
import re
import geohash

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold,StratifiedKFold
from lightgbm import LGBMClassifier
import lightgbm as lgb
from bayes_opt import BayesianOptimization

import shap
shap.initjs()

#%%
train_tran = pd.read_csv('transaction_train_new.csv')
train_oper=pd.read_csv('operation_train_new.csv')
train_label=pd.read_csv('tag_train_new.csv')

test_tran=pd.read_csv('transaction_round1_new.csv')
test_oper=pd.read_csv('operation_round1_new.csv')
test_label=pd.read_csv('sub.csv')
#%%
train_tran['mode'] = 'transaction'
train_df = train_oper.append(train_tran).reset_index(drop=True)
train_df = train_df.sort_values(by=['UID', 'day', 'time'], ascending=[True, True, True])


test_tran['mode'] = 'transaction'
test_df = test_oper.append(test_tran).reset_index(drop=True)
test_df = test_df.sort_values(by=['UID', 'day', 'time'], ascending=[True, True, True])

#%%
def one_hot_encoder(df, feat, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = feat
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

#%%
def featuring(data):
    data_nuniq = pd.DataFrame()
    for col in data.columns:
        data_nuniq['nuniq_' + col] = data.groupby('UID')[col].nunique()
    #%%    
    data['null']=data.isnull().sum(axis=1)/data.shape[1]
    #%%
    data['time2'] = data['time'].apply(lambda x:int(x[:2]))
    data['time2'] = data['time2'].map(lambda x:0 if x>9 and x<18 else 1)
    data['day2']=pd.cut(data['day'],[0,7,14,21,31],labels=[0,1,2,3])
    #%%
    data['device1']=data['device1'].fillna(-999).apply(lambda x: 'other' if x not in [-999,'09baf2f39bc3dc86','49dd36968dbfadda','d2cf44cec09806cc','630a1adff2a87007'] else x)
    #%%
    data['device2']=data['device2'].map(lambda x: re.sub(re.compile('IPHONE\s*\d*\w*|IPOD\s\w*\s\d\w*'),'IPHONE',str(x)))
    data['device2']=data['device2'].map(lambda x: re.sub(re.compile('OPPO\s*\w*|R7\w*\d*|PRO\s*\d*\s*\w*|PA\wM00'),'OPPO',str(x)))
    data['device2']=data['device2'].map(lambda x: re.sub(re.compile('VIVO\s*\w*'),'VIVO',str(x)))
    data['device2']=data['device2'].map(lambda x: re.sub(re.compile('\w*MI\s*\d*\w*|M\d*\s*\w*|REDMI|MI-\d*\w*'),'MI',str(x)))
    data['device2']=data['device2'].map(lambda x: re.sub(re.compile('(HUAWEI\s*\w*\d*)|(\w*\-\w+L\d*|(HUAWEI-\w*\d*)|(h\d+-\w+\d+))'),'HUAWEI',str(x)))
    data['device2']=data['device2'].replace('nan',-999).map(lambda x: 'other' if x not in ['IPHONE','OPPO','VIVO','MI','HUAWEI',-999] else x)
    #%%
    data['channel']=data['channel'].fillna(-999).apply(lambda x: 'other' if x not in [-999,140.0] else x)
    #%%
    data['amt_src1']=data['amt_src1'].fillna(-999).apply(lambda x: 'other' if x not in [-999,'c5fc631370cabc0d','155c9e1c32bd0fa2','4d7831c6f695ab19','f29829bc82459191'] else x)
    #%%
    data['amt_src2']=data['amt_src2'].fillna(-999).apply(lambda x: 'other' if x not in [-999,'9fefed0a981dcb7a','9a8ee16bde15e38a','cf6e3a074407c379','a2aa73cdb6621133'] else x)
    #%%
    data['trans_type1']=data['trans_type1'].fillna(-999).apply(lambda x: 'other' if x != 'c2f2023d279665b2' else x)
    #%%
    data['trans_type2']=data['trans_type2'].fillna(-999).apply(lambda x: 'other' if x not in [-999,105.0,102.0] else x)
    #%%
    data['mac1']=data['mac1'].fillna(-999).apply(lambda x: 'other' if x not in [-999,'a8dc52f65085212e'] else x)
    #%%
    data['mac2']=data['mac2'].fillna(-999).apply(lambda x: 'other' if x not in [-999,'a8dc52f65085212e'] else x)
    #%%
    data['merchant_ill'] = data['merchant'].apply(lambda x: 1 if x in ['5776870b5747e14e' ,'8b3f74a1391b5427' ,'0e90f47392008def' ,'6d55ccc689b910ee' ,'2260d61b622795fb' ,'1f72814f76a984fa' ,'c2e87787a76836e0' ,'4bca6018239c6201' ,'922720f3827ccef8' ,'2b2e7046145d9517' ,'09f911b8dc5dfc32' ,'7cc961258f4dce9c' ,'bc0213f01c5023ac' ,'0316dca8cc63cc17' ,'c988e79f00cc2dc0' ,'d0b1218bae116267' ,'72fac912326004ee' ,'00159b7cc2f1dfc8' ,'49ec5883ba0c1b0e' ,'c9c29fc3d44a1d7b' ,'33ce9c3877281764' ,'e7c929127cdefadb' ,'05bc3e22c112c8c9' ,'5cf4f55246093ccf' ,'6704d8d8d5965303' ,'4df1708c5827264d' ,'6e8b399ffe2d1e80' ,'f65104453e0b1d10' ,'1733ddb502eb3923' ,'a086f47f681ad851' ,'1d4372ca8a38cd1f' ,'29db08e2284ea103' ,'4e286438d39a6bd4' ,'54cb3985d0380ca4' ,'6b64437be7590eb0' ,'89eb97474a6cb3c6' ,'95d506c0e49a492c' ,'c17b47056178e2bb' ,'d36b25a74285bebb'] else 0)
    #%%
    #%%
    data['version']=data['version'].fillna(-999).apply(lambda x: 'other' if x not in [-999,'7.0.9','7.0.5'] else x)
    #%%
    data['os']=data['os'].fillna(-999).apply(lambda x: 'other' if x not in [-999,102.0,200.0,103.0] else x)

    #%%
    data['longitude'] = data['geo_code'].apply(lambda x: geohash.decode(x)[0] if isinstance(x, str) else np.nan).astype(float)
    data['latitude'] = data['geo_code'].apply(lambda x: geohash.decode(x)[1] if isinstance(x, str) else np.nan).astype(float)
    #%%
    feat_cat=['time2','day2','device1','device2','channel','amt_src1','amt_src2','trans_type1','trans_type2','market_type','mac1','mac2','version','os']

    feat_null=['merchant','code1', 'code2','acc_id1', 'device_code1','device_code2', 'device_code3', 
            'geo_code','ip1', 'acc_id2', 'acc_id3', 'market_code', 'ip1_sub']
    #%%
    for j in feat_null:
        data[j]=data[j].isnull().astype(int)

    #%%
    data,cat_cols = one_hot_encoder(data,feat_cat,nan_as_category= True)
    #%%
    aggregations = {'trans_amt': ['min', 'max', 'mean', 'var','size','sum'],
                    'null': ['min', 'max', 'mean', 'var'],
                    'bal': ['min', 'max', 'mean', 'var'],
                    'merchant_ill':['min', 'max', 'mean', 'var'],
                    'longitude':['min', 'max', 'mean', 'var'],
                    'latitude':['min', 'max', 'mean', 'var']}
    for cat in cat_cols:
        aggregations[cat] = ['min', 'max', 'mean', 'var','sum']

    for cat in feat_null:
        aggregations[cat] = ['min','max','mean','sum']    
    
    #%%    
    data_agg = data.groupby('UID').agg(aggregations)
    data_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in data_agg.columns.tolist()])
    #%%
    data_agg['trans_bal_mean2'] = data[data['mode']=='transaction'].sort_values(['UID','day','time'], ascending=[True,False,False]).drop_duplicates(subset=['UID','day'],keep='first')[['UID','bal']].groupby('UID').mean()
    data_agg['trans_transamt/balmean2']=data_agg['trans_amt_SUM']/data_agg['trans_bal_mean2']

    data_df = data_agg.merge(data_nuniq,on='UID')
    
    return data_df

#%%
train = featuring(train_df)
train = train_label.merge(train, on='UID')
test = featuring(test_df)
test = test_label.merge(test, on='UID')
print(f'Gen train shape: {train.shape}, test shape: {test.shape}')
#%%
feats = [i for i in train.columns if i in test.columns and i not in ['UID','Tag']]
