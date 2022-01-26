# Data preprocessing

import numpy as np
import pandas as pd
from scipy.linalg import hankel

#constants
setname='D1' #D2
n_cats=[63,60,51,29,13] #vused categories
NCATS = len(n_cats) 
L_win = 4 # time window for training
datadir='data/'
#Aggregated categories
cdf=pd.read_csv(datadir+'mcc2big.csv')
cdf=cdf.groupby('0').apply(lambda x: np.array(x)[:,0])
mcclist=np.concatenate(cdf[n_cats].values)
print('Proceeding '+setname)
if setname=='D1':
    df = pd.read_csv(datadir+'train_set.csv', usecols=['customer_id', 'transaction_date', 'mcc', 'amount'])
else:
    df = pd.read_csv(datadir+'D2row_data.csv', usecols=['customer_id', 'transaction_date', 'mcc', 'amount'])
df = df[df.mcc.isin(mcclist)]
df['MCC87']=np.zeros(len(df)).astype(int)
for c in n_cats:
    df.iloc[df.mcc.isin(cdf[c]),4]=c
df['WEEK'] = pd.to_datetime(df['transaction_date']).dt.isocalendar().week

table=df.groupby(['customer_id', 'MCC87', 'WEEK'], as_index = False)['amount'].sum()
table['COUNT']=df.groupby(['customer_id', 'MCC87', 'WEEK']).size().values
labels, uniques = pd.factorize(table['customer_id'])
table['id'] = labels
table = table.pivot_table(index = ['id','WEEK'], columns = 'MCC87', values = 'COUNT', fill_value = 0).reset_index()
if setname=='D2': #If New Year included... For new data with COVID-19
    w=table.WEEK.values
    q=np.where(w<=40, w+12, w-40)
    table.WEEK=q

N_weeks=table.WEEK.max() - table.WEEK.min()
print('Features aggregated. \nAdding missed...')
b=np.arange(table.WEEK.min(), table.WEEK.max()+1)
for user in table.id.unique():
    if user%500 == 0:
        print(user//500, end='-')
    if len(table[table.id==user])<len(b):
        w=np.in1d(b,table[table.id==user].WEEK.values, invert=True)
        table=pd.concat((table, pd.DataFrame({'id':[user]*len(b[w]), 'WEEK':b[w]}))).fillna(value=0)
table.WEEK=table.WEEK-table.WEEK.min()+1
table=table.sort_values(by=['id', 'WEEK']).reset_index().drop('index',axis=1).astype(int)
bad=np.in1d(table.id.values, np.where(table.groupby('id')[n_cats].sum().sum(axis=1)<10)[0])
table=table.drop(table[bad].index)
labels, uniques = pd.factorize(table['id'])
table['id']=labels
table=table.sort_values(by=['id', 'WEEK']).reset_index().drop('index',axis=1).astype(int)

n_cats=[63]#,'60','51','29','13']
table=table.drop(table[table.WEEK<7].index)
table.WEEK=table.WEEK-table.WEEK.min()+1
N_weeks=table.WEEK.max()-table.WEEK.min()+1
NCATS = table.shape[1] - 2
bad=np.in1d(table.id.values, np.where(table.groupby('id')[63].sum()<1))
table=table.drop(table[bad].index)
labels, uniques = pd.factorize(table['id'])
table['id']=labels
table=table.sort_values(by=['id', 'WEEK']).reset_index().drop('index',axis=1).astype(int)

table.to_csv(datadir+setname+'table.csv', index=False)
# indices for train/test subsets

print('\n',datadir+setname+'table.csv saved. Counting idices...')
ttab=pd.DataFrame(columns=['id', 'last_week']+list(np.arange(L_win+1)))
for user in table.id.unique():
    if user%500 == 0:
        print(user//500, end='-')
    ind=table[table.id==user].index
    t2=pd.DataFrame({'id':[user]*(N_weeks-L_win), 'last_week':np.arange(L_win+1,N_weeks+1)}) 
    t2[list(np.arange(L_win+1))]=hankel(ind)[:N_weeks-L_win,:L_win+1]
    ttab=pd.concat((ttab,t2))
ttab=ttab.astype(int)
ttab.to_csv(datadir+setname+'indtab.csv', index=False)
print('\n',datadir+setname+'indtab.csv saved. \n\tPreprocessing done.')
