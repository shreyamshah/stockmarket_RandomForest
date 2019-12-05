import pandas as pd
dataset = pd.read_csv('500209.csv')
print('Preprocessing starts')
#dataset = dataset.iloc[:,1:]
dataset = dataset.iloc[::-1]
close = dataset.iloc[:,3].values
dataset.drop('Close Price', axis=1)
low = dataset.iloc[:,2].values
high = dataset.iloc[:,1].values
momentum = [0] * len(close)
change = [0] * len(close)
roc1 = [0] * len(close)
roc2 = [0] * len(close)
wr5 = [0] * len(close)
ma5 = [0] * len(close)
ma10 = [0] * len(close)
d5 = [0] * len(close)
d10 = [0] * len(close)
oscp = [0] * len(close)
for i in range(1,len(close)) :
    momentum[i] =  1 if close[i]>close[i-1] else 0
    change[i] = (abs(close[i]-close[i-1]))/close[i-1]
    roc1[i] = (close[i]/close[i-1])*100

for i in range(2,len(close)) :
    roc2[i] = (close[i]/close[i-2])*100
for i in range(5,len(close)) :
    m = max(close[i-5:i])
    l = min(close[i-5:i])
    wr5[i] = ((m-close[i])/(m-l))*100
    ma5[i] = sum(close[i-5:i])/5
    #ma10[i] = sum(close[i-10:i])/10
for i in range(1,len(close)):
    ma10[i] = sum(close[i-10:i])/10
dataset['Momentum'] = momentum
dataset['Change'] = change
dataset['ROC1'] = roc1
dataset['ROC2'] = roc2
dataset['WR5'] = wr5
dataset['MA5'] = ma5
dataset['MA10'] = ma10
dataset.to_csv('infosys.csv',index=False)
print('Preprocessing Complete')