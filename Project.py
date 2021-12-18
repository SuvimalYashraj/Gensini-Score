import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score,mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings("ignore") 

data = pd.read_csv('Dataset1.csv')
data.drop('SYNTAX', axis=1, inplace=True)
#print(data)
#data.groupby(['GROUP'])

# set the index to be this and don't drop
#data.set_index(keys=['GROUP'], drop=False,inplace=True)
# get a list of names
#names=data['GROUP'].unique().tolist()
#
arr=np.array(data).T
cormatrix = np.corrcoef(arr)
for i in range(0,15):
    for j in range(0,15):
        if (cormatrix[i][j]>0.7 or cormatrix[i][j]<-0.7) and (i!=j) and (j>=i):
            print(cormatrix[i][j],data.columns.values[i],data.columns.values[j])
#print("########################>>>>>>>>>>>>>>>>",cormatrix[12][14],data.columns.values[12],data.columns.values[14])

#print(cormatrix[:][13][0:14])
#plt.bar(range(0,14), cormatrix[:][13][0:14], align='center')
#plt.xticks(range(0,14), data.columns.values[0:13])
#plt.show()
#print(cormatrix)
#print(cormatrix[:][3][0:14])
#plt.bar(range(0,15), cormatrix[:][3][0:14], align='center')
#plt.xticks(range(0,15), data.columns.values[0:15])
#plt.show()

# now we can perform a lookup on a 'view' of the dataframe
df1 = data.loc[data.GROUP==1]
df2 = data.loc[data.GROUP==2]
df3 = data.loc[data.GROUP==3]
df4 = data.loc[data.GROUP==4]
# now you can query all 'joes'
#df3.reset_index(level=['GROUP'])
df1.drop('GROUP', axis=1, inplace=True)
df2.drop('GROUP', axis=1, inplace=True)
df3.drop('GROUP', axis=1, inplace=True)
df4.drop('GROUP', axis=1, inplace=True)

#Fill missing values with average of the group
mask = np.array(df1['FI'].notnull())
tofill= mean(df1['FI'][mask])
df1['FI'][2]=tofill
df1['FI'][25]=tofill
print("dsfsdfsdfsadfas",tofill)
data['FI'][2]=tofill
data['FI'][25]=tofill

mask = np.array(df4['FI'].notnull())
tofill= mean(df4['FI'][mask])
df4['FI'][83]=tofill
print("dsfsdfsdfsadfas",tofill)
data['FI'][83]=tofill

#Feature importance using Random Forests(Entire Normalized Dataset including Group numbers)

df_array=np.array(data);
k=preprocessing.MinMaxScaler(feature_range=(-1,1))
df_array = k.fit_transform(df_array)
data=pd.DataFrame(data=df_array, columns=data.columns.values)
train= data.values
A = train[:,14]
train = train[:,0:14]
model = LinearRegression()
model.fit(train,A)
print(model.coef_)
#rfe = RFE(model, verbose=True)
#fit = rfe.fit(train, A)
#print("Num Features:", fit.n_features_)
#print("Selected Features:", fit.support_)
#
#model.fit(train, A)
#print(model.feature_importances_)

###############################################################################
arr=np.array(df1).T
cormatrix = np.corrcoef(arr)
for i in range(0,14):
    for j in range(0,14):
        if (cormatrix[i][j]>0.8 or cormatrix[i][j]<-0.8) and (i!=j) and (j>=i):
            print(cormatrix[i][j],df1.columns.values[i],df1.columns.values[j])           
plt.bar(range(0,13), cormatrix[:][13][0:13], align='center')
plt.xticks(range(0,13), df1.columns.values[0:13])
plt.show()

print(">")

arr1=np.array(df2).T
cormatrix1 = np.corrcoef(arr1)
for i in range(0,14):
    for j in range(0,14):
        if (cormatrix1[i][j]>0.8 or cormatrix1[i][j]<-0.8) and (i!=j) and (j>=i):
            print(cormatrix1[i][j],df3.columns.values[i],df3.columns.values[j])
plt.bar(range(0,13), cormatrix[:][13][0:13], align='center')
plt.xticks(range(0,13), df1.columns.values[0:13])
plt.show()

print(">")

arr=np.array(df3).T
cormatrix = np.corrcoef(arr)
for i in range(0,14):
    for j in range(0,14):
        if (cormatrix[i][j]>0.8 or cormatrix[i][j]<-0.8) and (i!=j) and (j>=i):
            print(cormatrix[i][j],df3.columns.values[i],df3.columns.values[j])
plt.bar(range(0,13), cormatrix[:][13][0:13], align='center')
plt.xticks(range(0,13), df1.columns.values[0:13])
plt.show()

print(">")

arr=np.array(df4).T
cormatrix = np.corrcoef(arr)
for i in range(0,14):
    for j in range(0,14):
        if (cormatrix[i][j]>0.8 or cormatrix[i][j]<-0.8) and (i!=j) and (j>=i):
            print(cormatrix[i][j],df3.columns.values[i],df3.columns.values[j])
plt.bar(range(0,13), cormatrix[:][13][0:13], align='center')
plt.xticks(range(0,13), df1.columns.values[0:13])
plt.show()
###############################################################################


from sklearn.cross_validation import train_test_split
df_array=np.array(df1);
k=preprocessing.MinMaxScaler(feature_range=(-1,1))
df_array = k.fit_transform(df_array)
df1=pd.DataFrame(data=df_array, columns=df1.columns.values)
train, test = train_test_split(df1, train_size = 0.8)

A = train.GENSINI
train.drop('GENSINI', axis=1, inplace=True)
model=LinearRegression();
rfe = RFE(model)
fit = rfe.fit(train, A)
print("Num Features:", fit.n_features_)
print("Selected Features:", fit.support_)
for mo in range(0,13):
    print(fit.ranking_[mo],train.columns.values[mo])
print("Feature Ranking:", fit.ranking_)

B = test.GENSINI
test.drop('GENSINI', axis=1, inplace=True)

y_pred = rfe.predict(test)
print("AAAAAAAAAAAAAAAA>>>>>",mean_squared_error(B, y_pred))
print(rfe._get_param_names)

#model.fit(df1,A);
#print("##################################")
#print(model.coef_)
#y_pred = model.predict(df1)
#print("AAAAAAAAAAAAAAAA>>>>>",explained_variance_score(A, y_pred))

df_array=np.array(df2);
k=preprocessing.MinMaxScaler(feature_range=(-1,1))
df_array = k.fit_transform(df_array)
df2=pd.DataFrame(data=df_array, columns=df3.columns.values)
df2.to_csv('second.csv')
train, test = train_test_split(df2, train_size = 0.8)

A = train.GENSINI
train.drop('GENSINI', axis=1, inplace=True)
model=LinearRegression();
rfe = RFE(model)
fit = rfe.fit(train, A)
print("Num Features:", fit.n_features_)
print("Selected Features:", fit.support_)
for mo in range(0,13):
    print(fit.ranking_[mo],train.columns.values[mo])
print("Feature Ranking:", fit.ranking_)

B = test.GENSINI
test.drop('GENSINI', axis=1, inplace=True)

y_pred = rfe.predict(test)
print("AAAAAAAAAAAAAAAA>>>>>",mean_squared_error(B, y_pred))
print(rfe._get_param_names)



#model.fit(df2,A);
#print("##################################")
#print(model.coef_)
#y_pred = model.predict(df2)
#print("AAAAAAAAAAAAAAAA>>>>>",explained_variance_score(A, y_pred))

df_array=np.array(df3);
k=preprocessing.MinMaxScaler(feature_range=(-1,1))
df_array = k.fit_transform(df_array)
df3=pd.DataFrame(data=df_array, columns=df3.columns.values)
train, test = train_test_split(df3, train_size = 0.8)

A = train.GENSINI
train.drop('GENSINI', axis=1, inplace=True)
model=LinearRegression();
rfe = RFE(model)
fit = rfe.fit(train, A)
print("Num Features:", fit.n_features_)
print("Selected Features:", fit.support_)
for mo in range(0,13):
    print(fit.ranking_[mo],train.columns.values[mo])
print("Feature Ranking:", fit.ranking_)

B = test.GENSINI
test.drop('GENSINI', axis=1, inplace=True)

y_pred = rfe.predict(test)

print("AAAAAAAAAAAAAAAA>>>>>",mean_squared_error(B, y_pred))
print(rfe._get_param_names)

#model.fit(df3,A);
#print("##################################")
#print(model.coef_)
#y_pred = model.predict(df3)
#print("AAAAAAAAAAAAAAAA>>>>>",explained_variance_score(A, y_pred))


df_array=np.array(df4);
k=preprocessing.MinMaxScaler(feature_range=(-1,1))
df_array = k.fit_transform(df_array)
train, test = train_test_split(df4, train_size = 0.8)

A = train.GENSINI
train.drop('GENSINI', axis=1, inplace=True)
model=LinearRegression();
rfe = RFE(model)
fit = rfe.fit(train, A)
print("Num Features:", fit.n_features_)
print("Selected Features:", fit.support_)
for mo in range(0,13):
    print(fit.ranking_[mo],train.columns.values[mo])
print("Feature Ranking:", fit.ranking_)

B = test.GENSINI
test.drop('GENSINI', axis=1, inplace=True)

y_pred = rfe.predict(test)

print("AAAAAAAAAAAAAAAA>>>>>",mean_squared_error(B, y_pred))
print(rfe._get_param_names)

#model=LinearRegression();
#model.fit(df4,A);
#print("##################################")
#print(model.coef_)
#y_pred = model.predict(df4)
#print("AAAAAAAAAAAAAAAA>>>>>",mean_squared_error(A, y_pred))