# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 11:40:18 2018

@author: Administrator
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import mode
import seaborn as sns
import matplotlib.pylab as plt
os.chdir('G:\Python27\work\online_pay_data')

data=pd.read_table("user_data.txt",encoding='gb2312')

##########将变量转换成数值型
data['lable']=data['user_pay_type']
data['lable']=data['lable'].map( {'cod':0, 'online':1} ).astype(int)

country_name=['RUH','JED','DHA','United Arab Emirates','Kuwait','Qatar','Bahrain','Oman','Jordan','Lebanon']
data=data[data['country_region_name'].isin(country_name)]
data['country_change']=data['country_region_name']
data['country_change']=data['country_change'].map( {'RUH':1,'JED':2,'DHA':3,'United Arab Emirates':4,'Kuwait':5,'Qatar':6,'Bahrain':7,'Oman':8,'Jordan':9,'Lebanon':10} ).astype(int)

data['work']=data['is_work']
data['work']=data['work'].map({'business':0,'un_business':1})

data['end_order']=data['is_end_order'x]
data['end_order']=data['end_order'].map({u'有终结':0,u'无终结':1})

data_change=data
data_change=data_change.drop(['user_id','is_work','country_region_name','user_pay_type','is_end_order'],axis=1)

###############构建新的变量
data_change['pay_rate']=data_change['pay_num']/data_change['add_num']
data_change['ship_rate']=data_change['ship_num']/data_change['pay_num']
data_change['reject_rate']=data_change['cod_reject_num']/data_change['add_num']
data_change['returned_rate']=data_change['returned_num']/data_change['pay_num']
data_change['rec_rate']=data_change['rec_num']/data_change['pay_num']
data_change['complaints_rate']=data_change['complaints_num']/data_change['pay_num']
data_change['address_problem_order_rate']=data_change['address_problem_order']/data_change['pay_num']
data_change['no_answer_problem_order_rate']=data_change['no_answer_problem_order']/data_change['pay_num']
data_change['discount_rate']=data_change['discount_num']/data_change['pay_num']
data_change['month_end_prop']=data_change['month_end_num']/(data_change['month_end_num']+data_change['month_unend_num'])
data_change['day_moring_prop']=data_change['day_moring_num']/(data_change['day_moring_num']+data_change['day_afternoon_num']+data_change['day_other_num'])
data_change['day_afternoon_prop']=data_change['day_afternoon_num']/(data_change['day_moring_num']+data_change['day_afternoon_num']+data_change['day_other_num'])

##############特征筛选构建
data_final_feature=data_change[['lable','sex']]
data_final_feature['sex']=data_change['sex'].map( {0:0, 1:1, 2:0} ).astype(int)

data_change['is_google_change']=data_change['is_google'].replace([3,4],1)
data_dummies=pd.get_dummies(data_change['is_google_change'],prefix = 'is_google')
data_final_feature = data_final_feature.join(data_dummies[['is_google_0','is_google_1']])

data_final_feature['register_from']=data_change['register_from'].replace([1,2,3],1)
data_final_feature['app_type']=data_change['app_type'].replace([3,4,5,6],0)

data_change['language_site']=data_change['language_site'].fillna(-1)
data_change['language_site_change']=data_change['language_site'].replace([3,5,6,8,9,12],2)
data_dummies=pd.get_dummies(data_change['language_site_change'],prefix = 'language_site_change')
data_final_feature = data_final_feature.join(data_dummies['language_site_change_0'])

data_dummies=pd.get_dummies(data_change['country_change'],prefix = 'country_change')
data_final_feature = data_final_feature.join(data_dummies[['country_change_2','country_change_4','country_change_7','country_change_8']])

data_final_feature['end_order']=data_change['end_order']

data_change['ship_num_change']=data_change['ship_num']
data_change.loc[(data_change['ship_num_change']>0)*(data_change['ship_num_change']<=4),'ship_num_change']=1
data_change.loc[data_change['ship_num_change']>4,'ship_num_change']=2
data_final_feature['ship_num']=data_change['ship_num_change']

data_change['returned_num_change']=data_change['returned_num']
data_change.loc[(data_change['returned_num_change']>0)*(data_change['returned_num_change']<=8),'returned_num_change']=1
data_change.loc[data_change['returned_num_change']>8,'returned_num_change']=2
data_dummies=pd.get_dummies(data_change['returned_num_change'],prefix = 'returned_num_change')
data_final_feature = data_final_feature.join(data_dummies['returned_num_change_0'])

data_change['rec_num_change']=data_change['rec_num']
data_change.loc[(data_change['rec_num_change']>0)*(data_change['rec_num_change']<=1),'rec_num_change']=1
data_change.loc[data_change['rec_num_change']>1,'rec_num_change']=2
data_dummies=pd.get_dummies(data_change['rec_num_change'],prefix = 'rec_num_change')
data_final_feature = data_final_feature.join(data_dummies[['rec_num_change_0','rec_num_change_2']])

data_change['address_problem_order']=data_change['address_problem_order'].fillna(-1)
data_change['address_problem_order_change']=data_change['address_problem_order']
data_change.loc[data_change['address_problem_order_change']>-1,'address_problem_order_change']=0
data_final_feature['address_problem_order_change']=data_change['address_problem_order_change']

data_change['no_answer_problem_order']=data_change['no_answer_problem_order'].fillna(-1)
data_change['no_answer_problem_order_change']=data_change['no_answer_problem_order']
data_change.loc[(data_change['no_answer_problem_order_change']>-1)*(data_change['no_answer_problem_order_change']<=5),'no_answer_problem_order_change']=0
data_change.loc[data_change['no_answer_problem_order_change']>5,'no_answer_problem_order_change']=1
data_dummies=pd.get_dummies(data_change['no_answer_problem_order_change'],prefix = 'no_answer_problem_order_change')
data_final_feature = data_final_feature.join(data_dummies['no_answer_problem_order_change_-1.0'])

data_change['day_afternoon_num_change']=data_change['day_afternoon_num']
data_change.loc[data_change['day_afternoon_num_change']<=0,'day_afternoon_num_change']=0
data_change.loc[data_change['day_afternoon_num_change']>0,'day_afternoon_num_change']=1
data_final_feature['day_afternoon_num_change']=data_change['day_afternoon_num_change']

data_change['customer_service_time']=data_change['customer_service_time'].fillna(-1)
data_change.loc[data_change['customer_service_time']<-1,'customer_service_time']=-2
data_change['customer_service_time']=data_change['customer_service_time'].apply(lambda x: np.round(x))
data_change['customer_service_time_change']=data_change['customer_service_time']
data_change.loc[(data_change['customer_service_time_change']<=10)*(data_change['customer_service_time_change']>=0),'customer_service_time_change']=0
data_change.loc[data_change['customer_service_time_change']>10,'customer_service_time_change']=1
data_dummies=pd.get_dummies(data_change['customer_service_time_change'],prefix = 'customer_service_time_change')
data_final_feature = data_final_feature.join(data_dummies[['customer_service_time_change_-1.0','customer_service_time_change_0.0','customer_service_time_change_1.0']])

data_change['ship_time']=data_change['ship_time'].fillna(-1)
data_change['ship_time']=data_change['ship_time'].apply(lambda x: np.round(x))
data_change['ship_time_change']=data_change['ship_time']
data_change.loc[data_change['ship_time_change']>=1,'ship_time_change']=1
data_dummies=pd.get_dummies(data_change['ship_time_change'],prefix = 'ship_time_change')
data_final_feature = data_final_feature.join(data_dummies[['ship_time_change_-1.0','ship_time_change_0.0']])

data_change['transport_time']=data_change['transport_time'].fillna(-1)
data_change['transport_time']=data_change['transport_time'].apply(lambda x: np.round(x))
data_change['transport_time_change']=data_change['transport_time']
data_change.loc[(data_change['transport_time_change']>=2)*(data_change['transport_time_change']<5),'transport_time_change']=2
data_change.loc[data_change['transport_time_change']>=5,'transport_time_change']=3
data_dummies=pd.get_dummies(data_change['transport_time_change'],prefix = 'transport_time_change')
data_final_feature = data_final_feature.join(data_dummies[['transport_time_change_-1.0','transport_time_change_3.0']])

data_change['deliver_time']=data_change['deliver_time'].fillna(-1)
data_change['deliver_time']=data_change['deliver_time'].apply(lambda x: np.round(x))
data_change['deliver_time_change']=data_change['deliver_time']
data_change.loc[(data_change['deliver_time_change']>=1)*(data_change['deliver_time_change']<4),'deliver_time_change']=1
data_change.loc[data_change['deliver_time_change']>=4,'deliver_time_change']=2
data_dummies=pd.get_dummies(data_change['deliver_time_change'],prefix = 'deliver_time_change')
data_final_feature = data_final_feature.join(data_dummies[['deliver_time_change_-1.0','deliver_time_change_0.0','deliver_time_change_1.0']])

data_change['ship_rate']=data_change['ship_rate'].apply(lambda x: np.round(x*20)/20)
data_change['ship_rate_change']=data_change['ship_rate']
data_change.loc[data_change['ship_rate_change']<=0.7,'ship_rate_change']=-2
data_change.loc[(data_change['ship_rate_change']>0.7)*(data_change['ship_rate_change']<0.9),'ship_rate_change']=-1
data_change.loc[data_change['ship_rate_change']==0.95,'ship_rate_change']=0
data_change.loc[data_change['ship_rate_change']>0.95,'ship_rate_change']=1
data_dummies=pd.get_dummies(data_change['ship_rate_change'],prefix = 'ship_rate_change')
data_final_feature = data_final_feature.join(data_dummies[['ship_rate_change_-2.0','ship_rate_change_1.0']])

data_change['reject_rate']=data_change['reject_rate'].apply(lambda x: np.round(x*20)/20)
data_change['reject_rate_change']=data_change['reject_rate']
data_change.loc[data_change['reject_rate_change']<1,'reject_rate_change']=0
data_change.loc[data_change['reject_rate_change']>=1,'reject_rate_change']=1
data_final_feature['reject_rate_change']=data_change['reject_rate_change']

data_change['returned_rate']=data_change['returned_rate'].apply(lambda x: np.round(x*20)/20)
data_change['returned_rate_change']=data_change['returned_rate']
data_change.loc[data_change['returned_rate_change']>=1,'returned_rate_change']=1
data_change.loc[data_change['returned_rate_change']==0.95,'returned_rate_change']=6
data_change.loc[data_change['returned_rate_change']==0.9,'returned_rate_change']=3
data_change.loc[(data_change['returned_rate_change']<0.9)*(data_change['returned_rate_change']>0.1),'returned_rate_change']=4
data_change.loc[data_change['returned_rate_change']==0.1,'returned_rate_change']=0
data_change.loc[data_change['returned_rate_change']==0.05,'returned_rate_change']=6
data_dummies=pd.get_dummies(data_change['returned_rate_change'],prefix = 'returned_rate_change')
data_final_feature = data_final_feature.join(data_dummies[['returned_rate_change_0.0','returned_rate_change_1.0','returned_rate_change_4.0']])

data_change['rec_rate']=data_change['rec_rate'].apply(lambda x: np.round(x*20)/20)
data_change['rec_rate_change']=data_change['rec_rate']
data_change.loc[data_change['rec_rate_change']>=1,'rec_rate_change']=1
data_change.loc[(data_change['rec_rate_change']<=0.95)*(data_change['rec_rate_change']>0.6),'rec_rate_change']=2
data_change.loc[data_change['rec_rate_change']<=0.6,'rec_rate_change']=3
data_dummies=pd.get_dummies(data_change['rec_rate_change'],prefix = 'rec_rate_change')
data_final_feature = data_final_feature.join(data_dummies)

data_change['address_problem_order_rate']=data_change['address_problem_order_rate'].apply(lambda x: np.round(x*20)/20)
data_change['address_problem_order_rate']=data_change['address_problem_order_rate'].fillna(-1)
data_change['address_problem_order_rate_change']=data_change['address_problem_order_rate']
data_change.loc[data_change['address_problem_order_rate_change']>=1,'address_problem_order_rate_change']=1
data_change.loc[(data_change['address_problem_order_rate_change']<1)*(data_change['address_problem_order_rate_change']>0),'address_problem_order_rate_change']=2
data_dummies=pd.get_dummies(data_change['address_problem_order_rate_change'],prefix = 'address_problem_order_rate_change')
data_final_feature = data_final_feature.join(data_dummies[['address_problem_order_rate_change_-1.0','address_problem_order_rate_change_0.0']])

data_change['no_answer_problem_order_rate']=data_change['no_answer_problem_order_rate'].apply(lambda x: np.round(x*20)/20)
data_change['no_answer_problem_order_rate']=data_change['no_answer_problem_order_rate'].fillna(-1)
data_change['no_answer_problem_order_rate_change']=data_change['no_answer_problem_order_rate']
data_change.loc[data_change['no_answer_problem_order_rate_change']>=1,'no_answer_problem_order_rate_change']=1
data_change.loc[(data_change['no_answer_problem_order_rate_change']<1)*(data_change['no_answer_problem_order_rate_change']>0.1),'no_answer_problem_order_rate_change']=2
data_change.loc[(data_change['no_answer_problem_order_rate_change']<=0.1)*(data_change['no_answer_problem_order_rate_change']>0),'no_answer_problem_order_rate_change']=3
data_dummies=pd.get_dummies(data_change['no_answer_problem_order_rate_change'],prefix = 'no_answer_problem_order_rate_change')
data_final_feature = data_final_feature.join(data_dummies[['no_answer_problem_order_rate_change_-1.0','no_answer_problem_order_rate_change_1.0','no_answer_problem_order_rate_change_2.0']])

data_change['month_end_prop']=data_change['month_end_prop'].apply(lambda x: np.round(x*20)/20)
data_change['month_end_prop_change']=data_change['month_end_prop']
data_change.loc[data_change['month_end_prop_change']>0.55,'month_end_prop_change']=1
data_change.loc[(data_change['month_end_prop_change']<=0.55)*(data_change['month_end_prop_change']>0),'month_end_prop_change']=2
data_dummies=pd.get_dummies(data_change['month_end_prop_change'],prefix = 'month_end_prop_change')
data_final_feature = data_final_feature.join(data_dummies['month_end_prop_change_2.0'])

data_change['day_afternoon_prop']=data_change['day_afternoon_prop'].apply(lambda x: np.round(x*20)/20)
data_change['day_afternoon_prop_change']=data_change['day_afternoon_prop']
data_change.loc[(data_change['day_afternoon_prop_change']<1)*(data_change['day_afternoon_prop_change']>0),'day_afternoon_prop_change']=2
data_dummies=pd.get_dummies(data_change['day_afternoon_prop_change'],prefix = 'day_afternoon_prop_change')
data_final_feature = data_final_feature.join(data_dummies[['day_afternoon_prop_change_0.0','day_afternoon_prop_change_2.0']])

data_final_feature=data_final_feature.drop(['rec_num_change_0','customer_service_time_change_-1.0','ship_time_change_-1.0','transport_time_change_-1.0',
                                            'deliver_time_change_-1.0','no_answer_problem_order_change_-1.0','address_problem_order_change','day_afternoon_num_change',
                                            'returned_num_change_0'],axis=1)

##############模型构建
dataX_final_deal=data_final_feature.drop(['lable'],axis=1)
dataY_final_deal=data_final_feature['lable']
dataX_final_deal.describe(),dataY_final_deal.describe()
##判断是否存在null值
dataX_final_deal.isnull().any()

###########模型构建
###数据集拆分，训练集、测试集、验证集
from sklearn.model_selection import train_test_split
from sklearn import cross_validation

X, X_verify, y, y_verify = train_test_split(dataX_final_deal,dataY_final_deal, test_size=0.2)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)

y_train.describe(),y_verify.describe(),y_test.describe()

'''
####数据不平衡处理
#smote方法过采样
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)
print(sorted(Counter(y_resampled).items()))
#adasyn方法过采样
X_resampled, y_resampled = ADASYN().fit_sample(X_train, y_train)
print(sorted(Counter(y_resampled).items()))
#随机过采样
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
#聚类中心欠采样
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_sample(X_train, y_train)
#最邻近欠采样
from imblearn.under_sampling import EditedNearestNeighbours
enn = EditedNearestNeighbours(random_state=0)
X_resampled, y_resampled = enn.fit_sample(X_train, y_train)
##欠采样和过采样结合
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_sample(X_train, y_train)
'''

###模型构建
##决策树
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report

result=pd.DataFrame(columns=['precision','recall','f1','precision_test','recall_test','f1_test'])

##用于确定最优深度值
for i in range(32):
    clf = tree.DecisionTreeClassifier(max_depth=i+1)
    clf.fit(X_train, y_train) 
    
    y_pro = clf.predict(X_train) 
    y_test_pro=clf.predict(X_test)
    
    precision=metrics.precision_score(y_train, y_pro, average='micro')
    recall=metrics.recall_score(y_train, y_pro, average='micro')
    f1=metrics.f1_score(y_train, y_pro, average='weighted') 
    precision_test=metrics.precision_score(y_test, y_test_pro, average='micro')
    recall_test=metrics.recall_score(y_test, y_test_pro, average='micro')
    f1_test=metrics.f1_score(y_test, y_test_pro, average='weighted') 
      
    result.loc[i,'precision']=precision
    result.loc[i,'recall']=recall
    result.loc[i,'f1']=f1
    result.loc[i,'precision_test']=precision_test
    result.loc[i,'recall_test']=recall_test
    result.loc[i,'f1_test']=f1_test
   
    
clf = tree.DecisionTreeClassifier(max_depth=9)
clf.fit(X_train, y_train) 

y_pro = clf.predict(X_train) 
y_test_pro=clf.predict(X_test)
y_verify_pro=clf.predict(X_verify)

np.mean(y_pro),np.mean(y_test_pro),np.mean(y_verify_pro)

##准确率召回率
print(classification_report(y_train, y_pro))
print(classification_report(y_test, y_test_pro))
print(classification_report(y_verify, y_verify_pro))
metrics.precision_score(y_train, y_pro, average='micro')
metrics.recall_score(y_train, y_pro, average='micro')
metrics.f1_score(y_train, y_pro, average='weighted') 
metrics.precision_score(y_test, y_test_pro, average='micro')
metrics.recall_score(y_test, y_test_pro, average='micro')
metrics.f1_score(y_test, y_test_pro, average='weighted') 

##输出各特征重要性
print(clf.feature_importances_) 
g=pd.DataFrame(dataX_final_deal.columns.values)
g1=clf.feature_importances_

##预测特征得分
y_score = clf.predict_proba(X_train)[:,1]
y_test_score=clf.predict_proba(X_test)[:,1]
y_verify_score=clf.predict_proba(X_verify)[:,1]


##pr curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_test_score)
average_precision = average_precision_score(y_test, y_test_score)

plt.step(recall, precision, color='b', alpha=0.2,where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


##计算auc值以及做图 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_train, y_score)
roc_auc_score(y_test, y_test_score)
roc_auc_score(y_verify, y_verify_score)

fpr, tpr, thresholds = roc_curve(y_train, y_score)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_score)
fpr_verify, tpr_verify, thresholds_verify = roc_curve(y_verify, y_verify_score)

plt.figure()
plt.plot(fpr, tpr,color='deeppink',label='train ROC curve', linestyle=':', linewidth=4)
plt.plot(fpr_test, tpr_test,color='darkorange',label='test ROC curve', linestyle=':', linewidth=4)
plt.plot(fpr_verify, tpr_verify,color='cornflowerblue',label='verify ROC curve', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

##输出决策树结果
feature_name=[ 'sex_change','age_change','device_change','pay_num_change',
          'returned_num_change','rec_num_change','returned_rate_change','rec_rate_change',
          'is_send_regemail','register_from','language','app_type','country_change']

from IPython.display import Image  
from sklearn import tree
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=feature_name,  
                         class_names=['0','1'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 
#graph.write_pdf("tree.pdf")

datay_score=clf.predict_proba(dataX_final_deal)[:,1]
datay_score=pd.DataFrame(datay_score,columns=['datay_score'])
datay_score['lable']=dataY_final_deal

p_result=[]
pV_result=[]
for i in range(10):
    pecent=1.0*len(datay_score[datay_score['datay_score']>=1.0*i/10])/len(datay_score)
    pecentV=1.0*len(datay_score[(datay_score['datay_score']>=1.0*i/10)*(datay_score['lable']==1)])/len(datay_score)
    p_result=p_result+[pecent]
    V_result=pV_result+[pecentV]

def data_analysis():
    x = {}
    