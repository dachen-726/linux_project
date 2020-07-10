from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
import joblib
import csv

#读取csv文件
def ReadMyCsv(savelist,filename):
    csv_reader = csv.reader(open(filename))
    for row in csv_reader:
        savelist.append(row)
    return

def WriteMyCsv(filename,data):
    with open(filename,"w",newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def Feature2Num(row):
    if row[1]=='private':
        row[1]=1
    elif row[1]=='self-emp-not-inc':
        row[1]=2
    elif row[1]=='self-emp-inc':
        row[1]=3
    elif row[1]=='federal-gov':
        row[1]=4
    elif row[1]=='local-gov':
        row[1]=5
    elif row[1]=='state-gov':
        row[1]=6
    else:
        row[1]=0

    if 'th' in row[2]:
        row[2]=0
    elif row[2] == 'preschool':
        row[2]=0
    elif 'hs' in row[2]:
        row[2]=1
    elif row[2] in ['some-college','assoc-acdm','assoc-voc']:
        row[2]=2
    elif row[2] in ['bachelors','masters']:
        row[2]=3
    elif row[2] in ['prof-school','doctorate']:
        row[2]=4

    if row[4] in ['married-civ-spouse','widowed']:
        row[4] = 0
    elif row[4] in ['divorced','separated','married-spouse-absent']:
        row[4] = 1
    else:
        row[4] = 2

    if row[5] in ['tech-support','sales','exec-managerial','prof-specialty','machine-op-inspct','adm-clerical','craft-repair']:
        row[5] = 0
    elif row[5] in ['farming-fishing','transport-moving','priv-house-serv','protective-serv','armed-Forces','handlers-cleaners','other-service']:
        row[5] = 1
    else:
        row[5] = 2

    if row[6] == 'wife':
        row[6] = 0
    elif row[6] == 'own-child':
        row[6] = 1
    elif row[6] == 'husband':
        row[6] = 2
    elif row[6] == 'not-in-family':
        row[6] = 3
    elif row[6] == 'other-relative':
        row[6] = 4
    elif row[6] == 'unmarried':
        row[6] = 5

    if row[7] == 'white':
        row[7] = 0
    elif row[7] == 'asian-pac-islander':
        row[7] = 1
    elif row[7] == 'amer-indian-eskimo':
        row[7] = 2
    elif row[7] == 'other':
        row[7] = 3
    elif row[7] == 'black':
        row[7] = 4

    if row[8] == 'female':
        row[8] = 0
    else:
        row[8] = 1

    counter = 0
    while counter < len(row):
        row[counter] = float(row[counter])
        counter += 1

def PreHandle(initData,incomeLv):
    for row in initData:
        if row[14]==' <=50K.':
            incomeLv.append(0)
        else:
            incomeLv.append(1)
        row.pop()   #删除myList中每一个个体的收入水平
        row.pop(2)   #删除无用的指标“权重”
        row.pop(12)   #删除指标“国家”
        

        counter = 0
        while counter < len(row):
            row[counter] = row[counter].lower()  #小写
            row[counter] = row[counter].strip()  #删除开头的空格
            counter += 1

        #下面将每一个指标都转换为相应的数字
        Feature2Num(row)

def FeatureExtract(features,initlist):
    counter = 0
    for row in initlist:
        features.append([row[0],row[4],row[3],row[9],row[10],row[11],row[1]])
        counter += 1

def NewFeature2Num(row):
    counter = 0
    while counter < len(row):
        row[counter] = row[counter].lower()
        counter += 1

    if row[1] in ['married-civ-spouse','widowed']:
        row[1] = 0
    elif row[1] in ['divorced','separated','married-spouse-absent']:
        row[1] = 1
    else:
        row[1] = 2 

    if row[6]=='private':
        row[6]=1
    elif row[6]=='self-emp-not-inc':
        row[6]=2
    elif row[6]=='self-emp-inc':
        row[6]=3
    elif row[6]=='federal-gov':
        row[6]=4
    elif row[6]=='local-gov':
        row[6]=5
    elif row[6]=='state-gov':
        row[6]=6
    else:
        row[6]=0

    print(row)
    counter = 0
    while counter < len(row):
        row[counter] = float(row[counter])
        counter += 1
   
#存储原始数据
myList=[]
ReadMyCsv(myList,'data.txt')
#print(myList)

incomeLv=[]   #存储每个人的收入水平，0='<=50K',1='>50K'
#数据预处理
PreHandle(myList,incomeLv)
    
#print(myList)
#WriteMyCsv('numed_data.csv',myList)

'''
数据集的属性信息:
1.年龄:数值

2.工作类别:Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov,  
State-gov, （Without-pay, Never-worked. ）=？     ==>[1,2,...6,0]

3.教育程度:Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, 
Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, 
Preschool.  
分组（pre+1-12）（HS）（somecollege，Assoc-acdm, Assoc-voc)（Bachelors, Masters) (Prof-school，Doctorate）  ==>[0,1,2,3,4]

4.受教育时间:数值. 

5.婚姻状况: Married-civ-spouse, Divorced, Never-married, Separated,   
Widowed, Married-spouse-absent, Married-AF-spouse. 
(婚姻质量)（Married-civ-spouse，Widowed）（Divorced，Separated，Married-spouse-absent）(other)  ==>[0,1,2]

6.职业: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, 
Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, 
Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, 
Armed-Forces. 
技术支持，工艺维修，其他服务，销售，执行管理，
专业教授，搬运工人清洁工，机器操作检验，行政助理，
农家捕鱼，运输，私人房屋服务，保护性服务，
武装部队。
（工作科技含量）
（脑力劳动:Tech-support,Sales, Exec-managerial, Prof-specialty,Machine-op-inspct,Adm-clerical,Craft-repair)
（体力劳动:Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces, Handlers-cleaners）
（other-service）
==>[0,1,2]

7.家庭关系: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.   ==>[0,1,2,3,4,5]

8.种族: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.  ==>[0,1,2,3,4]

9.性别: Female, Male.  ==>[0,1]

10.资本收益: 连续数值. 
11.资本亏损: 连续数值. 
12.每周工作小时数: 连续数值. （和工资的关系）
'''

#特征选择
'''
clf=DecisionTreeClassifier()
clf.fit(myList,incomeLv)
feat_importance = clf.tree_.compute_feature_importances(normalize=False)
print(str(feat_importance))

【输出结果】各项特征的重要性：
[0.06009538 0.01780836 0.0080967  0.04468487 0.06637365 0.00655111
 0.01077092 0.00714767 0.00414579 0.04005513 0.01505239 0.03569615]
 1.年龄  2.婚姻状况  3.受教育时间  4.资本收益  5.资本亏损  6.每周工作小时数  7.工作类别

'''

#提取重要性比较高的特征
feature_extracted=[]
FeatureExtract(feature_extracted,myList)
#WriteMyCsv('feature_extracted.csv',feature_extracted)

#构建随机森林模型
X_train,X_test,y_train,y_test = train_test_split(feature_extracted,incomeLv,test_size=0.3)
rfc = RandomForestClassifier(criterion='gini',n_estimators=25,max_features=3)
rfc = rfc.fit(X_train, y_train)
score_r = rfc.score(X_test, y_test)
print("预测效果：" + str(score_r))

#保存训练好的预测模型
joblib.dump(rfc,"RandomForest_model.m")
   
rfcs = []
tprs = []
fprs = []
aucs = []
colorlist = ['red', 'gold', 'purple', 'green', 'blue', 'black']

plt.subplots(figsize=(7, 5.5))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest 5-fold ROC Curve')

for i in range(5):
    #rfc = joblib.load('RandomForest_model.m')
    X_train,X_test,y_train,y_test = train_test_split(feature_extracted, incomeLv, test_size = 0.3)
    rfc = RandomForestClassifier(criterion='gini',n_estimators=25,max_features=3)
    rfc = rfc.fit(X_train, y_train)
    y_score = rfc.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:,1], pos_label = 1)
    roc_auc = auc(fpr, tpr)
    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, color=colorlist[i], ls='-', lw=1, label='AUC%d = %0.4f' % (i,roc_auc))
    #rfc_s = cross_val_score(rfc, X_train, y_train, cv = 10).mean()
    #rfc_l.append(rfc_s)
mean_auc = sum(aucs)/5
plt.plot(0,0,label='MEAN_AUC=%0.4f'%mean_auc)
plt.legend()
plt.show()
