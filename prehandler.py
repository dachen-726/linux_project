from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np
import csv

#��ȡcsv�ļ�
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


#�洢ԭʼ����
myList=[]
ReadMyCsv(myList,'C:/Users/lenovo/Desktop/RandomForest/data.csv')
#print(myList)

#�洢ÿ���˵�����ˮƽ��0='<=50K',1='>50K'
incomeLv=[]
for row in myList:
    if row[14]==' <=50K.':
        incomeLv.append(0)
    else:
        incomeLv.append(1)
    row.pop()   #ɾ��myList��ÿһ�����������ˮƽ
    row.pop(2)   #ɾ�����õ�ָ�ꡰȨ�ء�
    row.pop(12)   #ɾ��ָ�ꡰ���ҡ�
    

    counter = 0
    while counter < len(row):
        row[counter] = row[counter].lower()  #Сд
        row[counter] = row[counter].strip()  #ɾ����ͷ�Ŀո�
        counter += 1

    #���潫ÿһ��ָ�궼ת��Ϊ��Ӧ������
    Feature2Num(row)
    
#print(myList)
#WriteMyCsv('numed_data.csv',myList)

'''
myList�����ݼ���������Ϣ:
1.����:��ֵ

2.�������:Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov,  
State-gov, ��Without-pay, Never-worked. ��=��     ==>[1,2,...6,0]

3.�����̶�:Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, 
Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, 
Preschool.  
���飨pre+1-12����HS����somecollege��Assoc-acdm, Assoc-voc)��Bachelors, Masters) (Prof-school��Doctorate��  ==>[0,1,2,3,4]

4.�ܽ���ʱ��:��ֵ. 

5.����״��: Married-civ-spouse, Divorced, Never-married, Separated,   
Widowed, Married-spouse-absent, Married-AF-spouse. 
(��������)��Married-civ-spouse��Widowed����Divorced��Separated��Married-spouse-absent��(other)  ==>[0,1,2]

6.ְҵ: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, 
Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, 
Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, 
Armed-Forces. 
����֧�֣�����ά�ޣ������������ۣ�ִ�й���
רҵ���ڣ����˹�����๤�������������飬��������
ũ�Ҳ��㣬���䣬˽�˷��ݷ��񣬱����Է���
��װ���ӡ�
�������Ƽ�������
�������Ͷ�:Tech-support,Sales, Exec-managerial, Prof-specialty,Machine-op-inspct,Adm-clerical,Craft-repair)
�������Ͷ�:Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces, Handlers-cleaners��
��other-service��
==>[0,1,2]

7.��ͥ��ϵ: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.   ==>[0,1,2,3,4,5]

8.����: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.  ==>[0,1,2,3,4]

9.�Ա�: Female, Male.  ==>[0,1]

10.�ʱ�����: ������ֵ. 
11.�ʱ�����: ������ֵ. 
12.ÿ�ܹ���Сʱ��: ������ֵ. ���͹��ʵĹ�ϵ��
'''

#����ѡ��
clf=DecisionTreeClassifier()
clf.fit(myList,incomeLv)
feat_importance = clf.tree_.compute_feature_importances(normalize=False)
print(str(feat_importance))

'''
������������Ҫ�ԣ�
[0.06009538 0.01780836 0.0080967  0.04468487 0.06637365 0.00655111
 0.01077092 0.00714767 0.00414579 0.04005513 0.01505239 0.03569615]
 1.����  2.����״��  3.�ܽ���ʱ��  4.�ʱ�����  5.�ʱ�����  6.ÿ�ܹ���Сʱ��  7.�������
'''

feature_extracted=[]
counter = 0
for row in myList:
    feature_extracted.append([row[0],row[4],row[3],row[9],row[10],row[11],row[1],incomeLv[counter]])
    counter += 1
WriteMyCsv('feature_extracted.csv',feature_extracted)
