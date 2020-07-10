import tkinter as tk
from tkinter import ttk
from buildRF import NewFeature2Num
import joblib


root = tk.Tk()
screenwidth = root.winfo_screenwidth()
screenheight = root.winfo_screenheight()

width = 500
height = 800
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, (screenheight-height)/2)

root.config()
root.title('Income Predicter')
root.geometry('400x600+20+20')

lb1 = tk.Label(root,text='Age:')
lb1.grid(row=1,column=1,padx=20,pady=20)
age = tk.StringVar()
ageInput = tk.Entry(root, textvariable=age)
ageInput.grid(row=1,column=2)

lb2 = tk.Label(root,text='Marital Status:')
lb2.grid(row=2,column=1,padx=20,pady=20)
marrState = tk.StringVar()
marrStateSelect = ttk.Combobox(root, textvariable=marrState)
marrStateSelect['values'] = ('Married-civ-spouse','Married-spouse-absent','Married-AF-spouse',\
    'Divorced','Separated','Never-married','Widowed')
marrStateSelect.grid(row=2,column=2)

lb3 = tk.Label(root,text='Educated Years:')
lb3.grid(row=3,column=1,padx=20,pady=20)
eduYears = tk.StringVar()
eduYearsInput = tk.Entry(root, textvariable=eduYears)
eduYearsInput.grid(row=3,column=2)

lb4 = tk.Label(root,text='Capital Gain:')
lb4.grid(row=4,column=1,padx=20,pady=20)
capitalGain = tk.StringVar()
capGainInput = tk.Entry(root,textvariable=capitalGain)
capGainInput.grid(row=4,column=2)

lb5 = tk.Label(root,text='Capital Deficiency:')
lb5.grid(row=5,column=1,padx=20,pady=20)
capitalDef = tk.StringVar()
capDefInput = tk.Entry(root,textvariable=capitalDef)
capDefInput.grid(row=5,column=2)

lb6 = tk.Label(root,text='Working Hours per Week:')
lb6.grid(row=6,column=1,padx=20,pady=20)
wkHours = tk.StringVar()
wkHoursInput = tk.Entry(root,textvariable=wkHours)
wkHoursInput.grid(row=6,column=2)

lb7 = tk.Label(root,text='Type of Work:')
lb7.grid(row=7,column=1,padx=20,pady=20)
workType = tk.StringVar()
wkTypeSelect = ttk.Combobox(root,textvariable=workType)
wkTypeSelect['values']=('Private','Self-emp-not-inc','Self-emp-inc','Federal-gov',\
    'Local-gov','State-gov','Without-pay','Never-worked')
wkTypeSelect.grid(row=7,column=2)


def predict():
    Age = age.get()
    MarrState = marrState.get()
    EduYears = eduYears.get()
    CapitalGain = capitalGain.get()
    CapitalDef = capitalDef.get()
    WkHours = wkHours.get()
    WorkType = workType.get()

    features=[Age,MarrState,EduYears,CapitalGain,CapitalDef,WkHours,WorkType]
    print(features)

    NewFeature2Num(features)

    rlc = joblib.load("RandomForest_model.m")
    predicter = rlc.predict([features])
    if predicter == 0:
        predict_text = '<50K'
    elif predicter == 1:
        predict_text = '>50K'
    lb8 = tk.Label(root,text='Predict Result:')
    lb8.grid(row=9,column=1,padx=20,pady=20)
    lb_result = tk.Label(root,text=predict_text,font=('微软雅黑',20))
    lb_result.grid(row=9,column=2,padx=20,pady=20)

btn = tk.Button(root,text='Predict',command=lambda:predict())
btn.grid(row=8,column=2,padx=20,pady=20)

root.mainloop()