#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
grades=pd.read_csv("C:\\Users\\kalpraveen@deloitte.com\\Documents\\dataset\\grades.csv")
grades=pd.DataFrame(grades)

cs2m=pd.read_csv("C:\\Users\\kalpraveen@deloitte.com\\Documents\\dataset\\cs2m.csv")
cs2m=pd.DataFrame(cs2m)
print()
print(len(cs2m.Prgnt))
cs2m.DrugR.unique().shape
grades.quiz1.dtype
grades.info()
grades.describe()
cs2m.DrugR.value_counts()
grades.ethnicity.value_counts()
print(grades.quiz1.min())
print(grades.quiz1.max())
print(grades.quiz1.sum())
print(grades.quiz1.skew())
print(grades.quiz1.std())
print(grades.quiz1.kurtosis())
print(grades.quiz1.kurt())


# In[25]:


print(grades.info)
print(grades.quiz1.describe())
print(grades.quiz1.shape)
print(grades.skew())


# In[28]:


cs2m.head(3)
print(cs2m.tail(4))


# In[37]:


file1=grades.iloc[:,0:4].head(50)
file2=grades.iloc[:,1:4].tail(50)
file3=grades.iloc[50:71,:]
file3
file2.shape
file2.describe()


# In[40]:


grades.quiz1.compress((grades.quiz1>=8))


# In[44]:


import numpy as np
grades[grades.quiz1==8]


# In[57]:


cs2m_con=cs2m[cs2m.Age>20]
cs2m_con1=cs2m_con[cs2m_con.DrugR==1]
cs2m_con1


# In[67]:


cs2m_con=cs2m[(cs2m.Age>20) & (cs2m.DrugR==1)]
cs2m_con


# In[68]:


grades3=grades[grades.ethnicity==3]
grades3.head(3)
grades5=grades[grades.ethnicity==5]
grades3.head(3)
grade35=pd.concat([grades3,grades5])
grade35


# In[78]:


import numpy as np
cs2m['ChlstrlMeas']=np.where((cs2m['Chlstrl']>175) & (cs2m['BP']>175),'HP','NO')
cs2m


# In[74]:


cs2m['ChlstrlMeas']=np.where(cs2m['Chlstrl']>175,'H','L')
cs2m.head(10)


# In[81]:


def set_marks(row):
    if row['total']<75:
        return "Fail"
    elif row['total']>=75 and row['total']<=100:
        return "Pass"
    else:
        return "Great"
grades=grades.assign(Marksstat=grades.apply(set_marks,axis=1))
grades.head(10)


# In[104]:


##calculate the % from grades.total and create new variable having only % values
grades['percentage']=(grades['total']/200)*100
grades


# In[105]:


grades.shape


# In[ ]:


grades.drop(['percentage',])


# In[106]:


grades.total.groupby(grades.ethnicity).mean()


# In[107]:


grades.total.groupby(grades.ethnicity).describe()


# In[108]:


grades.total.groupby(grades.ethnicity).median()


# In[40]:


from scipy import stats
stats.sem(cs2m.Age)
stats.describe(cs2m.Age)


# In[117]:


print(cs2m)

stats.describe(cs2m)


# In[120]:


j=cs2m.groupby(['Prgnt','AnxtyLH','DrugR'])
cs2m_age=j['Age']
cs2m_age.agg('mean')


# In[123]:


pd.crosstab(cs2m.Prgnt,cs2m.DrugR,margins=True)


# In[124]:


pd.crosstab(cs2m.Prgnt,cs2m.DrugR,margins=True).head()


# In[126]:


def Hello():
    print("Praveen")
Hello()


# In[128]:


def plus(a,b=1):
    return a+b
plus(8)
plus(8,9)


# In[129]:


def sqr(a=5):
    return a**2
sqr(6)


# In[135]:


def ci(p,r,t):
    return p+(p*r*t)/100
ci(1000,5,5)


# In[3]:


import math
def factorial(n):
    return math.factorial(n)
factorial(3)


# In[8]:


def fact(n):
    j=1
    for i in range(1,n+1):
         j=j*i
    return j
fact(3)


# In[15]:


def fact(n):
    if n<=1:
        return 1
    else:
        return n*fact(n-1)
fact(5)


# In[11]:


def plus(*args):
    return sum(args) ##sum(*args)
plus(9,8,7,6)


# In[12]:


def big(a,b):
    if a>b:
        return a
    else:
        return b
big(23,34)


# In[16]:


def check(n):
    if n>=50 and n<=60:
        print("Good")
    else:
        print("Bad")
check(50)


# In[26]:


def even(*l):
    a=[]
    for i in l:
        if i%2==0:
            a.append(i)
    return a
even(1,2,3,445,6,7,9)


# In[41]:


def even(ls):
    a=[]
    for i in ls:
        if i%2==0:
            a.append(i)
    return a
even([1,2,3,4,5,6,7,8,9])


# In[ ]:




