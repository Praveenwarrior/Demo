#!/usr/bin/env python
# coding: utf-8

# In[7]:


l1=[[1,2,3],[2,3,4]]
type(l1)
l1.__class__


# In[12]:


import numpy as np
arr1=np.array(l1)
arr1
len(arr1)
arr1.shape
arr1.dtype
arr1.ndim
arr1.__class__


# In[19]:


l2=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
arr2=np.array(l2)
arr2
len(arr2)
arr2.shape
arr2.dtype
arr2.ndim
arr2.__class__
a=list(arr2)
print(l2[0])
a[0]


# In[26]:


a=np.zeros((3,6))
b=np.ones(4)
c=np.empty((2,2))
c


# In[30]:


a=np.ones(2,dtype=np.float64)
print(a)
a.dtype


# In[37]:


a=np.arange(10)
s=np.arange(0,10,2)
b=np.linspace(0,1,5)
c=np.linspace(0,2.5,5)
c
isinstance(c,tuple)


# In[41]:


a.reshape((2,5))


# In[48]:


a=np.arange(12)
b=a.reshape((3,4))
print(b)
b[0]
b[2,3]
b[1,2]
##b[5,2]
vl=b[2,2]
v1


# In[55]:


b[0:]
b[:1]
b[0,0:3]
b[0:2,0:3]
b[:,1:3]
b[[0,1],[0,1]]


# In[72]:


arr=np.arange(15)
arr1=arr.reshape((3,5))
print(arr1)
arr1[0:1,3:]
arr1[1:,2:4]
arr1[1:2,1:3]
##Tranpose for an array
arr1.T  
np.transpose(arr1)


# In[84]:


ar=np.array([2,8,4,6,9,7,10,18,6,19,24,5])
ar.sort()
ar
k1=np.array([[1,7],[3,2]])
k1
k2=np.sort(k1,axis=0)
k2
k2=np.sort(k1,axis=1)
k2
np.sqrt(k2)
k12=np.concatenate((k1,k2),axis=0)  ##Concatenation of 2 arrays 
k12
##Note:- Concatenate can also be done to the mutliple arrays of same dimension


# In[95]:


p=np.array([[1,2],[3,4]])
q=np.arange(6)
p1=q[:,np.newaxis]
p1


# In[100]:


e=np.array([[4,1],[7,2]])
np.sort(e,axis=0)


# In[101]:


np.sort(e)


# In[113]:


print(a[a%2==0])
print(a[(a>2)&(a<5)])
a=(a>2) | (a<5)
a


# In[119]:


arr11=np.array([1,2,3,4,5])
print(arr11*2)
print(arr11.prod())
print(arr11.sum())
print(arr11.min())
print(arr11.max())


# In[122]:


arr11=arr11*2
arr11


# In[180]:


arr12=np.array([1,2,3,5,6.9,6])
type(arr12)
arr12.dtype
print(arr12)
arr12.sort()
arr12
arr13=arr12.reshape((2,3))
print(arr13)
np.sqrt(arr13)
np.exp(arr13)
np.log(arr13)
arr13.min(axis=0)
arr13.min(axis=1)
arr13.max(axis=0)
arr13.max(axis=1)
np.var(arr13)
np.mean(arr13)
np.median(arr13)
np.std(arr13)
np.std(arr13)**2
np.info(arr13)
np.percentile(arr13,20)
np.quantile(arr13,0.6)
np.iqr(arr13)


# In[145]:


x=4.412583
print(np.ceil(x))
print(np.round(x,4))
print(np.floor(x))


# In[169]:


arb=np.array([2,8,4.0,6,9,7,10.3,18,6.9,19,24,5])
ar=arb.reshape(3,4)
print(ar)
np.flip(ar)
np.flip(ar).T
np.flip(ar[:,2:3])
ar[:,2:3]=np.flip(ar[:,2:3])
ar


# In[170]:


ar=np.array([2,8,4.0,6,9,7,10.3,18,6.9,19,24,5])
t=np.array([])
for i in range(len(ar),0):
    t=t+ar[i]
    print(t)


# In[173]:


ar=np.array([2,8,4.0,6,9,7,10.3,18,6.9,19,24,5])
t=np.array([])
for i in range(len(ar),0):
    t=t+ar[i]
    t


# In[193]:


import pandas as pd
iris=pd.read_csv("C:\\Users\\kalpraveen@deloitte.com\\Documents\\dataset\\Dataset 1\\DelProj-main\\Iris.csv")
iris=pd.DataFrame(iris)
del iris['Id']
iris=iris.rename(columns={'SepalLengthCm': 'sepal_length',
                          'SepalWidthCm':'sepal_width',
                          'PetalLengthCm':'petal_length',
                          'PetalWidthCm':'petal_width',
                          'Species':'class'})
iris.head(3)
iris


# In[192]:


import matplotlib.pyplot as plt
plt.hist(iris.sepal_length,bins='auto',facecolor='red')
plt.ylabel("count")
plt.xlabel("Sepal Length")
plt.title("Histogram for Sepal Length")


# In[195]:


iris.sepal_length.hist()


# In[203]:


plt.scatter(iris['sepal_width'],iris['sepal_length'])
iris.corr()


# In[204]:


iris.plot.scatter(x='sepal_length',y='sepal_width',title="Scatter Plot for IRIS")


# In[205]:


iris['sepal_length'].plot.line(title="Iris Dataset")


# In[208]:


iris.drop(['class'],axis=1).plot.line(title="Iris Dataset")
iris['sepal_length','petal_length'].plot.line(title="Iris Dataset")


# In[209]:


iris.plot.hist(subplots='true',layout=(2,2),figsize=(10,10),bins=20)


# In[220]:


#barchart
iris['sepal_length'].value_counts().sort_index().plot.bar()
iris['sepal_length'].value_counts().plot.bar()


# In[232]:


iris.groupby('class').sepal_length.mean().sort_values(ascending=False)[:6].plot.bar()


# In[223]:


iris.groupby('class').sepal_length.mean().sort_values(ascending=False).plot.bar()


# In[237]:


import pandas as pd
grades=pd.read_csv("C:\\Users\\kalpraveen@deloitte.com\\Documents\\dataset\\grades.csv")
grades=pd.DataFrame(grades)
grades.groupby('ethnicity').total.mean().sort_values(ascending=True)[:5].plot.bar()


# In[239]:


#heatmap
import numpy as np
import matplotlib.pyplot as plt

#get correlation matrix
corr=iris.corr()
fig,ax=plt.subplots()

#create a heatmap
im=ax.imshow(corr.values)

#set labels
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)

#Rotate the tick labels and set their alignment
plt.setp(ax.set_xticklabels(),rotation=45,ha="right",rotation_mode="Anchor")


# In[241]:


import seaborn as sns
sns.heatmap(iris.corr(),annot=True)


# In[243]:


import seaborn as sns
g=sns.FacetGrid(iris,col="class")
g=g.map(sns.kdeplot,'sepal_length')


# In[248]:


import seaborn 
seaborn.pairplot(iris, vars=['sepal_length','sepal_width','petal_length'],kind='reg')


# In[253]:


colors={'Iris-setosa':'r','Iris-versicolor':'g','Iris-virginica':'b'}
fig,ax=plt.subplots()
for i in range(len(iris['sepal_length'])):
    ax.scatter(iris['sepal_length'][i],iris['sepal_width'][i],color=colors[iris['class'][i]])
ax.set_title('Iris dataset')
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')


# In[255]:


columns=iris.columns.drop(['class'])
x_data=range(0,iris.shape[0])
fig,ax=plt.subplots()
for column in columns:
        ax.plot(x_data,iris[column],label=column)
ax.set_title("Iris Dataset")
ax.legend()


# In[ ]:




