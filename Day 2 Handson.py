#!/usr/bin/env python
# coding: utf-8

# In[7]:


x=20 
y=30.94 
print(x+y)
print('The sum of ',x,'and',y,'is',x+y)


# In[12]:


s="I Love India"
s.split()
s[5]


# In[15]:


import math   
x=int(input())  
print("the sqrt of {0} is {1}".format(x,math.sqrt(x)))


# In[22]:





# In[24]:


r=int(input())  
pi=3.14  
area=pi*r**2                         
print("the area of circle with radius {0} is {1}".format(r,area))


# In[33]:


import cmath
a=float(input())
b=float(input())
c=float(input())
dis=b**2-(4*a*c)
qe=(-b+(cmath.sqrt(dis)))/2*a
qe1=(-b-(cmath.sqrt(dis)))/2*a
print("The Quadratic roots are {0},{1}".format(qe,qe1))


# In[34]:


x=int(input())
y=bool(x%2==0)
print(y)


# In[40]:


x=int(input())
print("Even"*(x%2==0),"Odd"*(x%2!=0))


# In[37]:


n = int(input("Enter a number: "))
print("Even" * (n % 2 == 0), "Odd" * (n % 2 != 0))


# In[42]:


num=9
print("Positive"*(num>0),"Negative"*(num<0))


# In[46]:


import math
name=input()
luck=len(name)+len(name)**2
print(luck)
if luck%2==0:
    print("Even")
else:
    print("Odd")


# In[47]:


name=input('Enter the name :')
n=len(name)
c=0
r=n
while n>0:
    r=n%10
    c=c+r
    n=n//10
lucky=c**2
print(lucky)


# In[53]:


l=input()
if l in ('a,e,i,o,u,A,E,I,O,U'):
    print("Vowel")
else:
    print("Consonant")


# In[55]:


n=int(input("Enter the number"))


# In[ ]:


n=int(input("Enter the number: "))
if n%10==0 && (n%10)%3==0:
    print("Yes")
else:
    print("No")


# In[ ]:


n=int(input())
mid=(n//10)%10
mid


# In[63]:


n=input()


# In[ ]:


n=int(input("Enter the number: "))
if n%10==0 && (n%10)%3==0:
    print("Yes")
else:
    print("No")


# In[ ]:


n=input()
mid=((n%100)-(n%10))/10
mid


# In[ ]:




