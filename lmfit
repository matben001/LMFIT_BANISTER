
import pandas as pd
data = pd.read_csv('data_mod1.csv')


print(data)





import numpy as np
import math
x = data['stress'].fillna(0)
y = data['Performnace']


# In[4]:


a = [x[0]]
b = [x[0]]
def Banister(x, t1, t2, k1, k2, p0):
    for i in x: 
        a.append(a[-1]*np.exp(-1/t1) + i)
        b.append(b[-1]*np.exp(-1/t2) + i)
        data['fit'] = pd.Series(a)
        data['fat'] = pd.Series(b)
        data['perf'] = ((data['fit']*k1)-(data['fat']*k2))+p0
    return data['perf'] 


# In[6]:


from lmfit import Model


y2 = np.array(data['avg_perf'])


gmodel = Model(Banister)
result = gmodel.fit(y2, x=x, t1=50, t2=5,k1=1,k2=1.11, p0=60)

print(result.fit_report())
