import numpy as np
d=[('n','S12'),('class',int),('h',float)]
st=eval(input())
x=np.array(st,dtype=d)
print(np.sort(x,order=['class','h']))
