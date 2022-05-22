import numpy as np
x = np.array(eval(input()), dtype=str)
y=np.char.replace(x,'php','python')
print(y)
