import numpy as np
x = np.array(eval(input()), dtype=str)
en=np.char.encode(x)
de=np.char.decode(en)
print(f'{en} \n {de}')
