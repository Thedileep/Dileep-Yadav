dict1 = {'a': 12, 'for': 25, 'c': 9}
dict2 = {'Geeks': 100, 'geek': 200, 'for': 300}
for i in dict2:
    if i in dict1:
        dict2[i]=dict1[i]+dict2[i]
    else:
        pass
print(dict2)
