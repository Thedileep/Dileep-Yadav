def prime(a):
    for i in range(2,a):
        if(a%i==0):
            return 'not prime'
        return 'prime'
a=int(input())
print(prime(a))
