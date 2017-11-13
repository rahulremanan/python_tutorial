#!/usr/bin/python3
import math
# n = 600851475143
n=int(input("Enter an integer:"))
print("Factors for ", n, "are:")
i=1
prime_factors = []
while(i<=math.ceil(n**0.5)):
    k=0
    if(n%i==0):
        j=1
        while(j<=i):
            if(i%j==0):
                k=k+1
            j=j+1
        if(k==2):
            prime_factors.append(i)
            print(i)
    i=i+1

if len(prime_factors)==0:
    print ("Therefore", n, " is a prime number")
else:
    print ("Therefore", n, "is not a prime number")