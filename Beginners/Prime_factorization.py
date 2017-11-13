#!/usr/bin/python3
import math

n=int(input("Enter an integer:"))

def is_prime(n = 600851475143):
    i=1
    prime_factors = [1]
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
        i=i+1
    if len(prime_factors)==1:
        return (True, prime_factors)
    else:
        return (False, prime_factors)
    
print (is_prime(n))