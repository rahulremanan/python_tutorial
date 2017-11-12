#!/usr/bin/python3

start = int(input("Give me a start number :"))
end = int(input("Give me an end number :"))
            
          
def fizz_buzz(start, end):
    for num in range(start, end +1):
        if num % 3 == 0 and num % 5 == 0:
            print(num, "fizzbuzz")
        elif num % 3 ==0:
            print (num, "fizz")
        elif num % 5 ==0:
            print (num, "buzz")
        else:
            print (num, "nothing")
    return None

fizz_buzz(start, end)