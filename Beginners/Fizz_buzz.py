#!/usr/bin/python3
start = int(input("Give me a start number :"))
end = int(input("Give me an end number :"))
          
def fizz_buzz(start, end):
    fb_state = []
    for num in range(start, end +1):
        if num % 3 == 0 and num % 5 == 0:
            output = {"number": num, "fizz buzz state": "fizzbuzz"}
            fb_state.append(output)
        elif num % 3 ==0:
            output = {"number": num, "fizz buzz state": "fizz"}
            fb_state.append(output)
        elif num % 5 ==0:
            output = {"number": num, "fizz buzz state": "buzz"}
            fb_state.append(output)
        else:
            output = {"number": num, "fizz buzz state": "none"}
            fb_state.append(output)
    return (fb_state)

fb = (fizz_buzz(start, end))

for item in fb:
    print ("Number = ", item["number"], "; fizz buzz State = ", item["fizz buzz state"])