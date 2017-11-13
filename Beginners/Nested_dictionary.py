#!/usr/bin/python3

grocery_list = {'Fish in kilograms': [{"Salmon": 10, "Cuttle Fish": 10, "Tuna": 10}],
                'Apples in counts': [{"California Red Apple" : 100, "Granny Smith" : 100}],
                'Milk in liters': [{"Low Fat Milk": 500, "Whole Milk": 500, "Low Fat Milk -- Lactaid": 150, "Whole Milk -- Lactaid": 150}],
                'Eggs in counts': [{"Organic White Eggs": 100, "Organic Brown Eggs": 100}],
                'Chicken in kilograms': [{"Organic Whole Chicken": 50, "Deboned Chicken": 50}]}

for item in grocery_list:
    print (item)
    for g in grocery_list[item]:
        print (g)
        for i in g:
            print("Item :", i, "Value :", g[i])