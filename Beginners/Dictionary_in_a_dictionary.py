#!/usr/bin/python3

grocery_list = {'Fish': [{"Salmon": 10, "Cuttle Fish": 10}],
                'Apples': [{"Red Apple" : 100, "Granny Smith" : 300}],
                'Milk': [{"Low Fat": 500, "Whole Milk": 500}],
                'Eggs': [{"Organic White": 100, "Organic Brown": 100}],
                'Chicken': [{"Organic Whole Chicken": 50, "Deboned chicken": 50}]}

for item in grocery_list:
    for g in grocery_list[item]:
        print (g)
        for i in g:
            print("Item :", i, "Value :", g[i])