#!/usr/bin/python3
# Extract Items From A Dictionary:

grocery_list = {'Fish': [10],
                'Apples': [100, 300],
                'Milk': [1000],
                'Eggs': [100]}

for item in grocery_list:
    for g in grocery_list[item]:
        print("Item :", item, "Value :", g)