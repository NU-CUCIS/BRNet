import numpy as np
np.random.seed(1234567)
import random
random.seed(1234567)

from collections import Counter
import re, os, csv, math, operator
import pandas as pd

#Contains 86 elements (Without Noble elements as it does not forms compounds in normal condition)
elements = ['H','Li','Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
            'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu' ]

elements_all = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 
                'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 
                'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
                'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np',
                'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg',
                'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']

# Regex to Choose from Element Name, Number and Either of the Brackets
token = re.compile('[A-Z][a-z]?|\d+|[()]')

# Create a dictionary with the Name of the Element as Key and No. of elements as Value
def count_elements(formula):
    tokens = token.findall(str(formula))
    stack = [[]]
    for t in tokens:
        if t.isalpha():
            last = [t]
            stack[-1].append(t)
        elif t.isdigit():
             stack[-1].extend(last*(int(t)-1))
        elif t == '(':
            stack.append([])
        elif t == ')':
            last = stack.pop()
            stack[-1].extend(last)   
    return dict(Counter(stack[-1]))

#Normalize the Value of the Dictionary
def normalize_elements(dictionary):
    dic_val = sum(dictionary.values()) 
    if dic_val == 0:
        factor = 0
    else:    
        factor=1.0/ dic_val  
        
    for k in dictionary:
        dictionary[k] = dictionary[k]*factor
    return dictionary

def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 

print(Diff(elements_all, elements)) 

def elemental_fraction(dataframe, comp):
    print('The loaded dataset has %d entries'%len(dataframe[comp]))
    compounds = dataframe[comp]

    print('The reduced dataset has %d entries'%len(compounds))
    
    compounds = [count_elements(x) for x in compounds]
    compounds = [normalize_elements(x) for x in compounds]

    in_elements = np.zeros(shape=(len(compounds), len(elements)))
    comp_no = 0

    for compound in compounds:
        keys = compound.keys()
        for key in keys:
            in_elements[comp_no][elements.index(key)] = compound[key]
        comp_no+=1  
    
    data = in_elements
    
    return data

def pre_process_data(data, comp, prop):

    data = data[data[prop].notnull()]

    x_data = data.pop(comp).to_frame()
    y_data = data.pop(prop).to_frame()

    new_x_data = elemental_fraction(x_data, comp)
    new_y_data = np.array(y_data)
    new_y_data.shape = (len(new_y_data),)

    return new_x_data, new_y_data
