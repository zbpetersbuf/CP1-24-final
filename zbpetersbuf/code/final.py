"""
final.py
"""
import re
import os
import numpy as np

def dummy(x):
    return (x+2)**2

def fah_to_kel(f):
    """Converts from Fahrenheit to Kelvin
    enter the recorded temp as f"""
    if f < -459.66:
        print("The value for Fahrenheit should not be possible try again")
        return None
    return (5*(f-32))/9 + 273.15

def find_fah(file_name):
    fpath='zbpetersbuf/data/'.strip()
    filpath = os.path.join(fpath, file_name)
    with open(filpath, 'r', encoding='utf-8') as f:
        ftime = re.findall(r'\d+', f.read())[0]
    return ftime
