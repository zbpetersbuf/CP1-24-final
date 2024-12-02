"""
final.py
"""
import re
import os
import glob
import numpy as np

def dummy(x):
    """this is a dummy file"""
    return (x+2)**2

def fah_to_kel(f):
    """Converts from Fahrenheit to Kelvin enter the recorded temp as f"""
    if f < -459.66:
        print("The value for Fahrenheit should not be possible try again")
        return None
    return (5*(f-32))/9 + 273.15

def find_fah(file_name):
    """this function finds the temp recorded for the file you enter in file_name
    ie if you wanted to know the temp recorded in the test file LL13_sinewalktest.md, 
    you would enter 'LL13_sinewalktest.md' for file_name"""
    fpath='/workspaces/CP1-24-final/zbpetersbuf/data/'.strip()
    filpath = os.path.join(fpath, file_name)
    with open(filpath, 'r', encoding='utf-8') as f:
        ftime = re.findall(r'\d+', f.read())[1], re.findall(r'\d+', f.read())[2]
    return ftime

def filenamelister(exp_name):
    """this function finds and returns all markdown files of an experiment type
    ie if you wanted to find all .md files that relate to the total elevator
    movment experiment enter 'eletot' for exp_name"""
    pattern = os.path.join('/workspaces/CP1-24-final/zbpetersbuf/data/'.strip(), f"*{exp_name}*.md")
    md_files = glob.glob(pattern)
    return md_files

print(find_fah('LL13_sinewalktest.md'))