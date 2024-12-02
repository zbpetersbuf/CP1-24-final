"""
test_final.py
"""
import final as fin
import numpy as np

def test_dummy():
    """This tests the dummy function"""
    assert fin.dummy(2)==16

def test_fah_to_kel():
    """this tests the function that converts from Fahrenheit to Kelvin"""
    assert fin.fah_to_kel(32)==273.15
    #assert fin.fah_to_kel(-500)==None

def test_find_fah():
    """this tests the function that finds the temp in .md files,
    using the file i made for this unit test, LL13_sinewalktest.md"""
    assert np.all(fin.find_fah('LL13_sinewalktest.md',2)==[30, 29])

def test_filenamelister():
    """this tests the function that finds markdown files related to an experiment name"""
    assert fin.filenamelister('sinewalktest')==['/workspaces/CP1-24-final/zbpetersbuf/data/LL13_sinewalktest.md']
