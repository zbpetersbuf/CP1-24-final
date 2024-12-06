"""
test_final.py
"""
import final as fin
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def crtd_data():
    Lat = []
    Long = []
    tme = []
    "Time (s)"
    "Latitude (°)"
    "Longitude (°)"
    "Altitude WGS84 (m)"

def test_fah_to_kel():
    """this tests the function that converts from Fahrenheit to Kelvin"""
    assert fin.fah_to_kel(32)==273.15
    assert fin.fah_to_kel(-500)==None

def test_find_fah():
    """this tests the function that finds the temp in .md files,
    using the file i made for this unit test, LL13_sinewalktest.md"""
    assert np.all(fin.find_fah('LL13_sinewalktest.md')==30)

def test_filenamelister():
    """this tests the function that finds markdown files related to an experiment name"""
    assert fin.filenamelister('sinewalktest')==['/workspaces/CP1-24-final/zbpetersbuf/data/LL13_sinewalktest.md']

def test_sinfunk():
    """this tests the function I made to match a sine wave to my collected data"""
    assert fin.sinfunk(1.5, 2, 2, -3, 5)== 5

def test_stepnumb():
    """this function test to make sure that the function stepnumb correctly checks to see if there are missing data entries and
    to make sure that it shortens the data to the nerrist 2^n"""




[xax,yax]=[1,2,3][0,1]

print(yax)