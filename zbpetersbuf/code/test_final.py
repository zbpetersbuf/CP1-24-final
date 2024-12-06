"""
test_final.py
"""
import final as fin
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def crtd_data():
    tme = [1,2,3,4,5,6,7,8,9,10,11,12,13, 14,15,16, 17]
    tme = zip("Time (s)",tme)
    Lat = ["Latitude (°)", 1, 2.01, 3.02, 4.03, 5.04, 6.05, 7.05, 8.06, 9.07, 10.08, 11.09, 12.1, 13.2, 14.21, 15.22, 16.22, 17.23]
    Long = ["Longitude (°)", 1, 2.01, 0.9, 0, 0.9, 2, 1, 0.01, 1, 2.01, 1, 0.01, 1.02, 2, 1, 0, 1]
    alt = ["Altitude WGS84 (m)", 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    return[tme,Lat,Long,alt]

@pytest.fixture
def ypredic():
    return [1,2,1,0,1,2,1,0,1,2,1,0,1,2,1,0,1]

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
    testingmat = fin.stepnumb(crtd_data())

    assert len(testingmat[0]) == len(crtd_data()[0])

def test_adjs_Rsqr():
    yax = fin.stepnumb(crtd_data())[2]# mabye 2 mabye 1
    ypr = ypredic()[1:]
    assert 0<fin.adjs_Rsqr(yax,ypr)<1
    assert 0.9<fin.adjs_Rsqr(ypr,ypr)<=1 


def test_goldrule_sig():
    a = fin.goldrule_sig(crtd_data())
    assert len(a)== len(thisthing[1:])
    assert fin.goldrule_sig(crtd_data(), 2) == "Went over 1,000 iterations"



def test_fftpowerspec():

    assert len(fin.fftpowerspec(crtd_data()))==len(crtd_data()[0])/2

def test_inv_fft():
    thisthing= crtd_data()[2]
    thisthing = thisthing[1:]

    assert fin.inv_fft(np.fft.fft(thisthing)) == thisthing

def test_freqfer():
    thisthing= crtd_data()[2]
    thisthing = thisthing[1:]
    assert fin.freqfer(thisthing,np.fft.fft(thisthing),0) == thisthing
    assert fin.freqfer(thisthing,np.fft.fft(thisthing),10000) == np.zeros(len(thisthing))