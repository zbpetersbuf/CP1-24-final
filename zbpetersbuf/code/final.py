"""
final.py

just relized that i might have to make sure that the sin i walk in is par
to the 'y' axis or else i will need to remove the slant of the wave
"""
import re
import os
import glob
import numpy as np
import pandas as pd
import workinh as wrk
from scipy.optimize import curve_fit


def filenamelister(exp_name, filetype='.md'):
    """this function finds and returns all markdown files of an experiment type
    ie if you wanted to find all .md files that relate to the total elevator
    movment experiment enter 'eletot' for exp_name"""

    pattern = os.path.join('/workspaces/CP1-24-final/zbpetersbuf/data/', f"*{exp_name.strip()}*{filetype.strip()}")
    md_files = glob.glob(pattern)
    return md_files


def find_fah(file_name):
    """this function finds the temp recorded for the file you enter in file_name
    ie if you wanted to know the temp recorded in the test file LL13_sinewalktest.md, 
    you would enter 'LL13_sinewalktest.md' for file_name"""

    filpath = os.path.join('/workspaces/CP1-24-final/zbpetersbuf/data/', file_name.strip())
    with open(filpath, 'r', encoding='utf-8') as f:
        numbs = re.findall(r'\d+', f.read())
    return int(numbs[0])

def fah_to_kel(f):
    """Converts from Fahrenheit to Kelvin enter the recorded temp as f"""

    if f < -459.66:
        print("The value for Fahrenheit should not be possible try again")
        return None
    return (5*(f-32))/9 + 273.15

def sinfunk(x, a, b, c, d):
    """this is just the fit function for the fit stuff"""
    return a * np.sin(b*x + c) + d


def fitsincuve(xax,yax):
    a, b, c, d = curve_fit(sinfunk, xax, yax)[0]
    sin_fit = sinfunk(np.array(xax), a, b, c, d)
    isfft = np.fft.fft(sin_fit)
    return isfft



def stepnumb(files):
    """this just shortens the length of the data to the closest 2^n"""
    datta = pd.read_csv(files)
    tim = list(datta.loc[:, 'Time (s)'])
    xax, yax = wrk.gpsloc(datta)
    compare = [len(tim), len(xax), len(yax)]
    alcomp = np.all(compare)
    if not alcomp:
        raise ValueError("Data is not evenly spaced or data points are missing")
        #print("Data is not evenly spaced or data points are missing")
        #return None
    n = 0
    while len(tim) > 2**n:
        n+=1
    remv = int((len(tim) - 2**(n-1)) / 2) 

    # i have to modul this to remove and add a half, ie no half. integers, remove one more from begining then end
    return xax[remv:-remv], yax[remv:-remv], tim[remv:-remv]

def f1(file_name,i=0):
    """file_name to datta"""
    while i<20:
        stepnumb(file_name[i])
    return 

def adjs_Rsqr(yax,y_pred):
    """computs the adjusted r^2 value"""

    residuals = yax-y_pred
    tss = np.sum((yax-np.mean(yax))**2)
    rss = np.sum(residuals**2)
    r2 = 1-(rss/tss)
    adj_r2 = 1-((1-r2)*(len(yax)-1))/(len(yax)-4-1)

    return adj_r2



def ynewfunk(xax,yax, selec_filter=None):
    """this find the frequencies form the data"""

    n = len(xax)
    freq = np.fft.fftfreq(n, xax/n)
    magnitude = fitsincuve(xax,yax)
    threshold = selec_filter * np.max(magnitude)
    ynew = magnitude[magnitude>threshold]

    return ynew

def inv_fft(isfft):
    """ wright the docstring """
    ynew = np.fft.ifft(isfft)
    #ynew = np.abs(ynew)
    return ynew

def goldrule_sig(datta, adjRsqrd=0.8, selec_filter=0.001, filt_int_add=0.01):
    """This outpust the new y axis witch is the sdame asthe fft """
    xax = stepnumb(datta)[0]
    yax = stepnumb(datta)[1]
    adr = adjs_Rsqr(yax,fitsincuve(xax,yax))

    i=0
    fft = ynewfunk(xax,yax, selec_filter)

    while adjRsqrd > adr:
        ynew = inv_fft(fft)
        fft = ynewfunk(xax,ynew, selec_filter)
        adr = adjs_Rsqr(yax,fitsincuve(xax,ynew))
        selec_filter+=filt_int_add
        i+=1

        if i>1000:
            raise ValueError("Went over 1,000 iterations")
            #print("Data is not evenly spaced or data points are missing")
            #return None
    #return ynew, fft, selec_filter
    return fft


def fftpowerspec(fft):
    n = len(fft)
    mag = np.abs(np.fft.fft(fft))
    power = np.abs(mag)[:n // 2]
    return power


# i dont think i need this
def freqfer(ynew,fft, selec_filter=None):
    """this find the frequencies for the data"""

    threshold = selec_filter * np.max(fft)
    frequencies = ynew[fft>threshold]
    #i need to change the units of this to 1/100 m
    return frequencies


