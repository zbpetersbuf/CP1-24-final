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


def filenamelister(exp_name, filetype = '.md'):
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

def sinfunk(x, a, b, c, d, e):
    """this is just the fit function for the fit stuff"""
    return a * np.sin(b*x + c) + d*x + e



def fitsincuve(xax, yax, i=0):
    # Ensure that yax is real if it should be
    if not len(xax) == len(yax):
        print(xax, yax, i)
    yax = np.array(yax)
    guess_a = np.max(yax) - np.min(yax)
    guess_b = 3*np.pi/(np.max(xax) - np.min(xax))
    guess_c = xax[0]
    guess_d = np.mean(yax)
    p0 = [guess_a, guess_b, guess_c, 1, guess_d]

    a, b, c, d, e = curve_fit(sinfunk, xax, yax, p0=p0, maxfev=50000)[0]
    sin_fit = sinfunk(np.array(xax), a, b, c, d, e)

    return sin_fit




def stepnumb(datta):
    """this just shortens the length of the data to the closest 2^n"""

    tim = list(datta.loc[:, 'Time (s)'])
    xax, yax = wrk.gpsloc(datta)
    compare = [len(tim)==len(xax), len(tim)==len(yax), len(xax)==len(yax)]
    alcomp = np.all(compare)
    if not alcomp:
        raise ValueError("Data is not evenly spaced or data points are missing")

    n = 0
    while len(tim) > 2**n:
        n+=1
    twon = 2**(n-1)
    remv = (len(tim) - twon) // 2 
    remv1 = remv
    if abs(remv - (len(tim) - twon) / 2) == 0.5:
        remv1 = remv + 1 

    xax = xax[remv1:-remv]
    yax = yax[remv1:-remv]
    tim = tim[remv1:-remv]
    xminus = np.min([xax[0],xax[len(tim)-1]])
    #yminus = np.min([yax[0],yax[len(tim)-1]])
    return xax-xminus, yax-yax[0], tim



def f1(file_name,i=0):
    """file_name to datta"""
    while i<20:
        stepnumb(file_name[i])
    return 

def adjs_Rsqr(yax,ypred):
    """computs the adjusted r^2 value"""

    residuals = yax-ypred
    tss = np.sum((yax-np.mean(yax))**2)
    rss = np.sum(residuals**2)
    r2 = 1-(rss/tss)
    adj_r2 = 1-((1-r2)*(len(yax)-1))/(len(yax)-5)

    return adj_r2


def ynewfunk(magnitude, selec_filter):
    """this find the frequencies form the data"""

    threshold = selec_filter * np.max(magnitude)
    nfft = magnitude[magnitude>threshold]
    return nfft


def inv_fft(isfft):
    """ wright the docstring """
    ynew = np.fft.ifft(isfft)
    return ynew

def goldrule_sig(files, adjRsqrd=0.8, selec_filter=0.1, filt_int_add=0.1):
    """This outpust the new y axis witch is the sdame asthe fft """
    datta = pd.read_csv(files)
    xax, yax, tim = stepnumb(datta)
    i=0
    j=2
    ynew = fitsincuve(xax,yax,j)
    adr1 = adjs_Rsqr(yax,ynew)
    thing = np.fft.fft(yax)

    a = (np.max(yax) + np.min(yax))/2

    fft = ynewfunk(thing, selec_filter)
    print(adr1)
    return yax, ynew, xax
    #while adjRsqrd > adr:
    """
    adr2 = 100
    while np.abs(adr2-adr1) > 0:
        adr2 = adr1

        rry =  np.zeros(len(tim))
        ynew1 =  np.abs(inv_fft(fft))
        rry[:len(ynew1)] = ynew1

        ynewfit = fitsincuve(xax,rry,i)
        fft = ynewfunk(thing, selec_filter)
        adr1 = adjs_Rsqr(rry,ynewfit)
        selec_filter+=filt_int_add
        i+=1

        if i>1:
            print(adr1)
            return yax, ynewfit, rry, xax
            #raise ValueError("Went over 1,000 iterations")

    print(adr1,i)
    return yax, ynewfit, rry, xax"""


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

