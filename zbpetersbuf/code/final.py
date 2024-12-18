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

    filpath = os.path.join('/workspaces/CP1-24-final/zbpetersbuf/data/', file_name.strip())
    with open(filpath, 'r', encoding='utf-8') as f:
        numbs = re.findall(r'\d+', f.read())
    return int(numbs[0])


def filenamelister(exp_name, filetype = '.md'):
    """this function finds and returns all markdown files of an experiment type
    ie if you wanted to find all .md files that relate to the total elevator
    movment experiment enter 'eletot' for exp_name"""

    pattern = os.path.join('/workspaces/CP1-24-final/zbpetersbuf/data/', f"*{exp_name.strip()}*{filetype.strip()}")
    files = glob.glob(pattern)

    return files


def findmdfromcsv(filepath,i):
    match = re.search(r'LL\d+_(.*)\.csv', filepath)
    file_name = match.group(1)
    return f"{i}{file_name}.md"

def findmdfromcsv2(filepath):
    match = re.search(r'LL\d+_(.*)\.csv', filepath)
    file_name = match.group(1)
    return f"_{file_name}"

def sinfunk(x, a, b, c, d):
    """this is just the fit function for the fit stuff"""
    return a * np.sin(b*x - b*c) + d*x


def fitsincuve(xax, yax, i=0):
    # Ensure that yax is real if it should be
    if not len(xax) == len(yax):
        print(xax, yax, i)
    yax = np.array(yax)

    guess_a = (np.abs(np.max(yax) - np.min(yax)))
    guess_b = 4 * np.pi/(xax[-1] - xax[0])
    guess_c = 1
    guess_d = (xax[0]-xax[len(xax)-1])/((yax[0]-yax[len(xax)-1]))*(sum(yax)/sum(xax))
    p0 = [guess_a, guess_b, guess_c, guess_d]

    a, b, c, d = curve_fit(sinfunk, xax, yax, p0=p0, maxfev=50000)[0]
    sin_fit = sinfunk(np.array(xax), a, b, c, d)

    return sin_fit



def stepnumb(datta):
    """this just shortens the length of the data to the closest 2^n"""

    tim = list(datta.loc[:, 'Time (s)'])
    xax, yax = wrk.gpsloc(datta)
    compare = [len(tim)==len(xax), len(tim)==len(yax), len(xax)==len(yax)]
    alcomp = np.all(compare)
    if not alcomp:
        print("Data is not evenly spaced or data points are missing")
        return None
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

    #return xax-xax[0], yax-yax[0], tim
    mintim = np.min([tim[0],tim[len(tim)-1]])
    if mintim == tim[0]:
        return xax-xax[0], yax-yax[0], tim
    else:
        return xax-xax[len(tim)-1], yax-yax[len(tim)-1], tim


def adjs_Rsqr(yax,ypred):
    """computs the adjusted r^2 value"""

    residuals = yax-ypred
    tss = np.sum((yax-np.mean(yax))**2)
    rss = np.sum(residuals**2)
    r2 = 1-(rss/tss)
    adj_r2 = 1-((1-r2)*(len(yax)-1))/(len(yax)-5)

    return adj_r2


def funfit(file):
    datta = pd.read_csv(file)
    xax, yax, tim = stepnumb(datta)

    ynew = fitsincuve(xax,yax)
    adr = adjs_Rsqr(yax,ynew)

    return yax, ynew, xax, adr, tim


def fftfinding(ynew, adr, tim, mag='False'):
    n = len(tim)
    for i in range(n-1):
        timestamp_sum = sum(tim.index[i+1].timestamp() - tim.index[i].timestamp())
    compare = np.isclose(timestamp_sum/(n-1), tim.index[2].timestamp() - tim.index[1].timestamp(), atol=1e-5)
    if compare:
        if adr > 0.2:
            if mag == 'True':
                matrx = np.fft.fft(ynew.values)
                return np.abs(matrx)[:n // 2]
            return np.fft.fft(ynew)
    return None

def inv_fft(isfft, real='False'):
    """ wright the docstring """
    if real:
        return np.abs(np.fft.ifft(isfft))
    else:
        return np.fft.ifft(isfft)

def freqfinder(ynew, xax):
    n = len(ynew)
    d = sum(xax[i+1] - xax[i] for i in range(n-1)) / (n - 1)
    freqs = []
    for k in range(n):
        f = k / (n * d)*100
        freqs.append(f)

    return freqs



def calc_freq(data, tim):
    """this takes in the same data as the fft equations only gives the frequencies
    of the data, this gives out the frequencies in Hz, if you want to change it to
    days say day in the second imput, or if you want it in months say month (ie 365.25/12 days) """
    n = len(data)
    timestamp_sum = sum(data.index[i+1].timestamp() - data.index[i].timestamp() for i in range(n-1))
    diftim = data.index[2].timestamp() - data.index[1].timestamp()
    if not timestamp_sum/(n-1) == diftim:
        print("Data is not evenly spaced or data points are missing")
        return None
    if tim.strip().lower() == 'day':
        diftim = diftim/(60*60*24)
    if tim.strip().lower() == 'month':
        diftim = diftim/(60*60*24*30.4375)
    return np.fft.fftfreq(n, d = diftim)
