"""
final.py

just relized that i might have to make sure that the sin i walk in is par
to the 'y' axis or else i will need to remove the slant of the wave
"""
import re
import os
import glob
import numpy as np
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

def filenamelister(exp_name):
    """this function finds and returns all markdown files of an experiment type
    ie if you wanted to find all .md files that relate to the total elevator
    movment experiment enter 'eletot' for exp_name"""

    pattern = os.path.join('/workspaces/CP1-24-final/zbpetersbuf/data/', f"*{exp_name.strip()}*.md")
    md_files = glob.glob(pattern)
    return md_files

def sinfunk(x, a, b, c, d):
    """this is just the fit function for the fit stuff"""
    return a * np.sin(b*x + c) + d

def stepnumb(datta):
    """this just shortens the length of the data to the closest 2^n"""
    tim = zip(datta.loc[:, 'Time (s)'])
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

    return xax[:(2**(n-1))], yax[:(2**(n-1))], tim[:(2**(n-1))]



def analyze_signal(datta, selec_filter=0.1):
    """finds the frequencies"""

    xax = stepnumb(datta)[0]
    yax = stepnumb(datta)[1]

    n = len(xax)
    fs = n/xax
    isfft = np.fft.fft(yax, )
    
    freq = np.fft.fftfreq(n, 1/fs)

    magnitude = np.abs(isfft)
    threshold = selec_filter * np.max(magnitude)
    main_frequencies = freq[magnitude > threshold]

    return freq, main_frequencies



def fitsincuve(datta):

    xax = stepnumb(datta)[0]
    yax = stepnumb(datta)[1]
    #ivsfft main feq
    #ivsfft freq
    

    a, b, c, d = curve_fit(sinfunk, xax, yax)[0]
    sin_fit = sinfunk(np.array(xax), a, b, c, d)
    isfft = np.fft.fft(sin_fit)
    mag = np.abs(np.fft.fft(sin_fit))
    n=len(xax)
    power = np.abs(mag)[:n // 2]
    return isfft, power







"""
def fitsincuve(datta):
   
    #this onle works when the sin wave is parrellel to the x axis
    xax = stepnumb(datta)[0]
    yax = stepnumb(datta)[1]
    #this is a golden rule while loop
    #t = 0
    #error=0
    #while error > 0.1:
    # matrix = [[cos(t), sin(t)],[-sin(t),cos(t)]].*[[xax],[yax]]
    # xnew, ynew = matrix[0], matrix[1]
    # this has to be a 2x2 nmatrix trans, then i
    a, b, c, d = curve_fit(sinfunk, xax, yax)[0]
    sin_fit = sinfunk(np.array(xax), a, b, c, d)
    # error = descreet for minus my fft

    
    isfft = np.fft.fft(sin_fit)
    mag = np.abs(np.fft.fft(sin_fit))
    power = np.abs(mag)[:n // 2]
    return isfft, power
"""



def inv_fft(isfft):

    newthing = np.fft.ifft(isfft)
    return np.abs(newthing)



def freqfinder(datta, use_filter='no', selec_filter=None):
    """this find the frequencies form the data"""
    tim = stepnumb(datta)[2]

    n = len(tim)

    freq = np.fft.fftfreq(n, tim/n)
    magnitude = fitsincuve(datta)[1]

    use_filt = 0
    filt = 0.0
    yes_no = use_filter.strip().lower()
    if yes_no == 'yes':
        filt = selec_filter
        use_filt = 1
        threshold = filt * np.max(magnitude)
        main_frequencies = freq[magnitude > threshold]
        return main_frequencies, use_filt

    return freq, use_filt
