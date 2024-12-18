"""
code from hw 5 that was incorect forn that
"""
import numpy as np

def analyze_signal(signal, tim, use_filter, selec_filter):
    """finds the frequencies"""
    n = len(signal)
    fs = n/tim
    isfft = np.fft.fft(signal)
    freq = np.fft.fftfreq(n, 1/fs)

    use_filt = 0
    filt = 0.0
    yes_no = use_filter.strip().lower()
    if yes_no == 'yes':
        filt = selec_filter
        use_filt = 1

    magnitude = np.abs(isfft)
    threshold = filt * np.max(magnitude)
    main_frequencies = freq[magnitude > threshold]

    return freq, magnitude, main_frequencies, use_filt

def anze_revm_trnd(signal, tim, use_filter, selec_filter):
    """same as analyze_signal but removes overall trend"""
    n = len(signal)
    ss_total = np.sum((signal['y'] - np.mean(signal['y']))**2)


    params_quadratic = np.polyfit(signal['x'], signal['y'], 2)
    y_quad = np.polyval(params_quadratic, signal['x'])

    res_quad = np.sum((signal['y'] - y_quad)**2)
    r_sq_quad = 1 - (res_quad / ss_total)
    r_quad_adj = 1 - ((1 - r_sq_quad) * (n - 1) / (n - 3))


    params_cubic = np.polyfit(signal['x'], signal['y'], 3)
    y_cubic = np.polyval(params_cubic, signal['x'])

    res_cub = np.sum((signal['y'] - y_cubic)**2)
    r_sq_cub = 1 - (res_cub / ss_total)
    r_cub_adj = 1 - ((1 - r_sq_cub) * (n - 1) / (n - 4))


    params_quartic = np.polyfit(signal['x'], signal['y'], 4)
    y_tic = np.polyval(params_quartic, signal['x'])

    res_tic = np.sum((signal['y'] - y_tic)**2)
    r_sq_tic = 1 - (res_tic / ss_total)
    r_tic_adj = 1 - ((1 - r_sq_tic) * (n - 1) / (n - 5))


    log_y = np.log(signal['y'])
    params_linear = np.polyfit(signal['x'], log_y, 1)
    y_linear = np.polyval(params_linear, signal['x'])

    ss_tot_ep = np.sum((log_y - np.mean(log_y))**2)
    ss_residual = np.sum((log_y - y_linear)**2)
    r_sq_ep = 1 - (ss_residual / ss_tot_ep)
    r_ep_adj = 1 - ((1 - r_sq_ep) * (n - 1) / (n - 2))

    find_sml = min(r_quad_adj, r_cub_adj, r_tic_adj, r_ep_adj)

    if find_sml == r_quad_adj:
        signalnew = signal['y'] - y_quad
    if find_sml == r_cub_adj:
        signalnew = signal['y'] - y_cubic
    if find_sml == r_tic_adj:
        signalnew = signal['y'] - y_tic
    else:
        y_ep = (np.exp(params_linear[1])) * np.exp((params_linear[0]) * signal['x'])
        signalnew = signal['y'] - y_ep


    isfft = np.fft.fft(signalnew)
    freq = np.fft.fftfreq(n, tim/n)

    use_filt = 0
    filt = 0.0
    yes_no = use_filter.strip().lower()
    if yes_no == 'yes':
        filt = selec_filter
        use_filt = 1

    magnitude = np.abs(isfft)
    threshold = filt * np.max(magnitude)
    main_frequencies = freq[magnitude > threshold]

    return freq, magnitude, main_frequencies, use_filt





def goldrule_sig(files, adjRsqrd=0.8, selec_filter=0.1, filt_int_add=0.1):
    """This outpust the new y axis witch is the sdame asthe fft """
    datta = pd.read_csv(files)
    xax, yax, tim = stepnumb(datta)

    ynew = fitsincuve(xax,yax)
    adr1 = adjs_Rsqr(yax,ynew)
    thing = np.fft.fft(yax)

    #a = (np.max(yax) + np.min(yax))/2

    #fft = ynewfunk(thing, selec_filter)

    return yax, ynew, xax, adr1
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



def ynewfunk(magnitude, selec_filter):
    """this find the frequencies form the data"""

    threshold = selec_filter * np.max(magnitude)
    nfft = magnitude[magnitude>threshold]
    return nfft