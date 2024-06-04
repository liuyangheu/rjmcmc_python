# -*- encoding: utf-8 -*-
'''
@File    :   toolbox.py
@Author  :   Yang Liu (yang.liu3@halliburton.com) 
'''
# import lib below

import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import plotcolor as color
from statsmodels.stats.outliers_influence import summary_table

def modelfunMM(inc0, br0, azi0, wr0, kinc, kazi, deptharray, uinc, upazi, dc):
    mininc = 1

    # convert deg/100ft to deg/ft
    ks_inc = kinc[0]/100
    kr_inc = kinc[1]/100
    
    ks_pazi = kazi[0]/100
    kr_pazi = kazi[1]/100
    
    br0 = br0/100
    wr0 = wr0/100
    
    n = len(deptharray)
    
    # allocate
    yinc = np.ones(n)*999
    br = np.ones(n)*999
    
    yazi = np.ones(n)*999
    wr = np.ones(n)*999
    tr = np.ones(n)*999
    
    # initial conditions
    yinc[0] = inc0
    br[0] = br0
    yazi[0] = azi0
    wr[0] = wr0
    tr[0] = wr0*sind(max(mininc,inc0))
    
    deltad = np.concatenate((np.diff(deptharray), [0]))
    
    for i in range(len(deptharray)-1):
        # inc
        br[i+1] = ks_inc*uinc[i] + kr_inc*(1-dc[i]) # deg/ft
        yinc[i+1] = yinc[i] + br[i+1]*deltad[i]
        
        # azi
        tr[i+1] = ks_pazi*upazi[i] + kr_pazi*(1-dc[i])
        wr[i+1] = tr[i+1]/sind(max(mininc,yinc[i])) # use estimated inc to convert pazi to azi
        yazi[i+1] = yazi[i] + wr[i+1]*deltad[i] # deg
        
    # convert br wr from deg/ft to deg/100ft
    br = br*100
    wr = wr*100
    tr = tr*100
    
    return yinc, br, yazi, wr, tr

def ssfunMMbp(K, survey1, survey2, deptharray, uinc, upazi, tf, dc, ratio, contdata, ksb, krb, prior):

    incthres = 5 # deg. Minimum inc to start calibrating in azi plane
    kinc = K[0:2]
    kazi = K[2:4]
    
    ksi = kinc[0]
    kri = kinc[1]
    ksa = kazi[0]
    kra = kazi[1]
    
    survey_dist = survey2['depth'] - survey1['depth']
    
    # propagate model
    inc0 = survey1['inc']
    br0 = survey1['br']
    azi0 = survey1['azi']
    wr0 = survey1['wr']
    
    yinc,_, yazi,_,_  =  modelfunMM(inc0, br0, azi0, wr0, kinc, kazi, deptharray, uinc, upazi, dc)
    
    ## calculate cost of fitting survey and cont surveys
    #surveys
    cost_inc_survey = yinc[-1] - survey2['inc']
    cost_azi_survey = calAngDif(yazi[-1], survey2['azi'])
    
    if survey2['inc']< incthres:
        cost_azi_survey = 0
    cost_survey = cost_inc_survey**2 + cost_azi_survey**2
    
    # cont surveys
    cost_inc_cont = 0
    if not contdata['inc'].empty:
        continc = {}
        continc['depth'] = contdata['inc']['depth'].values
        continc['val'] = contdata['inc']['val'].values
        
        estinc = interp1(deptharray, yinc, continc['depth'])
        
        m = len(estinc)
        if m>0:
            cost_inc_cont = (1/m)*sum((estinc - continc['val'])**2)

    cost_azi_cont = 0 
    if not contdata['azi'].empty:
        contazi = {}
        contazi['depth'] = contdata['inc']['depth'].values
        contazi['val'] = contdata['inc']['val'].values
        
        estazi = interp1(deptharray, yazi, contazi['depth'] )
        
        m = len(estazi)
        if m>0:
            cost_azi_cont = (1/m)*sum(calAngDif(estazi, contazi['val'])**2)
        
    cost_cont = cost_inc_cont + cost_azi_cont
    wsurcont = 0.5
    cost_data =  cost_survey + cost_cont*wsurcont
    
    # calculate the weight between inc and azi => same as euinc
    a = tf[dc == 1]
    meantf = calAngMean(a)
    
    if sum(dc)!= 0:
        euinc = cosd(meantf)
        # the closer it is to 0.5, the less the penalty
        w = euinc**2
    else:
        w = 0
                
    norms = 1/(ksb[1]-ksb[0]) # for normalizing
    normr = 1/(krb[1]-krb[0])

    wdifincazi = (1/100)*survey_dist
    costdiffinc_azi = np.sum(abs(ksi-ksa) + abs(kri-kra)*normr/norms)*abs(0.5-w)/0.5 #divide by 0.5 to make scale 0 to 1
    
    ##cost for diff from priors
    wdifprior = (1/100)*survey_dist
    
    if len(prior)>0:
        ksi_prior = prior[0]        
        kri_prior = 0 #prior[1]        
        ksa_prior = prior[2]        
        kra_prior = 0 #prior[3]        
        
        # the more we slide the less we penalize kslide from deviating from kslide prior
        costdiffprior_inc = abs(ksi - ksi_prior)*(1-ratio) + abs(kri - kri_prior)*ratio*(normr/norms)
        costdiffprior_azi = abs(ksa - ksa_prior)*(1-ratio) + abs(kra - kra_prior)*ratio*(normr/norms)
        
        costdiffprior  =  abs(costdiffprior_inc) + abs(costdiffprior_azi)
    else:
        costdiffprior = 0 
    
    ## final cost
    cost  =  cost_data + (costdiffinc_azi*wdifincazi)**2 + (costdiffprior*wdifprior)**2
    
    if np.isnan(cost_survey) or (cost_survey < 0):
        print('NaN cost')
        
    return cost


def wrapdatainfo(a):
    
    a = a[~np.isnan(a)]
        
    n = len(a)
    amean = calAngMean(a)
    amedian = atan2d(np.median(sind(a)), np.median(cosd(a)))
    
    diffrommean = calAngDif(a,amean)
    astd = np.sqrt(sum(diffrommean**2)/(n-1))
    ameanad = np.mean(abs(diffrommean))
    
    diffrommedian = calAngDif(a,amedian)
    amedianad = np.median(abs(diffrommedian))
    
    return amean, amedian, astd, ameanad, amedianad

def wrapTF(tf_in, ds, de):

    tfwrapped = []
    tfmean = []
    tfstd = []
    
    tfwrapped = tf_in
    bool = np.logical_and((tf_in[:,0]>= ds),(tf_in[:,0]<= de))
    
    a = wrap(180,tf_in[bool,1])
    amean,_,astd,_,_ = wrapdatainfo(a)
    
    # output
    tfwrapped[bool,1] = a
    tfmean = amean
    tfstd = astd
    
    return tfwrapped, tfmean, tfstd

def simplifyDC(dcarr, ds, de):
    
    summary = {}
    
    # check inputs
    # 1. depthend > depthstart
    if de < ds:
        print('Warning: Ending depth is less than starting depth. Return empty.')
        return summary
    
    dcarr = dcarr[(dcarr[:,0] >= ds) & (dcarr[:,0] <= de)]
 
    # 2. less than 2 data point within depth window
    if len(dcarr) < 2:
        print('Warning: No DC within depth window. Return empty.')
        return summary
    
    depth = dcarr[:,0]
    dc = dcarr[:,1]
    summaryx = summarize_x(dc)

    summary['d1'] = depth[summaryx['i1'].tolist()]
    summary['d2'] = depth[summaryx['i2'].tolist()]
    summary['d2'][0:-1] = summary['d1'][1:]
    summary['dc'] = summaryx['x'].values
    summary['deltad'] = (summary['d2'] - summary['d1']).round(2)

    summary = pd.DataFrame(summary)
    return summary

def getSRinfo(dcarr, tfarr, ds, de):
    
    totalslidelen = []
    totalrotatelen = []
    ratio = []
    summary = []
    
    summary = simplifyDC(dcarr, ds, de)
    
    tfwrapped = tfarr
    # calculate average tf
    summary['tfmean'] = 0
    summary['tfstd'] = 0
    
    for i in range(len(summary)):
        if summary['dc'].iloc[i] == 1:
            d1 = summary['d1'].iloc[i]
            d2 = summary['d2'].iloc[i]
            
            tfwrapped, tfmean, tfstd = wrapTF(tfwrapped, d1, d2)
            
            summary['tfmean'].iloc[i] =  tfmean
            summary['tfstd'].iloc[i] = tfstd
            
    totalslidelen = sum(summary['deltad'][summary['dc'] == 1])
    totalrotatelen = sum(summary['deltad'][summary['dc'] == 0])
    
    ratio = totalslidelen/(totalslidelen + totalrotatelen)    
    
    return totalslidelen, totalrotatelen, ratio, summary, tfwrapped

def getSRratio(survey1, survey2, data):
    
    ratio = []
    deptharray = []
    tfwrapped = []
    dc = []
    tf = []
    summary = []
        
    ds = survey1['depth']
    de = survey2['depth'] 
    
    tempdata = data[(data['depth'] >= ds) & (data['depth'] <= de)] # tf,dc data within the window
    
    deptharray = tempdata['depth'].values
    deptharray = np.append(survey1['depth'], deptharray)
    deptharray = np.append(deptharray, survey2['depth'] )
    deptharray = np.unique(deptharray)
    
    dc = interp1(data['depth'], data['dc'], deptharray, 'nearest', 'extrap')
    tf = interp1(data['depth'], data['tf'], deptharray, 'nearest', 'extrap')
    
    tempdata = pd.DataFrame({'depth': deptharray, 'dc': dc, 'tf' : tf})    
    
    dcarr = np.array([deptharray, dc]).T
    tfarr = np.array([deptharray, tf]).T
    
    # dcarr = np.concatenate((deptharray, dc), axis=1)
    # tfarr = np.concatenate((deptharray, tf), axis=1)
    
    totalslidelen, totalrotatelen, ratio, summary, tfwrapped  = getSRinfo(dcarr, tfarr, ds, de)

    summary['uinc']= summary['dc']*cosd(summary['tfmean'])
    summary['upazi']= summary['dc']*sind(summary['tfmean'])
    
    tfwrapped = tfwrapped[:,1]
    
    return ratio, deptharray, tfwrapped, dc, tf, summary 

def return_same_type(x,y,dtype):
    if dtype == list:
        return y.tolist()
    elif dtype in (int,float):
        return y
    else:
        return y.item() if np.isscalar(x) else y.tolist() if type(x) == list else pd.Series(y) if type(x) == pd.Series else y

def wrap(n:int, x, dtype = None):
    if n not in [180, 360]:
        print("Invalid value for n. n must be 180 or 360. Return output the same as input.")
        return x
    y = np.array(x)
    if n == 180:
        y = np.remainder(y + 180, 360) - 180
    else:
        y = np.remainder(y, 360)
    return return_same_type(x,y,dtype)

def sind(x, dtype = None):
    y = np.sin(np.deg2rad(x))
    return return_same_type(x,y,dtype)
    
def cosd(x, dtype = None):
    y = np.cos(np.deg2rad(x))
    return return_same_type(x,y,dtype)
    
def tand(x, dtype = None):
    y = np.tan(np.deg2rad(x))
    return return_same_type(x,y,dtype)
    
def asind(x, dtype = None):
    y = np.rad2deg(np.arcsin(x))
    return return_same_type(x,y,dtype)
    
def acosd(x, dtype = None):
    y = np.rad2deg(np.arccos(x))
    return return_same_type(x,y,dtype)
    
def atan2d(y,x,dtype = None):
    z = np.rad2deg(np.arctan2(y,x))
    return return_same_type(y,z,dtype)

def calAngDif(x, ref = [], dtype = None):
    """
    Calculate angle difference between a and aref
    :param x: list or array of angles in degrees
    :param ref (optional) : scalar or list or array of angles in degrees. Must have the same size as x. if ref is empty, return angle differeces along x
    :return: angle difference(s) in degrees
    """
    ref = np.array(ref)
    if ref.size != 0:
        a2 = ref
        a1 = x
    else:
        a2 = x[1:]
        a1 = x[0:-1]

    y = np.remainder(np.array(a2) - np.array(a1) + 180,360) - 180
    
    return return_same_type(x,y,dtype)

def calAngMean(x):
    """
    Calculate the average angle of a list or array of angles in degrees.
    :param x: list of angles in degrees
    :return: average angle in degrees
    """
    y = atan2d(np.mean(sind(x)), np.mean(cosd(x)))
    return y

def interp1(x,v,xq, method = 'linear', extrap = None):
    if extrap == 'extrap':
        f = interp1d(x,v, kind = method, fill_value = 'extrapolate')
    else:
        f = interp1d(x,v, kind = method, bounds_error = 0, fill_value = np.nan)   
    vq = f(xq)
    return return_same_type(xq,vq,None)

def interpu(x, v, xq, dtype = None):
    if type(x) in (float,int):
        x = np.array([x])
    else:
        x = np.array(x)
        
    if type(v) in (float,int):
        v = np.array([v])
    else:
        v = np.array(v)
      
    if x.size == 1:
        if type(xq) in (float,int):
            vq = v[0]
        else:
            vq = np.ones(np.array(xq).shape)*v[0]
        return return_same_type(xq,vq,dtype)
    
    vq = interp1(x,v,xq, method='previous', extrap='extrap')
    return vq

def summarize_x(xin):
    if type(xin) in (float,int):
        x = np.array([xin])
    else:
        x = np.array(xin)
    
    summary = pd.DataFrame()
    
    if x.size == 0:
        return summary
    
    switch_points = np.diff(x, axis=0)
    if not switch_points.any() or (switch_points == 0).all(): # no switching
        newrow = {'i1': 0,
                  'i2': x.shape[0]-1,
                  'x': [x[0]]
                  }
        summary = pd.concat([summary, pd.DataFrame(newrow,index = [0])])

    else:
        iarr = np.concatenate(([0], np.where(switch_points != 0)[0] + 1, [x.shape[0]]))
        iarr = np.unique(iarr)
        
        for i in range(len(iarr) - 1):            
            newrow = {'i1': iarr[i],
                      'i2': iarr[i+1]-1,
                      'x': [x[iarr[i]]]
                      }

            summary = pd.concat([summary, pd.DataFrame(newrow,index = [0])])

    summary = summary.reset_index(drop=True)
    
    # if input is dataframe
    if isinstance(xin, pd.DataFrame):
        for col in xin.columns.values:
            summary[col] = xin.loc[summary['i1'],col].reset_index(drop = True)
        summary.drop(columns = 'x', inplace=True)
    
    return summary

def save_to_pickle(file_name: str, workspace: dict, var_list: list[str] = []):

    if var_list:
        # Get a list of all the variables in the current scope that are in the var_list
        variables = {name: value for name,
                     value in workspace.items() if name in var_list}
    else:
        # Get a list of all the variables in the current scope if none are specified
        variables = {name: value for name, value in workspace.items()}

    cwd = os.path.abspath(os.getcwd())  # get current file directory
    filepath = os.path.join(cwd, 'tests', 'test_data', file_name)

    with open(filepath, 'wb') as file:
        # Use pickle to serialize the list of variables into a byte stream
        pickle.dump(variables, file)


def load_pickle(filename: str):
    cwd = os.path.abspath(os.getcwd())  # get current file directory
    filepath = os.path.join(cwd, 'tests', 'test_data', filename)

    with open(filepath, 'rb') as file:
        variables = pickle.load(file)

    return variables

def conffit_linear(x,y, level=0.95, flagplot=0):
    alpha = 1-level
    
    if isinstance(x, pd.core.series.Series):
        x = x.values
    if isinstance(y, pd.core.series.Series):
        y = y.values 
    
    X = sm.add_constant(x)

    # fit model
    model = sm.OLS(y, X)
    result = model.fit() 
    # print(result.summary())
    
    # get fitted model parameters
    modelnom =  np.array([result.params[1], result.params[0]]) # best fit model [slope, yintercept]
    modelci = np.array([result.conf_int(alpha)[1], result.conf_int(alpha)[0]]) # model parameter ci [[slopelb, slopeub], [interceptlb, interceptub]]

    # get cband, pband
    st, data, ss2 = summary_table(result, alpha)

    yfit = data[:,2]
    cbandlb = data[:,4].T
    cbandub = data[:,5].T
    pbandlb = data[:,6].T
    pbandub = data[:,7].T

    if flagplot == 1:
        plt.scatter(x,y, label = 'data')
        plt.plot(x, yfit, '-', color ='r', linewidth=2, label = 'best fit')
        plt.fill_between(x, cbandlb, cbandub, color='#888888', alpha=0.4, label = 'cband') # confidence band
        plt.fill_between(x, pbandlb, pbandub, color='#888888', alpha=0.3, label = 'pband') # prediction band
        plt.grid(True)
        plt.legend()
    
    yspread =  (pbandub-pbandlb).mean()
    return modelnom, modelci, yspread, pbandlb, pbandub

def caltyfromsurvey(survey, steering):
    eps = 0.001
    surveyprior = deepcopy(survey)
    flagsurveyty = 1

    d1 = steering['depth'].iloc[0]
    d2 = steering['depth'].iloc[-1]
    surveyprior = surveyprior[(surveyprior['depth'] >= d1) & (surveyprior['depth'] <= d2)]
    n = surveyprior.shape[0]
    
    if n < 2 :
        print('CANNOT ESTIMATE TOOL YIELD FROM SURVEY. RESORT TO NEXT AVAILABLE PRIOR.')
        flagsurveyty = 0
        return surveyprior, flagsurveyty

    surveyprior['kact'] = 0
    surveyprior['kbiasi'] = 0
    surveyprior['kbiasa'] = 0
    surveyprior['tycertain'] = 0

    stepsize = 1
    depth = np.arange(surveyprior['depth'].iloc[0], surveyprior['depth'].iloc[-1] + eps, stepsize)
    dc = interpu(steering['depth'], steering['dc'], depth)
    ui = steering['dc']*cosd(steering['tf'])
    ua = steering['dc']*sind(steering['tf'])
    ui = interpu(steering['depth'], ui, depth)
    ua = interpu(steering['depth'], ua, depth)

    surveyprior['dc'] = np.NaN
    surveyprior['ui'] = np.NaN
    surveyprior['ua'] = np.NaN

    umin = 0.05 # percent
    # sanity check: calculate survey br/wr/dls instead of using survy inputs
    surveyprior.loc[surveyprior.index[1:],'br']  = 100*np.diff(surveyprior['inc'])/np.diff(surveyprior['depth'])
    surveyprior.loc[surveyprior.index[1:],'wr']  = 100*calAngDif(surveyprior['azi'].values)/np.diff(surveyprior['depth'])
    surveyprior['tr'] = surveyprior['wr']*sind(surveyprior['inc'])
    surveyprior['dls'] = np.sqrt(surveyprior['br']**2 + surveyprior['tr']**2)

    for i in range(len(surveyprior)-1):
                               
        d1 = surveyprior['depth'].iloc[i]
        d2 = surveyprior['depth'].iloc[i+1]
        br = surveyprior['br'].iloc[i+1]
        tr = surveyprior['tr'].iloc[i+1]
        dls = surveyprior['dls'].iloc[i+1]
        
        booli = (depth >= d1) & (depth < d2)
        meandc = np.mean(dc[booli])
        meanui = np.mean(ui[booli])
        meanua = np.mean(ua[booli])
        
        if meandc == 0:
            kbiasi =  br
            kbiasa  =  tr
            kact  =  np.NaN  # can't estimate kact
        else:
            kbiasi  =  0
            kbiasa  =  0
            kact = dls/meandc 

        if meandc < umin:
            tycertain = 0
        else:
            tycertain = 1

        surveyprior.loc[surveyprior.index[i+1],'dc'] = meandc
        surveyprior.loc[surveyprior.index[i+1],'ui'] = meanui
        surveyprior.loc[surveyprior.index[i+1],'ua'] = meanua
        surveyprior.loc[surveyprior.index[i+1],'kact'] = kact
        surveyprior.loc[surveyprior.index[i+1],'kbiasi'] = kbiasi
        surveyprior.loc[surveyprior.index[i+1],'kbiasa']= kbiasa
        surveyprior.loc[surveyprior.index[i+1],'tycertain']= tycertain

    if surveyprior['tycertain'].iloc[-1] == 0 :
        flagsurveyty = 0

    return surveyprior, flagsurveyty

def calculate_rt_ty(rtin, steering, d_window, planestring, surveyprior, flagplot):
    
    mindatapoint = 5 # data points
    mindepthrange = d_window
    eps = 0.001
    rtty = {}
    cal_status = 1
    kbias  = 0 # assume kbias=0

    rt = deepcopy(rtin)
    ds = steering['depth'].iloc[0]
    de = min([steering['depth'].iloc[-1], rt['depth'].iloc[-1]])

    rt = rt[(rt['depth']>=ds) & (rt['depth']<=de)]

    if (len(rt) < mindatapoint) or ((rt['depth'].iloc[-1] - rt['depth'].iloc[0]) < mindepthrange):
        # print('NOT ENOUGH RT DATA TO CALCULATE TOOL YIELD. RESORT TO NEXT AVAILABLE PRIOR.')
        cal_status = 0
        return rtty, cal_status

    stepsize = 1 #ft
    depth = np.arange(ds, de + eps, stepsize)
    surincinterp = interpu(surveyprior['depth'], surveyprior['inc'], depth)

    ui = steering['dc']*cosd(steering['tf'])
    ua = steering['dc']*sind(steering['tf'])
    dc = interpu(steering['depth'], steering['dc'], depth)
    ui = interpu(steering['depth'], ui, depth)
    ua = interpu(steering['depth'], ua, depth)

    if planestring == 'inc':
        u = ui
    else:
        u = ua

    #todo
    # if flagplot == 1:
    #     setplotcolor
    #     figure
    #     s1 = subplot(2,1,1) EditPlot ylabel('inc/azi (°)')
    #     plot(rtin['depth'],rtin.val,'o','color',bl,linewidth = 2,label = 'RT data')
    #     plot(surveyprior['depth'],surveyprior.(planestring),'o',markersize = 10, markerfacecolor = yel,  markeredgecolor = 'k',label = 'survey')
    #     yyaxis right
    #     ylabel('Build/Turn command')
    #     stairs(depth,u,'-',linewidth = 2,'color',yel)
    #     plot([ds de],[0 0],'--',linewidth = 2,'color',[0 0 0]+0.5)
    #     ax = gca
    #     ax.YAxis(2).Color = yel
    #     yyaxis left

    L = d_window
    d2 = de
    d1 = d2 - L

    depthincrement = 5 #ft
    booli = (rt['depth'] >= d1) & (rt['depth'] <= d2)
    x = rt['depth'][booli]
    y = rt['val'][booli]
    
    while (len(x) < mindatapoint) and (d1 > ds):
        d1 = d1 - depthincrement
        booli = (rt['depth'] >= d1) & (rt['depth'] <= d2)
        x = rt['depth'][booli]
        y = rt['val'][booli]

    booli = (depth>=d1) & (depth<d2)
    umean = np.mean(u[booli])
    dcmean = np.mean(dc[booli])
    if (dcmean == 0) or (umean == 0):
        cal_status = 0
        return rtty, cal_status
    curvdepth = d2
    m = len(x)

    # wrap azi
    if planestring == 'azi':
        if np.std(wrap(180,y)) <= np.std(wrap(360,y)) :
            y = wrap(180,y)
        else:
            y = wrap(360,y)

    modelnom, modelci, yspread, _, _ = conffit_linear(x,y,level= 0.95, flagplot=0)
    slope = modelnom[0]
    yintercept = modelnom[1]
    slopelower = modelci[0][0]
    slopeupper = modelci[0][1]
    
    curv = slope*100 # br/wr
    curvupper = slopeupper*100
    curvlower = slopelower*100
    if planestring == 'azi':
        inc = interpu(depth, surincinterp, curvdepth)
        curv = curv*sind(inc) # tr
        curvupper = curvupper*sind(inc)
        curvlower = curvlower*sind(inc)

    kact = curv/umean
    kactupper = max(np.array([curvupper, curvlower])/umean)
    kactlower = min(np.array([curvupper, curvlower])/umean)
    kactspread = kactupper-kactlower

    yend = yintercept + slope*(x.iloc[-1])

    fitlinex = [min(x), max(x)]
    fitliney = [yintercept, yend]

    rtty['depth'] = curvdepth
    rtty['curv'] = curv
    rtty['curvupper'] = curvupper
    rtty['curvlower'] = curvlower
    rtty['u'] = umean
    rtty['u_std'] = np.std(u[booli])
    rtty['dc'] = dcmean
    rtty['dc_std'] = np.std(dc[booli])
    rtty['yspread'] = yspread
    rtty['density'] = (x.iloc[-1] - x.iloc[0])/len(x)
    rtty['kact'] = kact
    rtty['kactupper'] = kactupper
    rtty['kactlower'] = kactlower
    rtty['kbias'] = kbias
    rtty['kactspread'] = kactspread
    rtty['numpoint'] = m

    # %% plot
    #todo
    # if flagplot == 1
    # s1 = subplot(2,1,1) EditPlot
    # plot(fitlinex,wrap(360,fitliney),'-r',linewidth = 3)
    # plot(x, wrap(360,cband(:,1)),'m--')
    # plot(x, wrap(360,cband(:,2)),'m--')
    
    # s2 = subplot(2,1,2) EditPlotylabel('BR/TR (°/100ft)')
    # plot(rtty['depth'], rtty['curv'],'.k',markersize = 20)
    # plot(rtty['depth'], rtty['kact'],'o-r','color',gr,linewidth = 2)
    # plot(rtty['depth'], rtty['kbias'],'o-','color',te,linewidth = 2)
    # yyaxis right
    # ylabel('Build/Turn command')
    # stairs(depth[booli], u[booli\,'--',linewidth = 2,'color',yel) hold on
    # plot([d1 d2],[0 0],'--',linewidth = 2,'color',[0 0 0]+0.5)
    # #     plot(depth[booli],umean*ones(sum(booli),1),'-',linewidth = 2,'color',ora)hold on
    # ax = gca
    # ax.YAxis(2).Color = yel
    # yyaxis left
    
    # linkaxes([s1,s2],'x')
    # xlim([d1-30 d2+10])

    return rtty, cal_status

def selectprior(data_in, prior_option, BHA):
    # prior_option 0- use user input as prior
    #          1- use survey ty as prior
    #          2- use rtty as prior

    # selectprior_message 0-final prior is the same as prior option
    #                 1-prior option survey, calculated survey ty is UNCERTAIN => use user input
    #                 2-prior option RT data, calculated rtty CERTAIN, but >50% different from survey ty (certain)
    #                 3-prior option RT data, calculated rtty UNCERTAIN, survey ty certain => use survey ty
    #                 4-prior option RT data, BOTH rtty, survey ty UNCERTAIN => use user input
    eps = 0.001
    r = 2 # how many digits to round
    print('PRIOR CALCULATION ======')
    data = deepcopy(data_in)
    selectprior_message = 0
    steering = data['steering']
    steering['dc'] = steering['dc']/100
    survey = data['survey']

    userkact = BHA['Kact']
    userkbiasi = BHA['KbiasInc']
    userkbiasa = BHA['KbiasAzi']

    priors = {}
    priors['kact'] = []
    priors['kbiasi'] = []
    priors['kbiasa'] = []
    
    priors['survey'] = {}
    priors['survey']['kact'] = np.NaN
    priors['survey']['kbiasi'] = np.NaN
    priors['survey']['kbiasa'] = np.NaN
    
    priors['rtty'] = {}
    priors['rtty']['kact'] = np.NaN
    priors['rtty']['kbiasi'] = np.NaN
    priors['rtty']['kbiasa'] = np.NaN

    #%% survey ty
    [surveyprior, flagsurveyty] = caltyfromsurvey(survey,steering)

    if not(surveyprior.empty) and ('kact' in surveyprior.keys()):
        priors['survey']['kact'] = round(surveyprior['kact'].iloc[-1],r)         
        priors['survey']['kbiasi'] = round(surveyprior['kbiasi'].iloc[-1],r)
        priors['survey']['kbiasa'] = round(surveyprior['kbiasa'].iloc[-1],r)

    depthlastsurvey = survey['depth'].iloc[-1]
    
    #%% rt ty
    d_window_default = 30 #ft
    d_window_extend = 10

    w_kactspread  = 0.2
    w_yspread     = 0.2
    w_d_window    = 0.1
    w_u           = 0.2
    w_ustd        = 0.1
    w_density     = 0.2

    umin =  0.10 # percent
    yspreadthres = 2 # deg
    kactub = 30 # dls
    kactlb = 0.2 # dls
    kbiasthres = 5
    rtprior_result = {}
    for planestring in ['inc','azi']:
        flagplot = 0 
        kact_temp = np.NaN
        u_temp = 0
        curv_temp = 0
        window_temp = 0
        numpoint_temp = 0
        source_temp = []
        index_best =[]
        yspread_temp = 0
        
        rttyall = pd.DataFrame()
        if data[planestring]:
            for fn in data[planestring]:
                rt = data[planestring][fn][['depth','val']]
                
                if (rt['depth'].iloc[-1] - depthlastsurvey) > d_window_default :
                    # break into several d_windows
                    d_window_max =  rt['depth'].iloc[-1] - depthlastsurvey
                    d_window_all =  np.arange(d_window_default, d_window_max + eps, d_window_extend)
                    
                    # trim data to the last survey
                    rt = rt[rt['depth'] > depthlastsurvey]
                else:
                    d_window_all = np.array([d_window_default])

                for d_window in d_window_all:
                    rtty, cal_status = calculate_rt_ty(rt, steering, d_window, planestring, survey, flagplot)
                    
                    if cal_status!= 0:
                        rtty['source'] = fn
                        rtty['d_window'] = d_window
                        # rttyall = rttyall.append(pd.DataFrame([rtty]))
                        rttyall = pd.concat([rttyall, pd.DataFrame([rtty])], ignore_index=True)


            if not rttyall.empty:
                # sanity check: remove results where kact too small
                rttyall = rttyall[(rttyall['kact'] >= kactlb) & (rttyall['kact'] <= kactub) & (rttyall['yspread'] <= yspreadthres)]

                if not rttyall.empty:
                    # Rank and score
                    rttyall = rttyall.reset_index(drop = True)
                    
                    _, ic = np.unique(rttyall['kactspread'], return_inverse = True)
                    rttyall['kactspeadscore'] = np.max(ic) - ic + 1 # smaller spread, higher score
                    
                    _, ic = np.unique(rttyall['yspread'], return_inverse = True)
                    rttyall['yspreadscore'] = np.max(ic) - ic + 1 # smaller spread, higher score
                    
                    _, ic = np.unique(rttyall['d_window'], return_inverse = True)
                    rttyall['d_windowscore'] = np.max(ic) - ic + 1 # smaller window, higher score
                    
                    _, ic = np.unique(rttyall['u_std'], return_inverse = True)
                    rttyall['u_stdscore'] = np.max(ic) - ic + 1 # smaller std, higher score
                    
                    _, ic = np.unique(abs(rttyall['u']), return_inverse = True)
                    rttyall['uscore'] = ic # bigger u, higher score
                    
                    rttyall['score'] = rttyall['kactspeadscore']*w_kactspread + rttyall['yspreadscore']*w_yspread + rttyall['d_windowscore']*w_d_window +\
                                    rttyall['u_stdscore']*w_ustd + rttyall['uscore']*w_u
                    
                    ind = rttyall['score'].idxmax() # highest score => most certain calculation
                    
                    # take weighted average based on score
                    rttyall['w'] = rttyall['score']/sum(rttyall['score'])
                    
                    kact_temp = sum(rttyall['w'] * rttyall['kact'])
                    u_temp = sum(rttyall['w'] * rttyall['u'])
                    yspread_temp = sum(rttyall['w'] * rttyall['yspread'])
                    curv_temp = sum(rttyall['w'] *rttyall['curv'])
                    window_temp = sum(rttyall['w'] *rttyall['d_window'])
                    index_best = ind
                    numpoint_temp = sum(rttyall['w'] *rttyall['numpoint'])
                    source_temp = rttyall['source'][ind]

        rtprior_result[planestring] = {}
        rtprior_result[planestring]['all'] = rttyall
        rtprior_result[planestring]['kact'] = kact_temp
        rtprior_result[planestring]['u'] = u_temp
        rtprior_result[planestring]['yspread'] = yspread_temp
        rtprior_result[planestring]['curv'] = curv_temp
        rtprior_result[planestring]['d_window'] = window_temp
        rtprior_result[planestring]['index'] = index_best
        rtprior_result[planestring]['numpoint'] = numpoint_temp
        rtprior_result[planestring]['source'] = source_temp

    # select whichever plane has higher u to calculate kact
    if (not np.isnan(rtprior_result['inc']['kact'])) or (not np.isnan(rtprior_result['azi']['kact'])):
        if (not np.isnan(rtprior_result['inc']['kact'])) and (not np.isnan(rtprior_result['azi']['kact'])):
            if abs(rtprior_result['inc']['u']) > abs(rtprior_result['azi']['u']):
                useinc = 1
            else:
                useinc = 0
        elif not np.isnan(rtprior_result['inc']['kact']):
            useinc = 1
        else:
             useinc = 0   
    else:
        useinc = None
                 
    if useinc == None:
        kact_rt = np.NaN     
        kbiasi  = np.NaN  
        kbiasa  = np.NaN
    elif useinc == 1:
        kact_rt = rtprior_result['inc']['kact']
        # calculate bias
        # kbiasa = rtprior_result.azi.curv -  kact_rt * rtprior_result.azi.u
        kbiasa = 0
        # sanity check
        if abs(kbiasa)>kbiasthres:
            kbiasa = 0
        
        kbiasi = 0
        rtprior_result['cal_source'] = rtprior_result['inc']
    elif useinc == 0:
        kact_rt = rtprior_result['azi']['kact']
        kbiasi = rtprior_result['inc']['curv'] -  kact_rt * rtprior_result['inc']['u']
        kbiasi = 0
        # sanity check
        if abs(kbiasi)>kbiasthres:
            kbiasi = 0
            
        kbiasa = 0
        rtprior_result['cal_source'] = rtprior_result['azi']
    else: pass
    
    rtprior_result['kact'] = kact_rt
    rtprior_result['kbiasi'] = kbiasi
    rtprior_result['kbiasa'] = kbiasa

    # if rtprior_result not empty, output for display regardless of certainty
    if not np.isnan(rtprior_result['kact']):
        priors['rtty']['kact'] = round(rtprior_result['kact'],r)
        priors['rtty']['kbiasi'] = round(rtprior_result['kbiasi'],r)
        priors['rtty']['kbiasa'] = round(rtprior_result['kbiasa'],r)
        priors['rtty']['inc'] = rtprior_result['inc']
        priors['rtty']['azi'] = rtprior_result['azi']
        
        rtty = rtprior_result['cal_source']
        
        # check of calculation is certain
        if rtty:
            if (abs(rtty['u']) > umin) and (rtty['kact'] > kactlb) and (rtty['kact'] < kactub) and (rtty['yspread'] < yspreadthres):
                flagrtprior = 1
            else:
                flagrtprior = 0
                
        else:
            flagrtprior = 0
    else:
        flagrtprior = 0
        print('NOT ENOUGH RT DATA TO CALCULATE TOOL YIELD. RESORT TO NEXT AVAILABLE PRIOR.')
        
    # if calculation is certain
    if flagrtprior == 1:
        rtty_final = {}
        rtty_final['kact'] = rtprior_result['kact']
        rtty_final['kbiasi'] = rtprior_result['kbiasi']
        rtty_final['kbiasa'] = rtprior_result['kbiasa']
        
        flagrtty_final= 1
    else:
        flagrtty_final= 0

    # %% output priors

    pdif = 0.5

    if prior_option == 0:
        prior_selected = 0
        selectprior_message = 0
    elif prior_option ==1:
        if flagsurveyty ==1:
            prior_selected = 1
            selectprior_message = 0
        else:
            prior_selected = 0
            selectprior_message = 1        
    elif prior_option == 2:
        if flagrtty_final == 1:
            prior_selected = 2
            selectprior_message = 0
            if flagsurveyty == 1:
                # sanity check with survey ty if available
                if (abs(rtty_final['kact'] - surveyprior['kact'].iloc[-1])/max([rtty_final['kact'], surveyprior['kact'].iloc[-1]]))> pdif:
                    selectprior_message = 2            
        elif flagsurveyty == 1:
            prior_selected = 1
            selectprior_message = 3
        else:
            prior_selected = 0
            selectprior_message = 4

    if prior_selected == 0: # user inputs
        priors['kact'] = round(BHA['Kact'],r)
        priors['kbiasi'] = round(BHA['KbiasInc'],r)
        priors['kbiasa'] = round(BHA['KbiasAzi'],r)
    elif prior_selected == 1: # survey ty
        priors['kact'] = round(surveyprior['kact'].iloc[-1],r)
        priors['kbiasi'] = round(surveyprior['kbiasi'].iloc[-1],r)
        priors['kbiasa'] = round(surveyprior['kbiasa'].iloc[-1],r)
    elif prior_selected == 2: # rt ty
        priors['kact'] = round(rtty_final['kact'],r)
        priors['kbiasi'] = round(rtty_final['kbiasi'],r)
        priors['kbiasa'] = round(rtty_final['kbiasa'],r)

    priors['option'] = prior_option
    priors['selected'] = prior_selected

    # %% display results and messages
    if prior_option == 0:
        prior_option_str = 'user input'
    elif prior_option == 1:
        prior_option_str = 'last survey'
    elif prior_option == 2:
        prior_option_str = 'RT data'

    if     prior_selected == 0:
        prior_select_str = 'user input'
    elif prior_selected == 1:
        prior_select_str = 'last survey'
    elif prior_selected == 2:
        prior_select_str = 'RT data'

    print('prior option    : ', prior_option_str)
    print('prior selected  : ', prior_select_str)

    print('                Kact  kbias_inc  kbias_azi')
    print('User input    = ', round(userkact,r), round(userkbiasi,r), round(userkbiasa,r))
    print('Last survey   = ', round(priors['survey']['kact'], r), round(priors['survey']['kbiasi'],r), round(priors['survey']['kbiasa'],r))
    print('RT data       = ', round(priors['rtty']['kact'],r), round(priors['rtty']['kbiasi'],r), round(priors['rtty']['kbiasa'],r))

    return priors, selectprior_message

def setplotprop(ax):
    ax.grid()
    return ax

def plotcalfitMM(deptharray,inc, azi, uinc, upazi, survey1,survey2, contdata, kinc, kazi,tf,tfnowrap,dc):
    
    n = 180 # wraping angle
    fignum = 1004
    fig, (s1,s2,s3) = plt.subplots(3,1,sharex=True, num=fignum, clear = True)

    # inc
    s1.set_title('Ksi = ' + str(kinc[0].round(2)) + ', Kri = ' + str(kinc[1].round(2)), fontsize = 15, fontweight = 'bold')
    s1 = setplotprop(s1)
    s1.plot(deptharray,inc, color = color.gr, linewidth=3, label='model')
    s1.plot([survey1['depth'], survey2['depth']], [survey1['inc'], survey2['inc']],'.k', markersize=15, label='survey ')

    if not contdata['inc'].empty:
        booli = (contdata['inc']['depth']>= survey1['depth']) & (contdata['inc']['depth']<= survey2['depth'])
        s1.plot(contdata['inc']['depth'][booli], contdata['inc']['val'][booli],'o', fillstyle='none',  label = 'RT survey inc')
    s1.set_ylabel('inc(°)')
    s1ax2 = s1.twinx()
    s1ax2.step(deptharray, uinc,'-', color = color.te, linewidth=3, label='Input')
    s1ax2.plot([deptharray[0], deptharray[-1]], [0,0],'--', color = color.grey, linewidth=1)
    s1ax2.set_ylabel('uinc', color = color.te)
    s1ax2.set_ylim(-1,1)

    # azi
    s2.set_title('Ksi = ' + str(kazi[0].round(2)) + ', Kri = ' + str(kazi[1].round(2)), fontsize = 15, fontweight = 'bold')
    s2 = setplotprop(s2)
    s2.plot(deptharray,azi,'-', color = color.gr, linewidth=3, label='model')
    s2.plot([survey1['depth'], survey2['depth']], [survey1['azi'], survey2['azi']],'.k', markersize=15, label='survey ')

    if not contdata['azi'].empty:
        booli = (contdata['azi']['depth']>= survey1['depth']) & (contdata['azi']['depth']<= survey2['depth'])
        s2.plot(contdata['depthazi'][booli],contdata['azi'][booli],'o', fillstyle='none', label='RT survey azi')
    s2.set_ylabel('azi(°)')
    s2ax2 = s2.twinx()
    s2ax2.step(deptharray, upazi,'-',  color = color.te, linewidth=3, label='Input')
    s2ax2.plot([deptharray[0], deptharray[-1]], [0,0],'--', color = color.grey, linewidth=1)
    s2ax2.set_ylabel('upazi', color = color.te)
    s2ax2.set_ylim(-1,1)
 
    # TF/DC
    s3 = setplotprop(s3)
    s3.plot(deptharray, wrap(n,tfnowrap),'.-',  label='Raw TF')
    s3.plot(deptharray, wrap(n,tf) ,'.-', label='mean TF')
    s3.legend()
    s3.set_ylabel('TF(°)')
    s3ax2 = s3.twinx()
    s3ax2.step(deptharray,dc, 'o-', color = color.yel, label='DC')
    s3ax2.set_ylabel('DC', color = color.yel)
    s3ax2.set_ylim(-1,3)

    plt.show()
    return