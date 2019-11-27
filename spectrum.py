from matplotlib import rcParams, rc
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from brokenaxes import brokenaxes
#plt.style.use(u"trygveplot_astro")

params = {'savefig.dpi'        : 300, # save figures to 300 dpi
          'ytick.major.size'   : 6,
          'ytick.minor.size'   : 3,
          'xtick.major.size'   : 6,
          'xtick.minor.size'   : 3,
          'xtick.top'          : False,
          'ytick.right'        : True, #Set to false
          'axes.spines.top'    : True, #Set to false
          'axes.spines.bottom' : True,
          'axes.spines.left'   : True,
          'axes.spines.right'  : True, #Set to false
          'axes.grid.axis'     : 'y',
          'axes.grid'          : False
          }

rcParams.update(params)

h    = 6.62607e-34 # Planck's konstant
k_b  = 1.38065e-23 # Boltzmanns konstant
Tcmb = 2.7255      # K CMB Temperature

def cmb(nu, A):
    x = h*nu/(k_b*Tcmb)
    g = (np.exp(x)-1)**2/(x**2*np.exp(x))
    s_cmb = A/g
    return s_cmb

def sync(nu, As, alpha, nuref=0.408):
    #alpha = 1., As = 30 K (30*1e6 muK)
    nu_0 = nuref*1e9 # 408 MHz
    fnu, f = np.loadtxt("Synchrotron_template_GHz_extended.txt", unpack=True)
    f = np.interp(nu, fnu*1e9, f)
    f0 = np.interp(nu_0, nu, f) # Value of s at nu_0
    s_s = As*(nu_0/nu)**2*f/f0
    return s_s


def ffEM(nu,EM,Te):
    #EM = 1 cm-3pc, Te= 500 #K
    T4 = Te*1e-4
    nu9 = nu/1e9 #Hz
    g_ff = np.log(np.exp(5.960-np.sqrt(3)/np.pi*np.log(nu9*T4**(-3./2.)))+np.e)
    tau = 0.05468*Te**(-3./2.)*nu9**(-2)*EM*g_ff
    s_ff = 1e6*Te*(1-np.exp(-tau))
    return s_ff

def ff(nu,A,Te, nuref=40.):
    nu_ref = nuref*1e9
    S =     np.log(np.exp(5.960 - np.sqrt(3.0)/np.pi * np.log(    nu/1e9*(Te/1e4)**-1.5))+2.71828)
    S_ref = np.log(np.exp(5.960 - np.sqrt(3.0)/np.pi * np.log(nu_ref/1e9*(Te/1e4)**-1.5))+2.71828)
    s_ff = A*S/S_ref*np.exp(-h*(nu-nu_ref)/k_b/Te)*(nu/nu_ref)**-2
    
    return s_ff

def sdust(nu, Asd, nu_p, nuref=22.):
    nu_ref = nuref*1e9
    nu_p0 = 30.*1e9
    
    fnu, f = np.loadtxt("spdust2_cnm.dat", unpack=True)
    fnu *= 1e9
    # MAKE SURE THAT THESE ARE BOTH IN 1e9
    scale = nu_p0/nu_p
    
    f = np.interp(scale*nu, fnu, f)
    f0 = np.interp(scale*nu_ref, scale*nu, f) # Value of s at nu_0
    s_sd = Asd*(nu_ref/nu)**2*f/f0
    return s_sd


def tdust(nu,Ad,betad,Td,nuref=545.):
    nu0=nuref*1e9
    gamma = h/(k_b*Td)
    s_d=Ad*(nu/nu0)**(betad+1)*(np.exp(gamma*nu0)-1)/(np.exp(gamma*nu)-1)
    return s_d

def lf(nu,Alf,betalf):
    return Alf*(nu)**(betalf)


# ---- Calculating spectra ----
pol = False
long = True

N = 1000
nu    = np.logspace(np.log10(0.1),np.log10(5000),N) #Text scaled to 0.2, 5000

## FIND MIN MAX
def findminmax(nu, function, numparams, range1=None,range2=None,range3=None, range4=None):
    vals = np.zeros((2, len(nu)))
    val = []
    for i in range1:
        if numparams >= 2:
            for j in range2:
                if numparams >= 3:
                    for k in range3:
                        if numparams == 4:
                            for l in range4:                                
                                val.append(function(nu, i, j,k,l))
                        else:
                            val.append(function(nu, i, j, k))
                else:
                    val.append(function(nu, i, j))
        else:
            val.append(function(nu, i))

    val = np.array(val)
    vals[0,:] = np.min(val,axis=0)
    vals[1,:] = np.max(val,axis=0)
    return vals

def mb(range, n=5):
    return np.linspace(range[0]-range[1],range[0]+range[1], 10)

cmb_range = [67,1]
te_range = [7000, 11]
EM_range = [30,5]#[13, 1]

ame1_a = [92,1]#[92,118]
ame2_a = [17,1]#[17,22]
ame_nu = [19,1] #[19,1]

ame1_a = [50,5]#[92,118]
ame2_a = [50,5]#[17,22]
ame_nu = [22.8e9, 1e9] #[19,1]

dust_a = [163,30] #[163,228]
dust_t = [21,2]
dust_b = [1.51, 0.05]

sync_a = [20*1e6,1*1e6] #[20,15]

cmb_pol = [0.67,0.03]
sync_pol = [12, 1] #[12,9]
dust_pol = [8,1] #[8,10]

CMB = findminmax(nu*1e9, cmb, numparams=1, range1=mb(cmb_range)) # 70
FF    = findminmax(nu*1e9, ff,numparams=2, range1=mb(EM_range),range2=mb(te_range)) #, 30., 7000.)
SYNC  = findminmax(nu*1e9, sync, numparams=2, range1=mb(sync_a), range2=[1]) # 30.*1e6,1.)
TDUST = findminmax(nu*1e9, tdust, numparams=4, range1=mb(dust_a), range2=mb(dust_b), range3=mb(dust_t), range4=[545.]) # 163, 1.6,21.)

SDUST1 = findminmax(1.5*nu*1e9, sdust, numparams=2, range1=mb(ame1_a),range2=[41e9])#,  (1.5*nu*1e9, 50, 41e9)+
SDUST2 = findminmax(0.9*nu*1e9, sdust, numparams=2, range1=mb(ame2_a), range2=mb(ame_nu))#,  sdust(0.9*nu*1e9, 50, 22.8e9)
SDUST = SDUST1+SDUST2

# REFERANGE?
"""
CMB   = cmb(  nu*1e9, 70)
FF    = ff(   nu*1e9, 30., 7000.)
SYNC  = sync( nu*1e9, 30.*1e6,1.)
SDUST = sdust(1.5*nu*1e9, 50, 41e9)+sdust(0.9*nu*1e9, 50, 22.8e9)
TDUST = tdust(nu*1e9, 163, 1.6,21.)
"""

# ---- Figure parameters ----
baralpha= 0.3
ratio=5
if long:
    xmin=0.3
    xmax=4000
    ymin=0.05
    ymax=7e2
    ymax2=1e8#ymax+1e8
    ymax15=1000#ymax+500

    fig, (ax2, ax) = plt.subplots(2,1,sharex=True,figsize=(16,8),gridspec_kw = {'height_ratios':[1, ratio]})
    ax2.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.tick_params(labelbottom=False)
    ax2.xaxis.set_ticks_position('none')

    # ---- Adding broken axis lines ----
    d = .005  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax2.plot((-d, +d), (-d*ratio, + d*ratio), **kwargs)        # top-left diagonal
    ax2.plot((1 - d, 1 + d), (-d*ratio, +d*ratio), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
    ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

else:
    xmin=10
    xmax=1000
    ymin=0.05
    ymax=7e2
    ymax2=ymax
    ymax15=ymax

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    ax2 = ax

freqtext = 16
fgtext = 16
labelsize = 18
ticksize = 16


if pol:
    CMB = findminmax(nu*1e9, cmb, numparams=1, range1=mb(cmb_pol)) # 70
    SYNC  = findminmax(nu*1e9, sync, numparams=3, range1=mb(sync_pol), range2=[1], range3=[30.]) # 30.*1e6,1.)
    TDUST = findminmax(nu*1e9, tdust, numparams=4, range1=mb(dust_pol), range2=mb(dust_b), range3=mb(dust_t),range4=[353.]) # 163, 1.6,21.)

    #CMB   = cmb(  nu*1e9, 0.67)
    #SYNC  = sync( nu*1e9, 12,1., nuref=30.)
    #TDUST = tdust(nu*1e9, 8, 1.51,21.,nuref=353. )
    
    sumf = SYNC+TDUST
    fgs=[CMB,SYNC,TDUST]
    col=["C9","C2","C3","C7"]
    label=["CMB", "Synchrotron","Thermal Dust", "Sum fg."]
    if long:
        rot=[-20, -50,22, -10] #mid
        idx=[70, 53,  115, -15]
        scale=[0.05, 0, 11, 2]
    else:
        rot=[-20, -45, 18, -10] #Regular
        idx=[70, 53,  115, -15]
        scale=[0.05, 0, 7, 2]

else:
    sumf = FF+SYNC+SDUST+TDUST
    fgs=[CMB,FF,SYNC,SDUST,TDUST]   
    col=["C9","C0","C2","C1","C3","C7"]
    label=["CMB","Free-Free","Synchrotron","Spinning Dust","Thermal Dust", "Sum fg."] 
    if long:
        rot=[-8, -40, -50, -73, 13, -45]
        idx=[17, 50, 50, -10, 160, -90]
        scale=[5,0,0,0,200,300]
    else:  
        rot=[-8, -35, -45, -70, 13, -40] #Regular
        idx=[17, 60, 60, -10, 160, -90]
        scale=[5,0,0,0,150,300]


idxshift = 600
idx = [x + idxshift for x in idx]


haslam = True
chipass = True
spass = True
cbass = True
quijote = False
wmap = True
planck = True
dirbe = True


# ---- Foreground plotting parameters ----

#scale=[5,105,195] # Scaling CMB, thermal dust and sum up and down


ax.loglog(nu,sumf[0], "--", linewidth=2, color='k', alpha=0.7)
ax.loglog(nu,sumf[1], "--", linewidth=2, color='k', alpha=0.7)
ax2.loglog(nu,sumf[0], "--", linewidth=2, color='k', alpha=0.7)
ax2.loglog(nu,sumf[1], "--", linewidth=2, color='k', alpha=0.7)
# ---- Plotting foregrounds and labels ----
j=0
for i in range(len(fgs)):
    ax.fill_between(nu, fgs[i][0], fgs[i][1], color=col[i])
    ax2.fill_between(nu, fgs[i][0], fgs[i][1], color=col[i])

    #ax.loglog(nu,fgs[i], linewidth=4,color=col[i])
    #ax2.loglog(nu,fgs[i], linewidth=4,color=col[i])
    ax.text(nu[idx[i]], fgs[i][1,idx[i]]+scale[i], label[i], rotation=rot[i], color=col[i],fontsize=fgtext)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

# ---- Plotting sum of all foregrounds ----        
ax.text(nu[idx[-1]], fgs[-1][1,idx[-1]]+scale[-1], label[-1], rotation=rot[-1], color='k', fontsize=fgtext, alpha=0.7)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#ax.text(10, find_nearest(fgs[-1], 10), label[-1], rotation=rot[-1], color=col[-1],fontsize=fgtext)

if not pol:
    # ---- Plotting CO lines ----
    ax.axvline(x=115., color='k', alpha=0.7)
    ax.axvline(x=230., color='k', alpha=0.7)
    ax.axvline(x=345., color='k', alpha=0.7)
    if long:
        ax2.axvline(x=115., color='k', alpha=0.7)
        ax2.axvline(x=230., color='k', alpha=0.7)
        ax2.axvline(x=345., color='k', alpha=0.7)

    ax2.text(115.,ymax2, "CO 1-0 ", color='k', alpha=0.7, ha='right',va='top',rotation=90,fontsize=fgtext)
    ax2.text(230.,ymax2, "CO 2-1 ", color='k', alpha=0.7, ha='right',va='top',rotation=90,fontsize=fgtext)
    ax2.text(345.,ymax2, "CO 3-2 ", color='k', alpha=0.7, ha='right',va='top',rotation=90,fontsize=fgtext)

# ---- Data band ranges ----
band_range1   = [.406,.410]      # Haslam?
band_range2   = [2.1,2.4]      # Spass?
band_range3   = [21,25.5]        # WMAP K
band_range4   = [30,37]          # WMAP Ka
band_range5   = [38,45]          # WMAP Q
band_range6   = [54,68]          # WMAP V
band_range7   = [84,106]         # WMAP W
band_range8   = [23.9,34.5]      # Planck 30
band_range9   = [39,50]          # Planck 44
band_range10  = [60,78]          # Planck 70
band_range11  = [82,120]         # Planck 100
band_range12  = [125,170]        # Planck 143
band_range13  = [180,265]        # Planck 217
band_range14  = [300,430]        # Planck 353
band_range15  = [450,650]        # Planck 545
band_range16  = [700,1020]       # Planck 857
band_range17  = [1000,1540]      # DIRBE 1250
band_range18  = [1780,2500]      # DIRBE 2140
band_range19  = [2600,3500]      # DIRBE 3000
band_range20  = [4.,6.]      # C-BASS
band_range21  = [10.,12.]      # QUIJOTE
band_range22  = [12.,14.]      # QUIJOTE
band_range23  = [16.,18.]      # QUIJOTE
band_range24  = [18.,21.]      # QUIJOTE
band_range25  = [26.,36.]      # QUIJOTE
band_range26  = [35.,47.]      # QUIJOTE
band_range27  = [1.3945-0.064/2,1.3945+0.064/2]  #CHIPASS


# ---- Plotting single data ----
if long:
    if haslam and not pol:
        ax2.text(np.mean(band_range1),ymax2-0.2,"Haslam \n 0.408",color='purple',va='bottom',horizontalalignment='center', size = freqtext)
        ax.axvspan(band_range1[0],band_range1[1],color='purple',alpha=baralpha, zorder=0,label="Haslam")
        ax2.axvspan(band_range1[0],band_range1[1],color='purple',alpha=baralpha, zorder=0)

    if spass:
        ax2.text(np.mean(band_range2)+.1 ,ymax2-0.2," S-PASS \n 2.303",color='green',va='bottom',horizontalalignment='center', size = freqtext)
        ax.axvspan(band_range2[0],band_range2[1],color='green',alpha=baralpha, zorder=0, label="S-PASS")
        ax2.axvspan(band_range2[0],band_range2[1],color='green',alpha=baralpha, zorder=0)

    if cbass:
        ax2.text(np.mean(band_range20),ymax2-0.2,"C-BASS \n 5.0",color='C0',va='bottom',horizontalalignment='center', size = freqtext)
        ax.axvspan(band_range20[0],band_range20[1],color='C0',alpha=baralpha, zorder=0, label="C-BASS")
        ax2.axvspan(band_range20[0],band_range20[1],color='C0',alpha=baralpha, zorder=0)

    if chipass and not pol:
        ax2.text(np.mean(band_range27)-0.1,ymax2-0.2,"CHIPASS \n 1.394",color='C5', va='bottom',horizontalalignment='center', size = freqtext)
        ax.axvspan(band_range27[0],band_range27[1],color='C5',alpha=baralpha, zorder=0,label='CHIPASS')
        ax2.axvspan(band_range27[0],band_range27[1],color='C5',alpha=baralpha, zorder=0)


# ---- Plotting QUIJOTE ----
if quijote:
    ax2.text(11,ymax2-0.2,"QUIJOTE \n 11",  color='C4', va='bottom',alpha=1,horizontalalignment='center', size = freqtext)
    ax2.text(13,ymax2-0.2,"13",  color='C4', va='bottom',alpha=1,horizontalalignment='center', size = freqtext)
    ax2.text(17,ymax2-0.2,"17",  color='C4', va='bottom',alpha=1,horizontalalignment='center', size = freqtext)
    ax2.text(19+1,ymax2-0.2,"19", color='C4', va='bottom',alpha=1,horizontalalignment='center', size = freqtext)
    ax2.text(31,ymax2-0.2,"31",color='C4', va='bottom',alpha=1,horizontalalignment='center', size = freqtext)
    ax2.text(41+1,ymax2-0.2,"41",color='C4', va='bottom',alpha=1,horizontalalignment='center', size = freqtext)

    ax.axvspan(band_range21[0], band_range21[1], color='C4',alpha=baralpha, zorder=0, label="QUIJOTE")
    ax.axvspan(band_range22[0], band_range22[1], color='C4',alpha=baralpha, zorder=0)
    ax.axvspan(band_range23[0], band_range23[1], color='C4',alpha=baralpha, zorder=0)
    ax.axvspan(band_range24[0], band_range24[1], color='C4',alpha=baralpha, zorder=0)
    ax.axvspan(band_range25[0], band_range25[1], color='C4',alpha=baralpha, zorder=0)
    ax.axvspan(band_range26[0], band_range26[1], color='C4',alpha=baralpha, zorder=0)
    if long:
        ax2.axvspan(band_range21[0],band_range21[1], color='C4',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range22[0],band_range22[1], color='C4',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range23[0],band_range23[1], color='C4',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range24[0],band_range24[1], color='C4',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range25[0],band_range25[1], color='C4',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range26[0],band_range26[1], color='C4',alpha=baralpha, zorder=0)

# ---- Plotting Planck ----
if planck:
    ax2.text(27-2,ymax2-0.2,"30",  color='C1', va='bottom',alpha=1, size = freqtext)
    ax2.text(42+2,ymax2-0.2,"44",  color='C1', va='bottom',alpha=1, size = freqtext)
    ax2.text(64,  ymax2-0.2,"Planck \n 70",  color='C1', va='bottom',alpha=1, size = freqtext)
    ax2.text(90+5,ymax2-0.2,"100", color='C1', va='bottom',alpha=1, size = freqtext)
    ax2.text(135, ymax2-0.2,"143", color='C1', va='bottom',alpha=1, size = freqtext)
    ax2.text(200, ymax2-0.2,"217", color='C1', va='bottom',alpha=1, size = freqtext)
    ax2.text(330, ymax2-0.2,"353", color='C1', va='bottom',alpha=1, size = freqtext)

    ax.axvspan(band_range8[0] ,band_range8[1], color='C1',alpha=baralpha, zorder=0, label="Planck")
    ax.axvspan(band_range9[0] ,band_range9[1], color='C1',alpha=baralpha, zorder=0)
    ax.axvspan(band_range10[0],band_range10[1],color='C1',alpha=baralpha, zorder=0)
    ax.axvspan(band_range11[0],band_range11[1],color='C1',alpha=baralpha, zorder=0)
    ax.axvspan(band_range12[0],band_range12[1],color='C1',alpha=baralpha, zorder=0)
    ax.axvspan(band_range13[0],band_range13[1],color='C1',alpha=baralpha, zorder=0)
    ax.axvspan(band_range14[0],band_range14[1],color='C1',alpha=baralpha, zorder=0)

    if long:
        ax2.axvspan(band_range8[0] ,band_range8[1], color='C1',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range9[0] ,band_range9[1], color='C1',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range10[0],band_range10[1],color='C1',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range11[0],band_range11[1],color='C1',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range12[0],band_range12[1],color='C1',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range13[0],band_range13[1],color='C1',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range14[0],band_range14[1],color='C1',alpha=baralpha, zorder=0)

    if not pol:
        ax2.text(490, ymax2-0.2,"545", color='C1', va='bottom',alpha=1, size = freqtext)
        ax2.text(770, ymax2-0.2,"857", color='C1', va='bottom',alpha=1, size = freqtext)
        ax.axvspan(band_range15[0],band_range15[1],color='C1',alpha=baralpha, zorder=0)
        ax.axvspan(band_range16[0],band_range16[1],color='C1',alpha=baralpha, zorder=0)
        if long:
            ax2.axvspan(band_range15[0],band_range15[1],color='C1',alpha=baralpha, zorder=0)
            ax2.axvspan(band_range16[0],band_range16[1],color='C1',alpha=baralpha, zorder=0)

# ---- Plotting WMAP ----
if wmap:
    ax2.text(22.8 -2  , ymax2-0.2,"WMAP \n K ", color='C9' ,va='bottom',alpha=1, size = freqtext)
    ax2.text(31.5   , ymax2-0.2,"Ka",color='C9', va='bottom',alpha=1, size = freqtext)
    ax2.text(39.  , ymax2-0.2,"Q", color='C9' ,va='bottom',alpha=1, size = freqtext)
    ax2.text(58.    , ymax2-0.2,"V", color='C9' ,va='bottom',alpha=1, size = freqtext)
    ax2.text(90.-8   , ymax2-0.2,"W", color='C9' ,va='bottom',alpha=1, size = freqtext)

    ax.axvspan(band_range3[0],band_range3[1],color='C9',alpha=baralpha, zorder=0,label='WMAP')
    ax.axvspan(band_range4[0],band_range4[1],color='C9',alpha=baralpha, zorder=0)
    ax.axvspan(band_range5[0],band_range5[1],color='C9',alpha=baralpha, zorder=0)
    ax.axvspan(band_range6[0],band_range6[1],color='C9',alpha=baralpha, zorder=0)
    ax.axvspan(band_range7[0],band_range7[1],color='C9',alpha=baralpha, zorder=0)
    if long:
        ax2.axvspan(band_range3[0],band_range3[1],color='C9',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range4[0],band_range4[1],color='C9',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range5[0],band_range5[1],color='C9',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range6[0],band_range6[1],color='C9',alpha=baralpha, zorder=0)
        ax2.axvspan(band_range7[0],band_range7[1],color='C9',alpha=baralpha, zorder=0)

# ---- Plotting DIRBE ----
if dirbe and not pol and long:
    ax2.text(1000  ,ymax2-0.2,"DIRBE \n 1249",color='C3', va='bottom',alpha=1, size = freqtext)
    ax2.text(1750  ,ymax2-0.2,"2141",color='C3', va='bottom',alpha=1, size = freqtext)
    ax2.text(2500  ,ymax2-0.2,"2998",color='C3', va='bottom',alpha=1, size = freqtext)

    ax.axvspan(band_range17[0],band_range17[1],color='C3',alpha=baralpha, zorder=0,label='DIRBE')
    ax.axvspan(band_range18[0],band_range18[1],color='C3',alpha=baralpha, zorder=0)
    ax.axvspan(band_range19[0],band_range19[1],color='C3',alpha=baralpha, zorder=0)

    ax2.axvspan(band_range17[0],band_range17[1],color='C3',alpha=baralpha, zorder=0)
    ax2.axvspan(band_range18[0],band_range18[1],color='C3',alpha=baralpha, zorder=0)
    ax2.axvspan(band_range19[0],band_range19[1],color='C3',alpha=baralpha, zorder=0)

# ---- Axis label stuff ----
ax.set_xticks(np.append(ax.get_xticks(),[3,30,300,3000]))
ax.set_xticklabels(np.append(ax.get_xticks(),300))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.tick_params(axis='both', which='major', labelsize=ticksize, direction='in')
ax.tick_params(which="both",direction="in")

ax2.tick_params(axis='both', which='major', labelsize=ticksize, direction='in')
ax2.tick_params(which="both",direction="in")

plt.ylabel(r"Brightness temperature [$\mu$K]",fontsize=labelsize)
plt.xlabel(r"Frequency [GHz]",fontsize=labelsize)

if long:
    ax2.set_ylim(ymax15,ymax2)
    ax2.set_xlim(xmin,xmax)

ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
#ax.legend(loc=6,prop={'size': 20}, frameon=False)

# ---- Plotting ----
plt.tight_layout(h_pad=0.3)
filename ="figs/spectrum"
filename += "_pol" if pol else ""
filename += "_long" if long else ""
plt.savefig(filename+".png", bbox_inches='tight',  pad_inches=0.02)
#plt.show()


