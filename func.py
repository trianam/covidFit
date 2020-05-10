import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy as sp
import scipy.optimize
import numpy as np
import pandas as pd
import sympy as sym
from sympy.core.evaluate import evaluate
import math
from cycler import cycler

plt.rcParams['figure.dpi'] = 100
plt.rc('axes', prop_cycle=(cycler('color', ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']) +
                           cycler('linestyle', ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', ':', ':', ':', ':', ':', ':', ':', ':', ':', ':'])))

func = {
    'quartic': lambda x,a,b,c,d,e: a*x**4 + b*x**3 +c*x**2 + d*x + e,
    'cubic': lambda x,a,b,c,d: a*x**3 + b*x**2 +c*x + d,
    'quadratic': lambda x,a,b,c: a*x**2 + b*x +c,
    'linear': lambda x,a,b: a*x + b,
    'power': lambda x,a,b,c: a*x**b + c,
    'powerNorm': lambda x,a,b: x**a + b,
    'powerSum': lambda x,a,b,c: (x+a)**b + c,
    'exp': lambda x,a,b,c: a*b**x + c,
    'expNorm': lambda x,a,b: a**x + b,
    'expGrowth': lambda x,a,b,c: a*b**(x/c), #a: f(0); b: growth factor; c: time to increase by a factor of b
    'expGrowth2': lambda x,a,b,c,d: a*b**(x/c)+d,
    'expGrowthNorm': lambda x,a,b: a**(x/b),
    'expGrowthNorm2': lambda x,a,b: a*b**x,
    'expGrowthNorm3': lambda x,a: a**x,
    'logistic': lambda x,a,b,c: c/(1+np.exp(-(x-b)/a)), #a: speed; b: peak; c: end
    'gompertz': lambda x,a,b,c: c*(np.exp(-np.log(c/b)*np.exp(-a*x))), #a: speed; b: start; c: end
}

values = {
    'Total cases':'totale_casi',
    'Hospitalizations':'totale_ospedalizzati',
    'Intensive care':'terapia_intensiva',
    'Deceased':'deceduti',
}

labels = {
    'Total cases':'Cases',
    'Hospitalizations':'Hosp.',
    'Intensive care':'ICU',
    'Deceased':'Dead',
}

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sym.Number)})

def getRegions():
    data = pd.read_csv('https://github.com/pcm-dpc/COVID-19/raw/master/dati-regioni/dpc-covid19-ita-regioni.csv')
    return list(data.denominazione_regione.unique())

def plotCurve(keys, zones='Italia', funcName=None, plotError=True, derivative=0, normalize=None, forecastDays=20):
    #normalize: None, 'population', 'swabs'
    ax = plt.axes()
    
    if not type(keys) is list:
        keys = [keys]
    if not type(zones) is list:
        zones = [zones]
        
    for zone in zones:
        for key in keys:
            if zone == 'Italia':
                data = pd.read_csv('https://github.com/pcm-dpc/COVID-19/raw/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')
            else:
                data = pd.read_csv('https://github.com/pcm-dpc/COVID-19/raw/master/dati-regioni/dpc-covid19-ita-regioni.csv')
                data = data[data.denominazione_regione == zone]

            x = np.array(np.linspace(1+53+derivative, len(data)+53, len(data)-derivative, dtype=float)) #start on 54th day of year
            y = np.array(data[values[key]], dtype=float)
            for _ in range(derivative):
                y = np.array([y[i]-y[i-1] for i in range(1,len(y))])
                
            if normalize == 'population':
                dataPop = pd.read_csv('DCIS_POPRES1_27032020185638165.csv')
                y = y / float(dataPop[dataPop.Territorio == zone].Value) * 100.
            
            elif normalize == 'swabs':
                y = y / np.array(data['tamponi'], dtype=float) * 100.
            
            if derivative > 0:
                plotLabel = "{} - {}\n(derivative {})".format(labels[key], zone, derivative)
            else:
                plotLabel = "{} - {}".format(labels[key], zone)
                
            xx = np.array(np.linspace(x[0], x[-1]+forecastDays, 50), dtype=float)
            dates = np.array(data['data'])

            plt.plot(x, y, label=plotLabel)

            if not funcName is None:
                try:
                    if funcName == 'logistic':
                        popt, pcov = sp.optimize.curve_fit(func[funcName], x, y, p0=[2,100,30000], maxfev=1000)
                        tex = '\\frac{{{}}}{{1+e^{{-(x-{})/{}}}}}'.format(round(popt[2],2),round(popt[1],2),round(popt[0],2))
                    else:
                        popt, pcov = sp.optimize.curve_fit(func[funcName], x, y)
                        #print(list(map(lambda v: '{0:.3f}'.format(v),popt)))

                        xs = sym.Symbol('x')
                        with evaluate(False):
                            #expr = func[funcName](sym.UnevaluatedExpr(xs),*(map(lambda v: round(v,2), popt)))
                            expr = func[funcName](xs,*(map(lambda v: round(v,2), popt)))
                        tex = sym.latex(expr).replace('$', '')
                        #tex = sym.latex(round_expr(func[funcName](xs,*popt), 3)).replace('$', '')

                    plt.plot(xx, func[funcName](xx, *popt), 'k')
                    
                    if plotError:
                        errors = np.sqrt(np.diag(pcov))
                        nstd = 1.0
                        fit_up = popt + nstd * errors
                        fit_dw = popt - nstd * errors
                        fit_up_p = func[funcName](xx, *fit_up)
                        fit_dw_p = func[funcName](xx, *fit_dw)

                        plt.fill_between(xx, fit_up_p, fit_dw_p, alpha=.35, color="gray")

                except RuntimeError as e:
                    print(e)
                    tex = 'ERROR'

    if not funcName is None:
        #plt.title('$f(x)= {}$\n{} - {}'.format(tex, dates[0], dates[-1]),fontsize=16)
        plt.title('$f(x)= {}$'.format(tex),fontsize=16)
    #else:
    #    plt.title('{} - {}'.format(dates[0], dates[-1]),fontsize=16)
        
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=int((len(keys)*len(zones))/15) + 1)
    #ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1.02), ncol=1, fancybox=True, shadow=True)
    #plt.legend(loc = 'upper left')
    if normalize is None:
        plt.ylabel('persons')
    elif normalize == 'population':
        plt.ylabel('% of population')
    elif normalize == 'swabs':
        plt.ylabel('% of swabs')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()
    plt.grid(which='minor', alpha=0.2)
    ax.minorticks_on()
    plt.grid()
    plt.show()