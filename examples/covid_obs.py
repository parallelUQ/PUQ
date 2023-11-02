import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from main_deterministic import runfunction

def obs_covid19(cls_data):
    lm = cls_data.truelims
    hosp_ad, daily_ad_benchmark = runfunction(None, cls_data.true_theta, lm, point=False)
    #2020-02-28
    colors = np.repeat('blue', 189)
    index = np.arange(0, 189, 15)
    #colors[index] = 'red'
    # Generate some random date-time data
    numdays = 222
    base = datetime.datetime(2020, 2, 28, 23, 30) 
    date_list = [base + datetime.timedelta(days=x) for x in range(0, numdays) if x >= 33]
    
    #print(date_list[0])
    #print(date_list[188])
    # Set the locator
    locator = mdates.MonthLocator()  # every month
    # Specify the format - %b gives us Jan, Feb...
    fmt = mdates.DateFormatter('%b')
    
    plt.scatter(date_list, hosp_ad, c=colors, label='Observed data')
    plt.plot(date_list, daily_ad_benchmark, c='red', label='Simulation output')
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    # Specify formatter
    X.set_major_formatter(fmt)
    plt.legend()
    plt.ylabel('COVID-19 Hospital Admissions')
    plt.show()

def obs_des(cls_data):
    
    xint = cls_data.x*221 #np.arange(0, 221)
    f = np.zeros((len(xint), 1))
    for j in range(len(xint)):
        f[j, 0] = cls_data.function(xint[j]/221, cls_data.true_theta[0], cls_data.true_theta[1], cls_data.true_theta[2], cls_data.true_theta[3])
    plt.plot(xint, f[:, 0])
    plt.scatter(cls_data.x*221, cls_data.real_data.flatten())
    plt.show()
    
def obs_output(thetamesh, ytest):
    
    xt_test = np.zeros((n_tot, 5))
    fall = []
    for j in range(n_t):
        hosp_ad, daily_ad_benchmark = runfunction(None, [thetamesh[j, 0], thetamesh[j, 1], thetamesh[j, 2], thetamesh[j, 3]], cls_data.truelims, point=False)
        fall.append(daily_ad_benchmark)

    date_list = [base + datetime.timedelta(days=x) for x in range(0, 222) if x >= 33]
    locator = mdates.MonthLocator()  
    fmt = mdates.DateFormatter('%b')
    for fa in fall:
        plt.plot(date_list, fa, color='gray', zorder=1)
    plt.plot(date_list, ytest.flatten(), color='red', zorder=2)  
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    # Specify formatter
    X.set_major_formatter(fmt)
    #plt.ylabel('COVID-19 Hospital Admissions')
    plt.show()
        
    