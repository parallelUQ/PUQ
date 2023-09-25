#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 18:13:28 2020

@author: haoxiangyang
"""

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib import rc
import matplotlib.patches as patches
import matplotlib.colors as pltcolors
import numpy as np
import pickle
import textwrap

def calCost(policy, profiles):
    # for each sampled path
    costList = np.zeros(len(profiles))
    for indp in range(len(profiles)):
        p = profiles[indp]
        cost_p = 0
        for item in p['tier_history']:
            cost_p += policy.tiers[item]['daily_cost']
        costList[indp] = cost_p
    return np.array(costList)

def getUnserved(hosp_capacity, profiles, t_start):
    unservedList = []
    vioNo = 0
    for p in profiles:
        overCap_reg = np.maximum(np.sum(p['ICU'], axis = (1,2))[t_start:] - hosp_capacity, 0)
        unservedNo = np.sum(overCap_reg)
        unservedList.append(unservedNo)
        if unservedNo > 0:
            vioNo += 1
    return np.array(unservedList), vioNo

def getTiermean(policy, profiles, t_start):
    tierList = []
    for indp in range(len(profiles)):
        p = profiles[indp]
        tierListp = [np.sum(p['tier_history'][t_start:] == tier) for tier in range(len(policy.tiers))]
        tierList.append(tierListp)
    return np.mean(np.array(tierList),axis = 0)

def getRedDays(policy, profiles, t_start):
    redList = []
    red_ind = 4
    for tierInd in range(len(policy.tiers)):
        tier = policy.tiers[tierInd]
        if tier['color'] == 'red':
            red_ind = tierInd
    for indp in range(len(profiles)):
        p = profiles[indp]
        tierListp = np.sum(p['tier_history'][t_start:] == red_ind)
        redList.append(tierListp)
    return np.array(redList)
    
    
def getMaxICU(profiles, t_start):
    maxICUList = []
    for indp in range(len(profiles)):
        p = profiles[indp]
        maxICUp = np.max(np.sum(p['ICU'],axis = (1,2))[t_start:])
        maxICUList.append(maxICUp)
    return np.array(maxICUList)

#%%
fileList = ["austin_test_IHT_r2_tiers5_opt_Final_opt5_rl1000.p",
            "austin_test_IHT_r2_tiers2_opt_Final_fixed_opt3_rl1000.p",
            "austin_test_IHT_IHT_r2_tiers5_opt_Final_opt5_rl1000.p",
            "austin_test_IHT_r2_tiers5_ICU_opt_Final_France_opt4_rl1000_99_199.p",
            "austin_test_IHT_r2_tiers5_ITot_opt_Final_fixed_10_opt5_rl1000.p"]

costPrint = []
unservedPrint = []
vioPrint = []
tierPrint = []
redPrint = []
maxICUPrint = []
t_start = 222
for file_item in fileList:
    file_path = "/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/Final_Tests/" + file_item
    with open(file_path, 'rb') as outfile:
        read_output = pickle.load(outfile)
    instance, interventions, best_params, best_policy, best_sim, profiles, config, cost_record, seeds_info = read_output
    costItem = calCost(best_policy, profiles)
    costPrint.append(np.mean(costItem))
    
    unservedList, vioNo = getUnserved(331, profiles, t_start)
    #unservedPrint.append(np.mean(unservedList))
    unservedPrint.append(np.sort(unservedList)[284])
    vioPrint.append(vioNo)
    
    tierMean = getTiermean(best_policy,profiles,t_start)
    tierPrint.append(tierMean)
    
    redList = getRedDays(best_policy, profiles, t_start)
    redPrint.append(redList)
    
    maxICUList = getMaxICU(profiles, t_start)
    maxICUPrint.append(maxICUList)

tierPrint1 = np.zeros(5)
tierPrint1[0] = tierPrint[1][0]
tierPrint1[2] = tierPrint[1][1]
tierPrint1[4] = tierPrint[1][2]
tierPrint[1] = tierPrint1

tierPrint3 = np.zeros(5)
tierPrint3[0] = tierPrint[3][0]
tierPrint3[2] = tierPrint[3][1]
tierPrint3[3] = tierPrint[3][2]
tierPrint3[4] = tierPrint[3][3]
tierPrint[3] = tierPrint3

tSum = np.sum(tierPrint[0])
textList = ["Optimal","Optimal two-stage","Optimal hospital","Percent ICU","Incidence"]

#%%
plt.rcParams["font.size"] = "18"
fig, (ax1, actions_ax) = plt.subplots(2, 1, figsize=(17, 9), gridspec_kw={'height_ratios': [10, 1]})
ax1.set_xlim(-10, 90)

ax1.scatter(vioPrint,costPrint,label = "Expected Cost",s = 80)
ax1.set_ylabel("Expected Cost")
ax1.bar(vioPrint,np.array([tierPrint[i][0]/tSum for i in range(5)])*1000000,2,
        bottom = np.array(costPrint)+50000,
        color = 'white',edgecolor = 'gray',alpha = 0.6)
endStart = np.array(costPrint)+50000+np.array([tierPrint[i][0]/tSum for i in range(5)])*1000000

ax1.bar(vioPrint,np.array([tierPrint[i][1]/tSum for i in range(5)])*1000000,2,
        bottom = endStart,
        color = 'blue',edgecolor = 'gray',alpha = 0.6)
endStart += np.array([tierPrint[i][1]/tSum for i in range(5)])*1000000

ax1.bar(vioPrint,np.array([tierPrint[i][2]/tSum for i in range(5)])*1000000,2,
        bottom = endStart,
        color = 'yellow',edgecolor = 'gray',alpha = 0.6)
endStart += np.array([tierPrint[i][2]/tSum for i in range(5)])*1000000

ax1.bar(vioPrint,np.array([tierPrint[i][3]/tSum for i in range(5)])*1000000,2,
        bottom = endStart,
        color = 'orange',edgecolor = 'gray',alpha = 0.6)
endStart += np.array([tierPrint[i][3]/tSum for i in range(5)])*1000000

ax1.bar(vioPrint,np.array([tierPrint[i][4]/tSum for i in range(5)])*1000000,2,
        bottom = endStart,
        color = 'red',edgecolor = 'gray',alpha = 0.6)
endStart += np.array([tierPrint[i][4]/tSum for i in range(5)])*1000000
for i in range(5):
    ax1.text(vioPrint[i],endStart[i]+30000,"{}".format(textList[i]),fontsize=20)

ax1.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    right=False,  # ticks along the top edge are off
    labelbottom=False,
    labelright=False)  # labels along the bottom edge are off

actions_ax.set_xlim(-10/300,0.3)
actions_ax.set_xlabel("ICU Violation Probability")
actions_ax.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both', # ticks along the bottom edge are off
    left=False,  # ticks along the top edge are off
    labelleft=False)  # labels along the bottom edge are off
ax1.spines['bottom'].set_visible(False)
actions_ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(hspace=0)

plot_filename = '/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/plots/austin_visualization.pdf'
plt.savefig(plot_filename)

#%%
plt.rcParams["font.size"] = "22"
fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
scatterList = []
colorList = ["black","brown","green","olive","purple"]

for i in range(5):
    scatterItem = ax1.scatter(np.array(maxICUPrint[i]),redPrint[i],color = colorList[i])
    scatterList.append(scatterItem)
    
ax1.vlines(331, -10, 120, colors='k',linewidth = 3)
ax1.set_ylabel("Days in red stage")
ax1.set_ylim(-10,120)
ax1.xaxis.label.set_fontsize(24)
ax1.yaxis.label.set_fontsize(24)

ax1.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    right=False,  # ticks along the top edge are off
    labelright=False)  # labels along the bottom edge are off

ax1.set_xlabel("Peak ICU demand (patients)")
ax1.legend([scatterList[0],scatterList[1],scatterList[2],scatterList[3],scatterList[4]],
           textList, markerscale = 3)

plot_filename = '/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/plots/austin_scatter.pdf'
plt.savefig(plot_filename)

#%%
plt.rcParams["font.size"] = "22"
fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
ax1.xaxis.label.set_fontsize(24)
ax1.yaxis.label.set_fontsize(24)

ax2 = ax1.twinx()
ax2.spines["right"].set_visible(True)
ax2.yaxis.label.set_fontsize(24)

X = np.arange(5)
barList1 = ax1.bar(X,np.array([tierPrint[i][4]/tSum for i in range(5)]),width = 0.15, color = 'red',alpha = 0.6)
barList2 = ax1.bar(X+0.15,np.array([tierPrint[i][3]/tSum for i in range(5)]), width = 0.15,color = 'orange',alpha = 0.6)
barList3 = ax1.bar(X+0.3,np.array([tierPrint[i][2]/tSum for i in range(5)]), width = 0.15,color = 'yellow',alpha = 0.6)
barList4 = ax1.bar(X+0.45,np.array([tierPrint[i][1]/tSum for i in range(5)]), width = 0.15,color = 'blue',alpha = 0.6)
barList5 = ax2.bar(X+0.6,np.array(unservedPrint), width = 0.15,color = 'gray',alpha = 0.6)

ax1.set_ylabel("Proportion of days")
ax2.set_ylabel("Unmet ICU demand (patient-days)")

ax1.set_ylim(0,1)
ax1.set_xticks([0.3,1.3,2.3,3.3,4.3])
f = lambda x: textwrap.fill(x, 10)
ax1.set_xticklabels(map(f, textList))
ax1.legend([barList1,barList2,barList3,barList4,barList5],
           ['Red stage','Orange stage','Yellow stage','Blue stage','Unserved ICU'])
plot_filename = '/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/plots/austin_bar.pdf'
plt.savefig(plot_filename)

#%%
fileList = ["houston_test_IHT_r2_tiers5_opt_Final_fixed_opt5_rl1000.p",
            "houston_test_IHT_r2_tiers2_opt_Final_fixed_opt3_rl1000.p",
            "houston_test_IHT_IHT_r2_tiers5_opt_Final_fixed_opt5_rl1000.p",
            "houston_test_IHT_r2_tiers5_ICU_opt_Final_France_opt4_rl1000.p",
            "houston_test_IHT_r2_tiers5_ITot_opt_Final_fixed_10_opt5_rl1000.p"]

costPrint = []
unservedPrint = []
vioPrint = []
tierPrint = []
redPrint = []
maxICUPrint = []
t_start = 231
for file_item in fileList:
    file_path = "/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/Final_Tests/" + file_item
    with open(file_path, 'rb') as outfile:
        read_output = pickle.load(outfile)
    instance, interventions, best_params, best_policy, best_sim, profiles, config, cost_record, seeds_info = read_output
    costItem = calCost(best_policy, profiles)
    costPrint.append(np.mean(costItem))
    
    unservedList, vioNo = getUnserved(1000, profiles, t_start)
    #unservedPrint.append(np.mean(unservedList))
    unservedPrint.append(np.sort(unservedList)[284])
    vioPrint.append(vioNo)
    
    tierMean = getTiermean(best_policy,profiles,t_start)
    tierPrint.append(tierMean)
    
    redList = getRedDays(best_policy, profiles, t_start)
    redPrint.append(redList)
    
    maxICUList = getMaxICU(profiles, t_start)
    maxICUPrint.append(maxICUList)

tierPrint1 = np.zeros(5)
tierPrint1[0] = tierPrint[1][0]
tierPrint1[2] = tierPrint[1][1]
tierPrint1[4] = tierPrint[1][2]
tierPrint[1] = tierPrint1

tierPrint3 = np.zeros(5)
tierPrint3[0] = tierPrint[3][0]
tierPrint3[2] = tierPrint[3][1]
tierPrint3[3] = tierPrint[3][2]
tierPrint3[4] = tierPrint[3][3]
tierPrint[3] = tierPrint3

tSum = np.sum(tierPrint[0])
textList = ["Optimal","Optimal two-stage","Optimal hospital","Percent ICU","Incidence"]

#%%
plt.rcParams["font.size"] = "22"
fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
scatterList = []
colorList = ["black","brown","green","olive","purple"]

for i in range(5):
    scatterItem = ax1.scatter(np.array(maxICUPrint[i]),redPrint[i],color = colorList[i])
    scatterList.append(scatterItem)
    
scatterItem = ax1.scatter(maxICUPrint[0],redPrint[0],color = colorList[0],alpha = 0.3)
scatterList.append(scatterItem)
    
ax1.vlines(1000, -10, 60, colors='k',linewidth = 3)
ax1.set_ylabel("Days in red stage")
ax1.set_ylim(-10,60)
ax1.xaxis.label.set_fontsize(24)
ax1.yaxis.label.set_fontsize(24)

ax1.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    right=False,  # ticks along the top edge are off
    labelright=False)  # labels along the bottom edge are off

ax1.set_xlabel("Peak ICU demand (patients)")
ax1.legend([scatterList[0],scatterList[1],scatterList[2],scatterList[3],scatterList[4]],
           textList, markerscale = 3, loc = 2)

plot_filename = '/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/plots/houston_scatter.pdf'
plt.savefig(plot_filename)

#%%
plt.rcParams["font.size"] = "22"
fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
ax1.xaxis.label.set_fontsize(24)
ax1.yaxis.label.set_fontsize(24)

ax2 = ax1.twinx()
ax2.spines["right"].set_visible(True)
ax2.yaxis.label.set_fontsize(24)

X = np.arange(5)
barList1 = ax1.bar(X,np.array([tierPrint[i][4]/tSum for i in range(5)]),width = 0.15, color = 'red',alpha = 0.6)
barList2 = ax1.bar(X+0.15,np.array([tierPrint[i][3]/tSum for i in range(5)]), width = 0.15,color = 'orange',alpha = 0.6)
barList3 = ax1.bar(X+0.3,np.array([tierPrint[i][2]/tSum for i in range(5)]), width = 0.15,color = 'yellow',alpha = 0.6)
barList4 = ax1.bar(X+0.45,np.array([tierPrint[i][1]/tSum for i in range(5)]), width = 0.15,color = 'blue',alpha = 0.6)
barList5 = ax2.bar(X+0.6,np.array(unservedPrint), width = 0.15,color = 'gray',alpha = 0.6)

ax1.set_ylabel("Proportion of time spent in each stage")
ax2.set_ylabel("Unmet ICU demand (patient-days)")

ax1.set_ylim(0,1)
ax1.set_xticks([0.3,1.3,2.3,3.3,4.3])
f = lambda x: textwrap.fill(x, 10)
ax1.set_xticklabels(map(f, textList))
ax1.legend([barList1,barList2,barList3,barList4,barList5],
           ['Red stage','Orange stage','Yellow stage','Blue stage','Unserved ICU'])
plot_filename = '/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/plots/houston_bar.pdf'
plt.savefig(plot_filename)

#%%
