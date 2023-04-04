import numpy as np
import matplotlib.pyplot as plt 

def find_threshold(acc_level, acc_list):
    threshold = [a for a in acc_list if a <= acc_level][0]
    id_thr = np.where(acc_list == threshold)[0][0]
    return threshold, id_thr

def plot_workers(PM, joblist, acqlist):
    wk = np.zeros(PM.worker)
    complete_stage = 0
    total_acq_time = 0
    fig, ax = plt.subplots()
    for jid, job in enumerate(joblist):
        if job['end'] <= PM.complete_time:
            ax.hlines(y = job['worker']+1, xmin = job['start'], xmax = job['end'], color='b', linewidth=1)
            ax.vlines(x = job['start'], ymin = job['worker']+1-0.25, ymax = job['worker']+1+0.25, color='b', linestyles='dashed' , linewidth=1)
            ax.vlines(x = job['end'], ymin = job['worker']+1-0.25, ymax = job['worker']+1+0.25, color='b', linestyles='dashed' , linewidth=1)
            
            wk[job['worker']] += (job['end'] - job['start'])
        
    for ida, acq in enumerate(acqlist):
        if acq['end'] <= PM.complete_time:
            ax.hlines(y = 0, xmin = acq['start'], xmax = acq['end'], color='r', linewidth=1)
            ax.vlines(x = acq['start'], ymin = 0-0.25, ymax = 0+0.25, linestyles='dashed', color='r', linewidth=1)
            ax.vlines(x = acq['end'], ymin = 0-0.25, ymax = 0+0.25, linestyles='dashed', color='r', linewidth=1)
            complete_stage += 1
            total_acq_time += (acq['end'] - acq['start'])
      
        
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Worker', fontsize=18)
    txt1 = 'Completion:' + str(np.round(PM.complete_time, 1)) + '\n' 
    txt2 = '# of jobs:' + str(np.round(PM.complete_no, 1)) + '\n' 
    
    txt3 = '# of stages:' + str(PM.completed_stage) + '\n' 
    txt4 = 'Acq time:' + str(np.round(PM.total_acq_time , 2)) + '\n' 
    
    txt5 = 'mean (idle):' + str(np.round(PM.complete_time - np.mean(wk))) + '\n' 
    txt6 = 'std (idle):' + str(np.round(np.std(wk)))
    plt.figtext(0.95, 0.2, txt1 + txt2 + txt3 + txt4 + txt5 + txt6, fontname="DejaVu Sans Mono", fontsize=18)
    #plt.xlim(0, sorted(endjob)[-1])
    plt.show()
    
def plot_accuracy(result, n, acclevel, labellist, worker=1, logscale=True):
    accuracylist = []
    acctime = []
    endlist = []
    for r in result:
        accuracylist.append(r.acc)
        acctime.append(r.gentime)
        end = [job['end'] for job in r.job_list]
        endlist.append(end)
        
    thr_list = []
    id_list = []
    for acc in accuracylist:
        thr, idl = find_threshold(acclevel, acc)
        thr_list.append(thr)
        id_list.append(idl)
        
    clist = ['blue', 'red', 'green', 'magenta', 'orange', 'dimgrey', 'lime']
    #llist = ['b=1', 'b=4', 'b=16', 'b=64', 'b=256']
    fig, axes = plt.subplots(1, 4, figsize=(44, 10)) 
    
    for accid, acc in enumerate(accuracylist):
        axes[0].plot(np.arange(1, n+1), acc, label=labellist[accid], color=clist[accid])
        axes[0].vlines(x=id_list[accid], ymin=0, ymax=thr_list[accid], linewidth=2, color=clist[accid], linestyles='dashed')
    axes[0].hlines(y=acclevel, xmin=0, xmax=n, linewidth=2, color = 'k')
    axes[0].set_xlabel("# of parameters", fontsize=30)
    axes[0].set_ylabel("Error", fontsize=30)
    #axes[0].set_ylim([0, 10])
    axes[0].tick_params(axis='both', which='major', labelsize=25)  
    if logscale:
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
    
    #axes[0].legend(bbox_to_anchor=(1, -0.1), ncol=3, fontsize=30)
    for timeid, time in enumerate(acctime):
        axes[1].plot(np.arange(1, n+1), np.cumsum(time), label=labellist[timeid], color=clist[timeid])
        axes[1].vlines(x=id_list[timeid], ymin=0, ymax = np.cumsum(time)[id_list[timeid]], linewidth=4, color=clist[timeid], linestyles='dashed')
    axes[1].set_xlabel("# of parameters", fontsize=30)
    axes[1].set_ylabel("Acquisition time", fontsize=30)
    axes[1].tick_params(axis='both', which='major', labelsize=25)
    if logscale:
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
    #axes[1].legend(bbox_to_anchor=(1, -0.1), ncol=3, fontsize=30)
    
    for endid, endtime in enumerate(endlist):
        axes[2].plot(np.arange(1, n+1), endtime, label=labellist[endid], color=clist[endid])
        axes[2].vlines(x=id_list[endid], ymin=0, ymax = endtime[id_list[endid]], linewidth=2, color=clist[endid], linestyles='dashed')
        axes[2].hlines(y=endtime[id_list[endid]], xmin=0, xmax = id_list[endid], linewidth=2, color=clist[endid], linestyles='dashed')
    #axes[2].legend(bbox_to_anchor=(1.2, -0.1), ncol=5, fontsize=30)
    axes[2].set_xlabel("# of parameters", fontsize=30)
    axes[2].set_ylabel("Completion time", fontsize=30)
    if logscale:
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
    axes[2].set_xlim([worker, n])
    axes[2].set_ylim([np.min([end[worker] for end in endlist]), np.max(endlist)])
    axes[2].tick_params(axis='both', which='major', labelsize=25)
    
    for endid, endtime in enumerate(endlist):    
        axes[3].plot(endtime, accuracylist[endid], label=labellist[endid], color=clist[endid])
        axes[3].vlines(x=endtime[id_list[endid]], ymin=0, ymax=acclevel, linewidth=2, color=clist[endid], linestyles='dashed')


    
    axes[3].hlines(y=acclevel, xmin=0, xmax=np.max(endlist), linewidth=2, color = 'k')
    axes[3].legend(bbox_to_anchor=(-0.6, -0.2), ncol=len(labellist), fontsize=30)
    if logscale:
        axes[3].set_xscale('log')
        axes[3].set_yscale('log')
    axes[3].set_xlim([np.min([end[worker] for end in endlist]), np.max(endlist)])
    axes[3].set_ylim([np.min(accuracylist), np.max([acc[worker] for acc in accuracylist])])
    axes[3].set_xlabel("Completion time", fontsize=30)
    axes[3].set_ylabel("Error", fontsize=30)
    axes[3].tick_params(axis='both', which='major', labelsize=25)
    plt.show()
    
def plot_accuracy2(result, n, acclevel, labellist, worker=1, logscale=True):
    accuracylist = []
    acctime = []
    endlist = []
    for r in result:
        accuracylist.append(r.acc)
        acctime.append(r.gentime)
        end = [job['end'] for job in r.job_list]
        endlist.append(end)
        
    thr_list = []
    id_list = []
    for acc in accuracylist:
        thr, idl = find_threshold(acclevel, acc)
        thr_list.append(thr)
        id_list.append(idl)
        
    clist = ['blue', 'red', 'green', 'magenta', 'orange', 'dimgrey', 'lime']
    #llist = ['b=1', 'b=4', 'b=16', 'b=64', 'b=256']
    fig, axes = plt.subplots(2, 2, figsize=(22, 20)) 
    
    for accid, acc in enumerate(accuracylist):
        axes[0, 0].plot(np.arange(1, n+1), acc, label=labellist[accid], color=clist[accid])
        axes[0, 0].vlines(x=id_list[accid], ymin=0, ymax=thr_list[accid], linewidth=2, color=clist[accid], linestyles='dashed')
    axes[0, 0].hlines(y=acclevel, xmin=0, xmax=n, linewidth=2, color = 'k')
    axes[0, 0].set_xlabel("# of parameters", fontsize=30)
    axes[0, 0].set_ylabel("Error", fontsize=30)
    #axes[0].set_ylim([0, 10])
    axes[0, 0].tick_params(axis='both', which='major', labelsize=25)  
    if logscale:
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
    
    #axes[0].legend(bbox_to_anchor=(1, -0.1), ncol=3, fontsize=30)
    for timeid, time in enumerate(acctime):
        axes[0, 1].plot(np.arange(1, n+1), np.cumsum(time), label=labellist[timeid], color=clist[timeid])
        axes[0, 1].vlines(x=id_list[timeid], ymin=0, ymax = np.cumsum(time)[id_list[timeid]], linewidth=4, color=clist[timeid], linestyles='dashed')
    axes[0, 1].set_xlabel("# of parameters", fontsize=30)
    axes[0, 1].set_ylabel("Acquisition time", fontsize=30)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=25)
    if logscale:
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
    #axes[1].legend(bbox_to_anchor=(1, -0.1), ncol=3, fontsize=30)
    
    for endid, endtime in enumerate(endlist):
        axes[1, 0].plot(np.arange(1, n+1), endtime, label=labellist[endid], color=clist[endid])
        axes[1, 0].vlines(x=id_list[endid], ymin=0, ymax = endtime[id_list[endid]], linewidth=2, color=clist[endid], linestyles='dashed')
        axes[1, 0].hlines(y=endtime[id_list[endid]], xmin=0, xmax = id_list[endid], linewidth=2, color=clist[endid], linestyles='dashed')
    #axes[2].legend(bbox_to_anchor=(1.2, -0.1), ncol=5, fontsize=30)
    axes[1, 0].set_xlabel("# of parameters", fontsize=30)
    axes[1, 0].set_ylabel("Completion time", fontsize=30)
    if logscale:
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlim([worker, n])
    axes[1, 0].set_ylim([np.min([end[worker] for end in endlist]), np.max(endlist)])
    axes[1, 0].tick_params(axis='both', which='major', labelsize=25)
    
    for endid, endtime in enumerate(endlist):    
        axes[1, 1].plot(endtime, accuracylist[endid], label=labellist[endid], color=clist[endid])
        axes[1, 1].vlines(x=endtime[id_list[endid]], ymin=0, ymax=acclevel, linewidth=2, color=clist[endid], linestyles='dashed')


    
    axes[1, 1].hlines(y=acclevel, xmin=0, xmax=np.max(endlist), linewidth=2, color = 'k')
    axes[1, 1].legend(bbox_to_anchor=(0.5, -0.2), ncol=len(labellist), fontsize=30)
    if logscale:
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlim([np.min([end[worker] for end in endlist]), np.max(endlist)])
    axes[1, 1].set_ylim([np.min(accuracylist), np.max([acc[worker] for acc in accuracylist])])
    axes[1, 1].set_xlabel("Completion time", fontsize=30)
    axes[1, 1].set_ylabel("Error", fontsize=30)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=25)
    plt.show()
    

def plot_acc(axes, n, acclevel, rlist, labellist, logscale=False, fontsize=18):
    clist = ['blue', 'red', 'green', 'magenta', 'orange', 'dimgrey', 'lime', 'dimgrey', 'lime']
    #fig, axes = plt.subplots(1, 1, figsize=(4, 4)) 
    
    for accid, res in enumerate(rlist):
        axes.plot(np.arange(1, n+1), res.acc, label=labellist[accid], color=clist[accid])
        axes.vlines(x=res.complete_no, ymin=0, ymax=res.acc_threshold, linewidth=2, color=clist[accid], linestyles='dashed')
    axes.hlines(y=acclevel, xmin=0, xmax=n, linewidth=2, color = 'k')
    axes.set_xlabel("# of parameters", fontsize=fontsize)
    axes.set_ylabel("Error", fontsize=fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-5)  
    if logscale:
        axes.set_xscale('log')
        axes.set_yscale('log')    

def plot_acqtime(axes, n, acclevel, rlist, labellist, logscale=False, fontsize=18):
    clist = ['blue', 'red', 'green', 'magenta', 'orange', 'dimgrey', 'lime', 'dimgrey', 'lime']
    #fig, axes = plt.subplots(1, 1, figsize=(4, 4)) 
    
    for accid, res in enumerate(rlist):
        axes.plot(np.arange(1, n+1), np.cumsum(res.gentime), label=labellist[accid], color=clist[accid])
        # axes.vlines(x=res.complete_no, ymin=0, ymax=res.acc_threshold, linewidth=2, color=clist[accid], linestyles='dashed')
    # axes.hlines(y=acclevel, xmin=0, xmax=n, linewidth=2, color = 'k')
    axes.set_xlabel("# of parameters", fontsize=fontsize)
    axes.set_ylabel("Acquisition time", fontsize=fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-5)
    if logscale:
        axes.set_xscale('log')
        axes.set_yscale('log') 
        
def plot_endtime(axes, n, acclevel, rlist, labellist, worker, logscale=False, fontsize=18):
    clist = ['blue', 'red', 'green', 'magenta', 'orange', 'dimgrey', 'lime', 'dimgrey', 'lime']
    #fig, axes = plt.subplots(1, 1, figsize=(4, 4)) 
    
    minworker = []
    maxtime = 0
    for endid, res in enumerate(rlist):
        endtime = np.sort([job['end'] for job in res.job_list])
        minworker.append(endtime[worker])
        if maxtime < np.max(endtime):
            maxtime = np.max(endtime)
        axes.plot(np.arange(1, n+1), endtime, label=labellist[endid], color=clist[endid])
        axes.vlines(x=res.complete_no, ymin=0, ymax = endtime[res.complete_no], linewidth=2, color=clist[endid], linestyles='dashed')
        axes.hlines(y=endtime[res.complete_no], xmin=0, xmax=res.complete_no, linewidth=2, color=clist[endid], linestyles='dashed')

    axes.set_xlabel("# of parameters", fontsize=fontsize)
    axes.set_ylabel("Completion time", fontsize=fontsize)
    if logscale:
        axes.set_xscale('log')
        axes.set_yscale('log')
    axes.set_xlim([worker, n])
    axes.set_ylim([np.min(minworker), maxtime])
    axes.tick_params(axis='both', which='major', labelsize=fontsize-5)
    
def plot_errorend(axes, n, acclevel, rlist, labellist, worker, logscale=False, fontsize=18):
    clist = ['blue', 'red', 'green', 'magenta', 'orange', 'dimgrey', 'lime', 'dimgrey', 'lime']
    #fig, axes = plt.subplots(1, 1, figsize=(4, 4)) 
    minworker = []
    maxtime = 0
    minacc = 10
    for endid, res in enumerate(rlist):
        endtime = np.sort([job['end'] for job in res.job_list])
        minworker.append(endtime[worker])
        if maxtime < np.max(endtime):
            maxtime = np.max(endtime)
        if minacc > np.min(res.acc):
            minacc = np.min(res.acc)
        axes.plot(endtime, res.acc, label=labellist[endid], color=clist[endid])
        axes.vlines(x=endtime[res.complete_no], ymin=0, ymax=acclevel, linewidth=2, color=clist[endid], linestyles='dashed')
        
    axes.hlines(y=acclevel, xmin=0, xmax=maxtime, linewidth=2, color = 'k')

    if logscale:
        axes.set_xscale('log')
        axes.set_yscale('log')
    axes.set_xlim([np.min(minworker), maxtime])
    axes.set_ylim([minacc, np.max([res.acc[worker] for res in rlist])])
    axes.set_xlabel("Completion time", fontsize=fontsize)
    axes.set_ylabel("Error", fontsize=fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-5)    
    
