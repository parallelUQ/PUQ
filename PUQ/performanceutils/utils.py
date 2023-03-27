import numpy as np
import matplotlib.pyplot as plt 

def find_threshold(acc_level, acc_list):
    threshold = [a for a in acc_list if a <= acc_level][0]
    id_thr = np.where(acc_list == threshold)[0][0]
    return threshold, id_thr

def plot_workers(PM, joblist, acqlist):
    fig, ax = plt.subplots()
    for jid, job in enumerate(joblist):
        ax.hlines(y = job['worker']+1, xmin = job['start'], xmax = job['end'], linewidth=1)
        ax.vlines(x = job['start'], ymin = job['worker']+1-0.25, ymax = job['worker']+1+0.25, linestyles='dashed' , linewidth=1)
        ax.vlines(x = job['end'], ymin = job['worker']+1-0.25, ymax = job['worker']+1+0.25, linestyles='dashed' , linewidth=1)
    
        
    for ida, acq in enumerate(acqlist):
        ax.hlines(y = 0, xmin = acq['start'], xmax = acq['end'], color='r', linewidth=1)
        ax.vlines(x = acq['start'], ymin = 0-0.25, ymax = 0+0.25, linestyles='dashed', color='r', linewidth=1)
        ax.vlines(x = acq['end'], ymin = 0-0.25, ymax = 0+0.25, linestyles='dashed', color='r', linewidth=1)
      
    plt.xlabel('Time')
    plt.ylabel('Worker')
    txt1 = 'End:' + str(np.round(PM.end_time, 1)) + '\n' 
    txt2 = '# of stages:' + str(PM.stage_list[-1]['id']) + '\n' 
    txt3 = 'mean (idle):' + str(np.round(np.mean(PM.worker_list[:, 0]))) + '\n' 
    txt4 = 'std (idle):' + str(np.round(np.std(PM.worker_list[:, 0])))
    plt.figtext(0.95, 0.2, txt1 + txt2 + txt3 + txt4, fontname="DejaVu Sans Mono")
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
    axes[2].set_ylabel("End time", fontsize=30)
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
    axes[3].set_xlabel("End time", fontsize=30)
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
    axes[1, 0].set_ylabel("End time", fontsize=30)
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
    axes[1, 1].set_xlabel("End time", fontsize=30)
    axes[1, 1].set_ylabel("Error", fontsize=30)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=25)
    plt.show()