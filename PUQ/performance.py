import numpy as np
import PUQ.performanceutils.gentime as gentime
import PUQ.performanceutils.simtime as simtime
import PUQ.performanceutils.accuracy as accuracy

class performanceModel(object):
    def __init__(self, worker, batch, n):
        self.worker = worker
        self.batch = batch
        self.n = n
    
    def gen_gentime(self, *args, typeGen='linear'):
        self.genmethod = eval('gentime.' + typeGen)
        self.gentime = self.genmethod(*args, n=self.n, batch=self.batch)
        return 
        
    def gen_simtime(self, *args, typeSim='linear'):
        self.simmethod = eval('simtime.' + typeSim)
        self.simtime = self.simmethod(*args, n=self.n)
        return 
    
    def gen_accuracy(self, *args, typeAcc='exponential'):
        self.accmethod = eval('accuracy.' + typeAcc)
        self.acc = self.accmethod(*args, n=self.n, batch=self.batch)   
        return    
        
    def simulate(self, sim_times=None, a_time=None):
        
        if sim_times == None:
            sim_times = self.simtime
        if a_time == None:
            a_time = self.gentime
        
        self.job_list, self.worker_list, self.stage_list = simulate_parallel(n=self.n, 
                                                                             worker=self.worker, 
                                                                             batch=self.batch, 
                                                                             sim_times=sim_times, 
                                                                             a_time=a_time)
        endjob = [job['end'] for job in self.job_list]
        self.end_time = max(endjob)
 
    def complete(self, acclevel):
        

        threshold = [a for a in self.acc if a <= acclevel][0]
        id_thr = np.where(self.acc == threshold)[0][0]
   
        endjob = [job['end'] for job in self.job_list]
        
        job_end = np.sort(endjob)[id_thr-1]
        print(job_end)
        print(id_thr)
        for job in self.job_list:
            if job['end'] == job_end:
                stage_id = job['stage']

        stage_end = np.max([job['end'] for job in self.stage_list[stage_id]['jobs']])
        print(stage_id)
        print(stage_end)

        self.complete_time = np.max([job_end, stage_end]) 
        #print(self.complete_time)
        count_finish_job = 0
        for job in self.job_list:
            if job['end'] <= self.complete_time:
                count_finish_job += 1
                
        self.complete_no = count_finish_job
        self.acc_threshold = threshold
        
        #sortedjobs = np.argsort(endjob)
        #for sid, s in enumerate(sortedjobs):
        #    self.job_list[s]['endid'] = sid
        #    self.job_list[s]['acc'] = self.acc[sid]
     

        self.completed_stage = stage_id + 1
        self.total_acq_time = np.sum([stage['end'] - stage['start'] for stage in self.stage_list if stage['end'] <= self.complete_time])


        return 
        
    def summarize(self):
        print('Done with ' + str(self.worker) + ' workers' + ' and batch size ' +  str(self.batch) + '\n' )
        print('# of parameters acquired: ' + str(self.n) + '\n')
    

        
        
def create_stage(idstage, jobstage, endstage, startstage):
    stage = {"id": idstage, "jobs": jobstage, "end": endstage, "start": startstage}
    return stage

def create_job(idjob, pending, startjob, endjob, stageid, workerid):
    job = {"id": idjob, "pending": pending, "start": startjob, "end": endjob, "stage": stageid, "worker": workerid}
    return job

def simulate_parallel(n, worker, batch, sim_times, a_time):
    stage_list, job_list, worker_list, acquisitions = [], [], [], []
    stage_no = 0
    stage = create_stage(idstage=stage_no, jobstage=[], endstage=0, startstage=0)
    
    # Update job list
    for i in range(worker):
        s   = sim_times[i] 
        job = create_job(i, True, stage['end'], stage['end'] + s, stage_no, i)
        job_list.append(job)
        stage['jobs'].append(job)

    stage_list.append(stage)
    worker_list = np.zeros((worker, 2))        
    job_no = len(job_list)
 
    while (job_no < n):
        stage_no += 1

        end_list = [job['end'] for job in job_list if job['pending'] == True]
   
        job_end_time   = sorted(end_list)[batch-1]
        
        stage_end_time = stage_list[stage_no-1]['end']
        
        acqtime   = np.sum(a_time[job_no: (job_no + batch)])
        acq_start = max(job_end_time, stage_end_time)
        acq_end   = acq_start + acqtime
        
        acquisitions.append({'start': acq_start, 'end': acq_end})

        stage = create_stage(idstage=stage_no, jobstage=[], endstage=acq_end, startstage=acq_start)
        
        # Receive batch_size jobs
        count = 0
        idle_worker = []
        for job in job_list:
            if job['pending'] == True:
                if job['end'] <= job_end_time:
                    if count < batch:
                        job['pending'] = False
                        idle_worker.append(job['worker'])
                        count += 1
                    
        
        # Update job list with acquired ones
        stage_no = len(stage_list)
        for i in range(batch):
            s   = sim_times[i + job_no] 
            job = create_job(i + job_no, True, stage['end'], stage['end'] + s, stage_no, idle_worker[i])
            job_list.append(job)
            stage['jobs'].append(job)

        stage_list.append(stage)
        job_no = len(job_list)
                
    endjob = [job['end'] for job in job_list]
    end_time = max(endjob)

    for i in range(0, worker):
        sum_j = 0
        for j in job_list:
            if j['worker'] == i:
                sum_j += (j['end'] - j['start'])
        worker_list[i, 0] = end_time - sum_j
        
    return job_list, worker_list, stage_list