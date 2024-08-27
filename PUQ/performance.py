import numpy as np
import PUQ.performanceutils.gentime as gentime
import PUQ.performanceutils.simtime as simtime
import PUQ.performanceutils.accuracy as accuracy


class performanceModel(object):
    def __init__(self, worker, batch, n, n0):
        self.worker = worker
        self.batch = batch
        self.n = n
        self.n0 = n0

    def gen_acqtime(self, *args, typeGen="linear"):
        self.genmethod = eval("gentime." + typeGen)
        self.gentime = self.genmethod(*args, n=self.n, batch=self.batch, n0=self.n0)
        return

    def gen_simtime(self, *args, typeSim="linear", seed=1):
        self.simmethod = eval("simtime." + typeSim)
        self.simtime = self.simmethod(*args, n=self.n, seed=seed)
        return

    def gen_curve(self, *args, typeAcc="exponential"):
        self.accmethod = eval("accuracy." + typeAcc)
        self.acc = self.accmethod(*args, n=self.n, batch=self.batch, ninit=self.n0)
        return

    def simulate(self, sim_times=None, a_time=None):

        if sim_times == None:
            sim_times = self.simtime
        if a_time == None:
            a_time = self.gentime

        self.jobs = []
        self.stages = []
        simulate_parallel(self)

        endtime = [job.end for job in self.jobs]
        sortedjob_ids = np.argsort(endtime)
        self.sortedjobs = [self.jobs[i] for i in sortedjob_ids]

        nt = self.n - self.n0
        b = self.batch
        # accrem = self.acc[self.n0:self.n]
        accrem = self.acc[0:nt]
        accu = [accrem[idacc] for idacc in np.arange(b - 1, nt, b)]
        accu_all = np.repeat(accu[0:-1], b)
        accu_all = np.concatenate((np.repeat(1, b - 1), accu_all))
        accu_all = np.concatenate((accu_all, np.array([accu[-1]])))
        accu_all = np.concatenate((np.repeat(1, self.n0), accu_all))
        self.acc = accu_all

        self.end_time = max(endtime)

    def complete(self, acclevel):

        self.acc_threshold = [a for a in self.acc if a <= acclevel][0]
        id_thr = np.where(self.acc == self.acc_threshold)[0][0]

        self.complete_job_list = []

        sum_total_run_time = 0
        for jobid, job in enumerate(self.sortedjobs):
            job.acc = self.acc[jobid]
            if jobid <= id_thr - 1:
                self.complete_job_list.append(job)
                sum_total_run_time += job.runtime

        self.complete_time = np.max([j.end for j in self.complete_job_list])
        self.computing_hours = self.complete_time * self.worker
        self.avg_idle_time = (self.computing_hours - sum_total_run_time) / self.worker
        self.complete_no = len(self.complete_job_list)
        self.completed_stage = np.max([j.stageid for j in self.complete_job_list])
        self.total_acq_time = np.sum(
            [
                stage.end - stage.start
                for stage in self.stages
                if stage.end <= self.complete_time
            ]
        )

        return

    def summarize(self):
        print(
            "Done with "
            + str(self.worker)
            + " workers"
            + " and batch size "
            + str(self.batch)
            + "\n"
        )
        print("# of parameters acquired: " + str(self.n) + "\n")


def simulate_parallel(self):
    worker = self.worker
    sim_times = self.simtime
    n = self.n
    batch = self.batch
    a_time = self.gentime

    stage_no = 0
    stage = stagecls(stageid=stage_no, jobs=[], start=0, end=0)

    # Update job list
    for i in range(worker):
        job = jobcls(
            jobid=i,
            pending=True,
            start=stage.end,
            runtime=sim_times[i],
            stageid=stage_no,
            workerid=i,
            acc=0,
        )
        self.jobs.append(job)
        stage.jobs.append(job)

    self.stages.append(stage)
    worker_list = np.zeros((worker, 2))
    job_no = len(self.jobs)

    while job_no < n:
        stage_no += 1

        end_list = [job.end for job in self.jobs if job.pending == True]

        job_end_time = sorted(end_list)[batch - 1]

        stage_end_time = self.stages[stage_no - 1].end

        acqtime = np.sum(a_time[job_no : (job_no + batch)])
        acq_start = max(job_end_time, stage_end_time)
        acq_end = acq_start + acqtime

        stage = stagecls(stageid=stage_no, jobs=[], start=acq_start, end=acq_end)

        # Receive batch_size jobs
        count = 0
        idle_worker = []
        for job in self.jobs:
            if job.pending == True:
                if job.end <= job_end_time:
                    if count < batch:
                        job.pending = False
                        idle_worker.append(job.workerid)
                        count += 1

        # Update job list with acquired ones
        stage_no = len(self.stages)
        for i in range(batch):
            job = jobcls(
                jobid=i + job_no,
                pending=True,
                start=stage.end,
                runtime=sim_times[i + job_no],
                stageid=stage_no,
                workerid=idle_worker[i],
                acc=0,
            )
            self.jobs.append(job)
            stage.jobs.append(job)

        self.stages.append(stage)
        job_no = len(self.jobs)

    endjob = [job.end for job in self.jobs]
    end_time = max(endjob)

    for i in range(0, worker):
        sum_j = 0
        for j in self.jobs:
            if j.workerid == i:
                sum_j += j.end - j.start
        worker_list[i, 0] = end_time - sum_j


class jobcls:
    def __init__(self, jobid, pending, start, runtime, stageid, workerid, acc):
        self.jobid = jobid
        self.pending = pending
        self.start = start
        self.runtime = runtime
        self.end = self.start + self.runtime
        self.stageid = stageid
        self.workerid = workerid
        self.acc = acc


class stagecls:
    def __init__(self, stageid, jobs, start, end):
        self.stageid = stageid
        self.jobs = jobs
        self.start = start
        self.end = end
