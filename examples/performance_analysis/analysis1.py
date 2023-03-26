from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import plot_accuracy

timeparams = [0.008, 0.006, 0.004]
accparams = [[-1,0.2], [-1, 0.3], [-1, 0.4]]
result = []
for mid, m in enumerate(timeparams):
    
    PM = performanceModel(worker=1, batch=1, n=4096)
    
    PM.gen_gentime(timeparams[mid], typeGen='constant')
    PM.gen_simtime(0.0001, 0.0001, typeSim='normal')
    PM.gen_accuracy(accparams[mid][0], accparams[mid][1], typeAcc='exponential')
    
    PM.simulate()
    
    PM.summarize()
    result.append(PM)

plot_accuracy(result, n=4096, acclevel=0.1, labellist=['M1', 'M2', 'M3'], logscale=False)






# Batch version
timeparams = [0.008, 0.006, 0.004]
batches = [1, 64, 128]
accparams = [[-1,0.2], [-1, 0.3], [-1, 0.4]]
result = []
for mid, m in enumerate(timeparams):
    
    PM = performanceModel(worker=128, batch=batches[mid], n=4096)
    
    PM.gen_gentime(timeparams[mid], typeGen='constant')
    PM.gen_simtime(0.0001, 0.0001, typeSim='normal')
    PM.gen_accuracy(accparams[mid][0], accparams[mid][1], typeAcc='exponential')
    
    PM.simulate()
    
    PM.summarize()
    result.append(PM)

plot_accuracy(result, n=4096, acclevel=0.1, labellist=['b=1', 'b=64', 'b=128'], logscale=False)

