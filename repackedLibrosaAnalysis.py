import os
import json
import numpy as np

samplesData = os.path.join(os.path.dirname(__file__), "SampleAnalysisData", "Samples")

ld = os.listdir(samplesData)
ld.sort()
print("Samples found: {}".format(len(ld)))

allDat = {}

for x in range(len(ld)):
    print("File {} of {}".format(x, len(ld)))
    with open(os.path.join(samplesData, ld[x]), "r") as f:
        d = json.load(f)
    outDat = {}
    outDat['duration'] = d['duration']
    outDat['sample_filename']=d['sample_filename']
    for key, val in d.items():
        print(key)
        if key not in ['duration', 'sample_filename']:
            dat = np.array(val)
            outDat[key+'_m']  =np.mean(dat, axis=1).tolist()
            outDat[key+'_std']=np.std(dat, axis=1).tolist()
            outDat[key+'_sum']=np.sum(dat, axis=1).tolist()
    allDat[ld[x]] = outDat
    quit()
    
with open("allDat_Prepped.json", "w") as f:
    json.dump(allDat,f)
