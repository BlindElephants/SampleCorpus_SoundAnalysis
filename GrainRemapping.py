import json
import os
import numpy as np

SampleJsonDataFolder=os.path.join("SampleAnalysisData", "Samples")
assert os.path.isdir(SampleJsonDataFolder), "[ERROR] invalid folder."

listDir = os.listdir(SampleJsonDataFolder)
listDir.sort()

for idx, i in enumerate(listDir):
    print("{} of {}\t\t{}".format(idx, len(listDir), i))
    dat = None
    with open(os.path.join(SampleJsonDataFolder, i), "r") as f:
        try:
            dat = json.load(f)
        except json.JSONDecodeError:
            print("Skip")
            dat=None
    if dat is not None:
        if 'rearranged' not in dat:
            dat['rearranged']=[]

            stft = np.array(dat['stft'])
            cqt  = np.array(dat['cqt'])
            sCentroid=np.array(dat['spec centroid'])
            sBandwidth=np.array(dat['spec bandwidth'])
            sContrast=np.array(dat['spec contrast'])
            sFlatness=np.array(dat['spec flatness'])
            sRolloff =np.array(dat['spec rolloff'])

            assert stft.shape[1]==cqt.shape[1], "STFT / CQT shape mismatch"
            assert stft.shape[1]==sCentroid.shape[1], "STFT / Spectral Centroid shape mismatch"
            assert stft.shape[1]==sBandwidth.shape[1], "STFT / Spectral Bandwidth shape mismatch"
            assert stft.shape[1]==sContrast.shape[1], "STFT / Spectral Contrast shape mismatch"
            assert stft.shape[1]==sFlatness.shape[1], "STFT / Spectral Flatness shape mismatch"
            assert stft.shape[1]==sRolloff.shape[1], "STFT / Spectral Rolloff shape mismatch"

            for x in range(stft.shape[1]):
                dat['rearranged'].append(np.concatenate((stft[:,x], cqt[:,x], sCentroid[:,x], sBandwidth[:,x], sContrast[:,x], sFlatness[:,x], sRolloff[:,x]), axis=0).tolist())
            with open(os.path.join(SampleJsonDataFolder, i), "w") as f:
                json.dump(dat, f)
        else:
            print("Already completed. Skipping")
    