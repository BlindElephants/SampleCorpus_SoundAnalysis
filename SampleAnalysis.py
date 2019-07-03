import os
import numpy as np
import json
import librosa
import datetime

SoundSamplesFolder=os.path.join("/home", "blindelephants", "slab", "SoundSamples")
SamplesPaths = []

SampleAnalysisDataFolder=os.path.join(os.path.dirname(__file__), "SampleAnalysisData")

os.makedirs(SampleAnalysisDataFolder, exist_ok=True)
os.makedirs(os.path.join(SampleAnalysisDataFolder, "Samples"), exist_ok=True)
JsonMasterFileName = os.path.join(os.path.dirname(__file__), "JsonMaster.json")

JsonMasterData = {
    'start time': str(datetime.datetime.now()),
    'end time': None,
    'time elapsed': None,
    'files parsed': [],
    'finished': False
}

START_TIME = datetime.datetime.now()

with open(JsonMasterFileName, "w") as f:
    json.dump(JsonMasterData, f)

if os.path.exists(SoundSamplesFolder):
    print("Found Sound Samples at:\t{}".format(SoundSamplesFolder))
    listDir = os.listdir(SoundSamplesFolder)
    listDir.sort()
    for d in listDir:
        p = os.path.join(SoundSamplesFolder, d)
        if os.path.isdir(p):
            samplesList = os.listdir(p)
            samplesList.sort()            
            for s in samplesList:
                if s.split(".")[-1] in ['wav', 'WAV', 'aif', 'aiff', 'AIF', 'AIFF']:
                    SamplesPaths.append([os.path.join(p, s), s, os.path.join(d,s)])
else:
    print("[ERROR] Folder not found: {}".format(SoundSamplesFolder))

print("Found {} .WAV samples.".format(len(SamplesPaths)))


for idx, samples in enumerate(SamplesPaths):
    print("Sample {} of {}\t\t".format(idx, len(SamplesPaths)), end='')
    y, sr = librosa.load(samples[0])

    if y.shape[0]>0:
        stft = librosa.core.stft(y)
        cqt = np.abs(librosa.core.cqt(y, sr=sr)).tolist()
        specCentroid=librosa.feature.spectral_centroid(y, sr=sr).tolist()
        specBandwidth=librosa.feature.spectral_bandwidth(y, sr=sr).tolist()
        specContrast= librosa.feature.spectral_contrast(y, sr=sr).tolist()
        specFlatness =librosa.feature.spectral_flatness(y=y).tolist()
        specRolloff =librosa.feature.spectral_rolloff(y=y, sr=sr).tolist()
        
        duration = librosa.core.get_duration(y, sr)

        ThisSampleData = {
            'sample_filename': samples[2],
            'duration': duration,
            'stft': np.abs(stft).tolist(),
            'cqt': cqt,
            'spec centroid': specCentroid,
            'spec bandwidth': specBandwidth,
            'spec contrast': specContrast,
            'spec flatness': specFlatness,
            'spec rolloff': specRolloff
        }

        print(samples[2])

        with open(os.path.join(SampleAnalysisDataFolder, "Samples", samples[1]+".json"), "w") as f:
            json.dump(ThisSampleData, f)

        JsonMasterData['files parsed'].append(samples[0])

        with open(JsonMasterFileName, "w") as f:
            json.dump(JsonMasterData, f)

        print("[DONE]")
    else:
        print("Abort.")

JsonMasterData['finished']=True
JsonMasterData['end time']=str(datetime.datetime.now())
JsonMasterData['time elapsed'] = str(datetime.datetime.now()-START_TIME)
with open(JsonMasterFileName, "w") as f:
    json.dump(JsonMasterData, f)

