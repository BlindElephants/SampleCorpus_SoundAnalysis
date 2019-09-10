import os, json
import numpy as np
import librosa
import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("SoundSamplesFolder", type=str, help="Path to folder containing all Tidal samples. This folder should have inside of it a number of folders which each contain WAV samples.")
args = parser.parse_args()

assert os.path.exists(args.SoundSamplesFolder), "[ERROR] Folder not found: {}".format(args.SoundSamplesFolder)

SampleAnalysisDataFolder=os.path.join(os.path.dirname(__file__), "SampleAnalysisData")
os.makedirs(SampleAnalysisDataFolder, exist_ok=True)
os.makedirs(os.path.join(SampleAnalysisDataFolder, "Samples"), exist_ok=True)

SamplesPaths = []

OutputAnalysisFileName=os.path.join(SampleAnalysisDataFolder, "OutputAnalysis.json")

if os.path.exists(OutputAnalysisFileName):
    with open(OutputAnalysisFileName, "r") as f:
        OutputAnalysisData = json.load(f)
else:
    OutputAnalysisData = {
        'start time': str(datetime.datetime.now()),
        'end time': None,
        'files parsed': [],
        'finished': False
    }

with open(OutputAnalysisFileName, "w") as f:
    json.dump(OutputAnalysisData, f)

listDir = os.listdir(args.SoundSamplesFolder)
listDir.sort()
for d in listDir:
    p=os.path.join(args.SoundSamplesFolder, d)
    if os.path.isdir(p):
        samplesList = os.listdir(p)
        samplesList.sort()
        for i,s in enumerate(samplesList):
            if s.split(".")[-1] in ['wav', 'WAV', 'aif', 'aiff', 'AIF', 'AIFF']:
                SamplesPaths.append([os.path.join(p,s), s, os.path.join(d,s), d+":"+str(i)])

print("Found {} samples.".format(len(SamplesPaths)))

SaveCheckpointEvery=100

for idx, samples in enumerate(SamplesPaths):

    if idx==8914: continue

    if samples[2] not in OutputAnalysisData['files parsed']:
        print("Analyzing sample {} of {}".format(idx, len(SamplesPaths)), end='')

        try:
            y, sr=librosa.load(samples[0])
            if y.shape[0]>0:
                stft=np.abs(librosa.core.stft(y))
                cqt =np.abs(librosa.core.cqt(y,sr=sr))
                specCentroid=librosa.feature.spectral_centroid(y,sr=sr)
                specBandwidth=librosa.feature.spectral_bandwidth(y,sr=sr)
                specContrast =librosa.feature.spectral_contrast(y,sr=sr)
                specFlatness =librosa.feature.spectral_flatness(y=y)
                specRolloff  =librosa.feature.spectral_rolloff(y=y,sr=sr)
                duration=librosa.core.get_duration(y,sr)

                ThisSampleData={
                    'sample_filename': samples[2],
                    'sample_tidalName': samples[3],
                    'duration': duration,
                    'stft_m': np.mean(stft, axis=1).tolist(),
                    'stft_std': np.std(stft, axis=1).tolist(),
                    'stft_sum': np.sum(stft, axis=1).tolist(),
                    'cqt_m': np.mean(cqt, axis=1).tolist(),
                    'cqt_std': np.std(cqt, axis=1).tolist(),
                    'cqt_sum': np.sum(cqt, axis=1).tolist(),
                    'spec centroid_m': np.mean(specCentroid, axis=1).tolist(),
                    'spec centroid_std': np.std(specCentroid,axis=1).tolist(),
                    'spec centroid_sum': np.sum(specCentroid,axis=1).tolist(),
                    'spec bandwidth_m': np.mean(specBandwidth, axis=1).tolist(),
                    'spec bandwidth_std': np.std(specBandwidth,axis=1).tolist(),
                    'spec bandwidth_sum': np.sum(specBandwidth,axis=1).tolist(),
                    'spec contrast_m': np.mean(specContrast, axis=1).tolist(),
                    'spec contrast_std':np.std(specContrast, axis=1).tolist(),
                    'spec contrast_sum':np.sum(specContrast, axis=1).tolist(),
                    'spec flatness_m':np.mean(specFlatness, axis=1).tolist(),
                    'spec flatness_std':np.std(specFlatness,axis=1).tolist(),
                    'spec flatness_sum':np.sum(specFlatness,axis=1).tolist(),
                    'spec rolloff_m': np.mean(specRolloff, axis=1).tolist(),
                    'spec rolloff_std': np.std(specRolloff,axis=1).tolist(),
                    'spec rolloff_sum': np.sum(specRolloff,axis=1).tolist(),
                    'generated': str(datetime.datetime.now())
                }

                with open(os.path.join(SampleAnalysisDataFolder, "Samples", samples[1]+".json"), "w") as f:
                    json.dump(ThisSampleData, f)
                OutputAnalysisData['files parsed'].append(samples[2])
                print("[DONE]")

                if (idx+1)%SaveCheckpointEvery==0:
                    print("Saving checkpoint.")
                    with open(OutputAnalysisFileName, "w") as f:
                        json.dump(OutputAnalysisData, f)
            else:
                print("Abort.")
        except ValueError:
            print("[ValueError]")
        except RuntimeError:
            print("[RuntimeError]")

OutputAnalysisData['finished']=True
OutputAnalysisData['end time']=str(datetime.datetime.now())
with open(OutputAnalysisFileName, "w") as f:
        json.dump(OutputAnalysisData, f)