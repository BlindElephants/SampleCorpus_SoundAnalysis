# Sample Corpus Sound Analysis

The code in this repo is used to analyze sound samples that are used during TidalCycles live coding performances. From these analyses, a number of unsupervised machine learning methods are applied to organize sound data, allowing for different ways of traversing a given sample set, both for the human user and for an AI agent.

## Requirements

- Python 3.6 or 3.7
- Librosa
    - can be installed via pip / pip3: ```pip install librosa```

## Analyzing Sound Samples

Sound samples should be organized as they would be for TidalCycles use (i.e., with SuperDirt/SuperCollider).

```
SoundSamplesFolder
        808_cl
                808_cl_1.WAV
                808_cl_2.WAV
                808_cl_3.WAV
                allOtherSamplesIn_808_cl.WAV
        808_clave
        808_congas
        808bd
        arpy
        baa
                baa_1.WAV
                baa_2.WAV
        bass
        bbq
        cp
        dr
        glitch
        
        
...etc...
```

run ```Analyze_SampleSet.py``` with python3, specifying the location of this SoundSamplesFolder, for example:

```
python Analyze_SampleSet.py /home/path/to/SoundSamplesFolder
```

## Output

A folder will be created alongside this script, called ```SampleAnalysisData```. 

Within this folder will be 
- a file called ```OutputAnalysis.json```
    - has a list of all samples analyzed
- a folder called ```Samples``` which has .json files for each analyzed sound sample

Compress the entire ```SampleAnalysisData``` folder and upload it to my Google Drive folder.