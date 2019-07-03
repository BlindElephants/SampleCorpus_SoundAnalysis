import os

try:
    print("Beginning Sample Analysis")
    os.system("python GrainAnalysis.py")
    print("[DONE]")

    print("Repacking and Flattening analysis")
    os.system("python repackedLibrosaAnalysis.py")
    os.system("python flattenPreppedData.py")
    print("[DONE]")

    print("Generating figure for sample lengths")
    os.system("python CountLengths.py")
    print("[DONE]")

    print("Generating visualization of tSNE embeddings")
    os.system("python viz_tSNEs.py")
    print("[DONE]")

    print("generating k-means clusters")
    os.system("python kmeans_tSNE.py")
    print("[DONE]")

    print("training neural network for tSNE embedding coordinates and k-means clustering ids")
    os.system("python Network_Predictors/train.py")
    print("[DONE]")
except KeyboardInterrupt:
    print("exiting early.")
    quit()