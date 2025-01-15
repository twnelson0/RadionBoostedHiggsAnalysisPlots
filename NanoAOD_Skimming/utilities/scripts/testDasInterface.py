#!/usr/bin/env python
#test dasInterface.py

from bbtautauAnalysisScripts.utilities.dasInterface import dasInterface
import json

def main():
    theDasInterface = dasInterface()

    dasQuery = 'dataset=/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2/MINIAODSIM'
    print(dasQuery + '\n' + theDasInterface.performDASQuery(dasQuery))
    
    theDasInterface.getCompleteDictionaryOfFilesFromPathList(['RandomString'])
    
    fileJSON = theDasInterface.getCompleteDictionaryOfFilesFromPathList(['/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2/MINIAODSIM','/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2/MINIAODSIM','/MET/Run2016B-21Feb2020_ver2_UL2016_HIPM-v1/MINIAOD'])
    print(json.dumps(fileJSON,sort_keys=True,indent=4))
    

if __name__ == '__main__':
    main()
