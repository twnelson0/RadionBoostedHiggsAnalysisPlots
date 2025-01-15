#!/usr/bin/env python
#script for skimming large nano-aod ntuples on hdfs down to size.
import ROOT 
ROOT.PyConfig.IgnoreCommandLineOptions = True
import argparse
import glob
import re
import json
import math
import datetime
import os
from tqdm import tqdm
from bbtautauAnalysisScripts.boostedTauNanoMaker.skimModules.skimManager import skimManager

def main(args):
    print('Setting up the skim...')
    #first things first, let's load the json
    jsonInputFile = open(args.skimFileConfiguration)
    jsonFileGlobs = json.load(jsonInputFile)
    jsonInputFile.close()
    
    try:
        datasetREs = [re.compile(x) for x in args.datasets]
    except Exception as err:
        print('Failed to make proper RE\'s for datasets:')
        print(err)
        exit(1)    


    print('Performing skim...')
    for datasetIndex in tqdm(range(len(jsonFileGlobs)), desc = 'Datasets'):
        #let's figure out the list of files we need to operate on
        listOfGlobs = []
        globKey = jsonFileGlobs.keys()[datasetIndex]
        for REIndex in tqdm(range(len(datasetREs)),leave = False, desc = 'RE Check: '+globKey):
            if datasetREs[REIndex].search(globKey):
                globsToAdd = glob.glob(jsonFileGlobs[globKey])
                if globsToAdd == []:
                    print('empty glob: '+globKey+', '+jsonFileGlobs[globKey])
                else:
                    listOfGlobs += globsToAdd
        if listOfGlobs == []: #we found no files to work with
            continue
            
        if args.prepareCondorSubmission: #prepare submission files for condor sub, and submit
            #farmoutAnalysisJobs --fwklite --infer-cmssw-path --input-files-per-job=1 --input-file-list=inputFileList.txt --assume-input-files-exist --input-dir=/ skimTest_20 boostedTauNanoMaker/scripts/singleFileSkimForSubmission.py -- '--inputFile=$inputFileNames' "--theCutFile=$CMSSW_BASE/src/bbtautauAnalysisScripts/metaData/skimmingCuttingConfigurations/testCutConfiguration.json" "--outputFileName=Test.root"
            #make the dag area so we can resubmit in the future
            overallSubmitDir = args.submitDirPath+'/'+globKey+'_'+datetime.datetime.now().strftime('%d%B%y_%H%M_skim')+ ('' if args.skimSuffix == '' else '_'+args.skimSuffix)

            dagLocation = overallSubmitDir+'/dags'
            os.system('mkdir -p '+dagLocation+'/daginputs')

            inputFileTextName = globKey+'_input.txt'
            inputFileText = open(inputFileTextName,'w+')
            inputFileText.write(''.join(x+'\n' for x in listOfGlobs))
            inputFileText.close()
            
            #we need the actual names of the cutt and branch cancelation files, because
            #now they need to be local
            
            cutFileName = args.skimCutConfiguration.split("/")[len(args.skimCutConfiguration.split("/"))-1]
            branchCancelationFileName = (args.skimBranchCancelations.split("/")[len(args.skimBranchCancelations.split("/"))-1] if args.skimBranchCancelations != None else '')


            commandList = [
                'farmoutAnalysisJobs --fwklite --infer-cmssw-path --input-files-per-job=1',
                '--input-file-list='+inputFileTextName,
                '--assume-input-files-exist',
                '"--submit-dir=%s"' % (overallSubmitDir+'/submit'),
                '--output-dag-file=%s' %(dagLocation+'/dag'),
                #'--output-dir=%s' % overallSubmitDir,
                '--output-dir=%s' % (args.destination + '/' +globKey+'_'+datetime.datetime.now().strftime('%d%B%y_%H%M_skim')+ ('' if args.skimSuffix == '' else '_'+args.skimSuffix)),
                '--extra-inputs="'+args.skimCutConfiguration+' '+args.skimBranchCancelations+'"',
                '--input-dir=/',
                globKey+'_'+datetime.datetime.now().strftime('%d%B%y_%H%M_skim')+ ('' if args.skimSuffix == '' else '_'+args.skimSuffix),#name of what we're doing. 
                os.environ['CMSSW_BASE']+'/src/bbtautauAnalysisScripts/boostedTauNanoMaker/scripts/singleFileSkimForSubmission.py',
                '--', #seperates options for the script from the submission options
                '\'--inputFile=$inputFileNames\'',
                ''+('"--branchCancelationFile='+branchCancelationFileName+'"' if args.skimBranchCancelations != None else ''),
                '"--theCutFile='+cutFileName+'"',
                '\'--outputFileName=$outputFileName\'',
                
            ]
            theCommand  = ' '.join(commandList)
            #os.system(theCommand)
            print()
            print()
            print(theCommand)
        else: #attempt the skim locally
            #load the files and get to work on them
            for loadFileIndex in tqdm(range(len(listOfGlobs)), desc = 'Process: '+globKey):
                outputFileName = globKey+'_'+('{:0'+str(int(math.floor(math.log10(len(jsonFileGlobs))))+1)+'}').format(loadFileIndex)+'.root'
                outputFileName = args.destination+'/'+outputFileName
                loadFileName = listOfGlobs[loadFileIndex]
                theSkimManager = skimManager()
                theSkimManager.skimAFile(fileName = loadFileName,
                                         branchCancelationFileName = args.skimBranchCancelations,
                                         theCutFile = args.skimCutConfiguration,
                                         outputFileName = outputFileName)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Skim HDFS nanoAOD ntuples down to size in a configurable way')
    ##Added by GP
    #parser.add_argument('--type',nargs='?',required=True,help='Where these are data or mc files, accordingly we will store the gen weights on cutflow',choices=['Data','MC'],type=str)
    ####
    parser.add_argument('--skimFileConfiguration',nargs='?',required=True,help='JSON file describing the paths/files to be skimmed',type=str)
    parser.add_argument('--datasets',nargs='+',default=['.*'],help='select datasets from the configuration file')
    parser.add_argument('--skimCutConfiguration',nargs='?',required=True,help='JSON file describing the cuts to be implemented in the files')
    parser.add_argument('--skimBranchCancelations',nargs='?',help='JSON file describing the branches that do not need to be ported around with the skimmed nanoAOD file')
    parser.add_argument('--destination',nargs='?',type=str,required=True,help='destination path for resut files')
    parser.add_argument('--skimSuffix',nargs='?',type=str,default='',help='String to identify the set of skims with')
    parser.add_argument('--prepareCondorSubmission',action='store_true',help='Instead of attempting the overall skimming on a local CPU, prepare a "Combine-style" submission to condor')
    parser.add_argument('--submitDirPath',nargs='?',default='/nfs_scratch/'+os.environ['USER']+'/',help='usually a place in nfs_scratch where the submit files wil be stored. Please do not end it with /')
    args = parser.parse_args()
    main(args)
