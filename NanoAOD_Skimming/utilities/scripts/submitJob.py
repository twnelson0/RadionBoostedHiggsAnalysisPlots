#a utility for submitting large batches of jobs to condor based on the file list JSONs
#the overall goal of this script will be to have a few things happen
#1.) the script should take care of creating a space to be used on /hdfs/store/user/
#2.) it should interact with the condor dag system to make rescue of jobs possible
#3.) Output should get written straight to the created /hdfs/store/user/ area
#4.) Should create a script that creates a farmout job. The script should be modifiable so that parameters can be tuned before a final submission is made

import argparse
from bbtautauAnalysisScripts.utilities.dagInterface import dagInterface
import datetime
import os
import json
import re
import sys

def main(args):
    #Load the json file list
    jsonInputFile = open(args.inputFile)
    jsonFileList = json.load(jsonInputFile)

    finalSubmissionScript = open(args.submissionName+'_finalSubmission.sh','w')
    finalSubmissionScript.write('#Submission generated with command: '+' '.join(sys.argv))
    finalSubmissionScript.write('\n')

    #make the dag area and get ready for submission
    try:
        campaignPattern = re.compile(args.campaigns)
    except Exception as err:
        print('Something failed in compiling the RE used for campaigns:')
        print(err)
        exit(1)
    try:
        datasetPattern = re.compile(args.datasets)
    except Exception as err:
        print('Something failed in compiling the RE used for datasets:')
        print(err)
        exit(1)

    #filter the list into what we need
    listOfMatchingFiles = list()
    theDagInterface = dagInterface()
    for campaign in jsonFileList:
        if not campaignPattern.search(campaign):
            continue

        for dataset in jsonFileList[campaign]:
            if not datasetPattern.search(dataset):
                continue

            dagPath = theDagInterface.createCampaignDatasetDagArea(
                theCampaign = campaign,
                theDataset = dataset,
                submissionName = args.submissionName,
                thePath = args.submissionPath)
            #while we're here, make the input file list
            inputFilePath = theDagInterface.createInputFileFromJSONList(
                theDataset = dataset,
                theList = jsonFileList[campaign][dataset][:args.numberOfFiles],
                theFileLocation = dagPath+'/daginputs/'
            )
            #we should now have a fully setup dag input area. Let's create a script entry
            command = [
                'farmoutAnalysisJobs --vsize-limit 8000 --memory-requirements=4000',
                '--infer-cmssw-path',
                '"--submit-dir=%s"' % (args.submissionPath+'/'+args.submissionName+'/'+campaign+'/'+dataset+'/'+'submit'),
                '"--output-dag-file=%s"' % (dagPath+'/dag'),
                #'"--output-dir=%s"' % ('gsiftp://cms-lvs-gridftp.hep.wisc.edu:2811//hdfs/store/user/'+os.environ['USER']+'/'+args.submissionName+'/'+campaign+'/'+dataset+'/'),
                '"--output-dir=%s"' % ('/store/user/'+os.environ['USER']+'/'+args.submissionName+'/'+campaign+'/'+dataset+'/'),
                '--input-files-per-job=%i' % args.filesPerJob,
                '--input-file-list=%s'%inputFilePath,
                '--assume-input-files-exist',
                '--input-dir=/',
                '%s-%s' % (args.submissionName, dataset), #job ID
                args.config,
                "'inputFiles=$inputFileNames'",
                "'outputFile=$outputFileName'",                
            ]
            finalFarmoutCommand = ' '.join(command)+'\n'
            finalSubmissionScript.write('#'+dataset+' submission command\n')
            finalSubmissionScript.write(finalFarmoutCommand)
            finalSubmissionScript.write('\n')
    #close the file.
    finalSubmissionScript.close()
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Create submission file to re-nano-ify files')
    parser.add_argument('--config',nargs='?',required=True,help='configuration to run to reNANO files',type=str)
    parser.add_argument('--submissionName',nargs='?',default=datetime.datetime.now().strftime('%d%B%y_%H%M_Submission'),help='Name to store scratch submissions under, and to store result tuples in the store.')
    parser.add_argument('--inputFile', nargs='?',required=True,help='input json to create dag input file lists from.')
    parser.add_argument('--submissionPath',nargs='?',default='/nfs_scratch/'+os.environ['USER']+'/')
    parser.add_argument('--datasets',default='.*',help='RE to specify desired datasets from the JSON.',nargs='?',type=str)
    parser.add_argument('--campaigns',default='.*',help='RE to specify desired campaigns from the JSON. Accepts multiple RE\'s and will accept a match to any.',nargs='?',type=str)
    parser.add_argument('--numberOfFiles',nargs='?',type=int,default=None,help='Specify the number of files to process in the job')#note, list[:None] get you the whole list!, not none of it (i.e. we default to the whole list here)
    parser.add_argument('--filesPerJob',nargs='?',type=int,default=1,help='Number of input files each job should handle')

    args = parser.parse_args()

    main(args)
