#Class for managing the dag input/output system for farmoutanalysisjobs

import json
import os

class dagInterface():
    def __init__(self):
        pass
    #Create the dag output area for a sepcified dataset/campaign
    #Will take a dataset name and a campaign name,
    #And a path if necessary, for non wisconsin computers that may have separate working/scratch areas
    def createCampaignDatasetDagArea(self,theCampaign, theDataset, submissionName, thePath = '/nfs_scratch/'+os.environ['USER']+'/'):
        dagLocation = thePath+'/'+submissionName+'/'+theCampaign+'/'+theDataset+'/dags'
        os.system('mkdir -p '+dagLocation+'/daginputs')
        return dagLocation
    def createInputFileFromJSONList(self,theDataset,theList,theFileLocation):
        inputFileName = theFileLocation+theDataset+'_inputFiles.txt'
        theInputListFile = open(theFileLocation+theDataset+'_inputFiles.txt','w')
        for filePath in theList:
            theInputListFile.write(filePath+'\n')
        return inputFileName
