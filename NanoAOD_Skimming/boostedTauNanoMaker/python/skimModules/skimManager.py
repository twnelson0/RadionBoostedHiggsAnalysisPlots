#!/user/bin/env python
#A class that will act as a master manager for the file skimming process
#should abstract the act of skimming a single file, and this should allow for easier submission

import ROOT
import json
import re
from tqdm import tqdm
from cutManager import cutManager

class skimManager():
    def __init__(self):
        pass
    def skimAFile(self,
                  fileName,
                  #branchCancelations,
                  branchCancelationFileName,
                  theCutFile,
                  outputFileName):

        try:
            #print("trying to open the file")
	    #hdfsFileName = fileName.replace('/hdfs','root://cmsxrootd.hep.wisc.edu//')
            theLoadFile = ROOT.TFile(fileName)
            theInputTree = theLoadFile.Events
            theRunTree = theLoadFile.Runs
        except: #we failed to open the file properly, so let's try it the other way
            try:
		#print ("Went to Ecept statement")
                theLoadFile = ROOT.TFile.Open(fileName)
                theInputTree = theLoadFile.Events
                theRunTree = theLoadFile.Runs
            except: #we have failed again to find the file. Let's try to open it this way
                hdfsFileName = ''
                if 'xrootd' in fileName: #we're already tried toopen xrootd style. We're done here
                    hdfsFileName = fileName.replace('hdfs/','')
                else:
                    hdfsFileName = fileName.replace('/hdfs','root://cmsxrootd.hep.wisc.edu//')
                    print(hdfsFileName)
                print(hdfsFileName)
                print("Last attempt to load the file at /hdfs/ with: "+hdfsFileName)
                try:
                    theLoadFile = ROOT.TFile.Open(hdfsFileName)
                    theInputTree = theLoadFile.Events
                    theRunTree = theLoadFile.Runs
                except:
                    print("Failed to load the files properly!")
                    print("Exiting with code -1")
                    exit(-1)

        print('Loaded the file, and retrieved the trees...')
        #load the branch cancelation FILE
        branchCancelations = None
        if branchCancelationFileName != None:
            branchCancelationFile = open(branchCancelationFileName)
            branchCancelationJSON = json.load(branchCancelationFile)
            branchCancelationFile.close()
            
            try:
                branchCancelations = [re.compile(branchCancelationJSON[x]) for x in branchCancelationJSON]
            except Exception as err:
                print('Failed to make proper RE\'s for branch cancelations')
                print(err)
                exit(-1)

        theCutManager = cutManager(theInputTree,theCutFile)
        
        nBranches = theInputTree.GetNbranches()
        listOfBranches = theInputTree.GetListOfBranches()
        #now we loop over branches, and the branch cancellation REs
        #if one of the RE's matches, disable the branch.
        if branchCancelations != None:
            for branchIndex in range(nBranches):
                for REIndex in range(len(branchCancelations)):
                    theRE = branchCancelations[REIndex]
                    theBranchName = listOfBranches[branchIndex].GetName()
                    if theRE.search(theBranchName):
                        theInputTree.GetBranch(theBranchName).SetStatus(0) # close out the branch and don't copy it
        #all branches should be canceled now
        #now let's set up a new file/tree
        theOutputFile = ROOT.TFile(outputFileName,'RECREATE') #this has to happen last to keep things associated with it

        theCutFlow = theCutManager.createCutFlowHistogram()
        theCutFlow.Write('cutflow', ROOT.TFile.kOverwrite)

        #now, let's do the magic copy
        finalCut = theCutManager.createAllCuts()
        print('Performing final tree copy...')
        theOutputTree = theInputTree.CopyTree(finalCut)

        theOutputFile.cd()
        print('Writing all output...')
        theOutputTree.Write('Events', ROOT.TFile.kOverwrite)
        
        #let's get the old run tree
        theRunTree.CopyTree('').Write('Runs',ROOT.TFile.kOverwrite)

        #okay, now let's get the remaining object in the nano file and

        theOutputFile.Write()
        theOutputFile.Close()
        theLoadFile.Close()        
