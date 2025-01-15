import ROOT
import json
import re
from tqdm import tqdm
from cutManager import cutManager
import math

class skimManager():
    def __init__(self):
        pass

    def deltaR(self, eta1, phi1, eta2, phi2):
        """Calculate delta R."""
        dEta = eta1 - eta2
        dPhi = abs(phi1 - phi2)
        if dPhi > math.pi:
            dPhi -= 2 * math.pi
        return math.sqrt(dEta**2 + dPhi**2)

    def objectSelection(self, eta, phi, pt, additionalCriteria=None):
        """
        Placeholder for object selection. 
        Add any additional criteria in `additionalCriteria` as a callable.
        """
        if additionalCriteria:
            return additionalCriteria(eta, phi, pt)
        return True  # Default: Keep all objects

    def skimAFile(self,
                  fileName,
                  branchCancelationFileName,
                  theCutFile,
                  outputFileName,
                  deltaR_threshold=0.4,
                  muonSelection=None,
                  tauSelection=None):

        try:
            theLoadFile = ROOT.TFile(fileName)
            theInputTree = theLoadFile.Events
            theRunTree = theLoadFile.Runs
        except:
            try:
                theLoadFile = ROOT.TFile.Open(fileName)
                theInputTree = theLoadFile.Events
                theRunTree = theLoadFile.Runs
            except:
                hdfsFileName = fileName.replace('/hdfs', 'root://cmsxrootd.hep.wisc.edu//')
                try:
                    theLoadFile = ROOT.TFile.Open(hdfsFileName)
                    theInputTree = theLoadFile.Events
                    theRunTree = theLoadFile.Runs
                except:
                    print("Failed to load the files properly!")
                    exit(-1)

        print('Loaded the file, and retrieved the trees...')
        
        branchCancelations = None
        if branchCancelationFileName != None:
            with open(branchCancelationFileName) as branchCancelationFile:
                branchCancelationJSON = json.load(branchCancelationFile)
            try:
                branchCancelations = [re.compile(branchCancelationJSON[x]) for x in branchCancelationJSON]
            except Exception as err:
                print('Failed to make proper RE\'s for branch cancelations')
                print(err)
                exit(-1)

        theCutManager = cutManager(theInputTree, theCutFile)
        
        nBranches = theInputTree.GetNbranches()
        listOfBranches = theInputTree.GetListOfBranches()
        if branchCancelations:
            for branchIndex in range(nBranches):
                for REIndex in range(len(branchCancelations)):
                    theRE = branchCancelations[REIndex]
                    theBranchName = listOfBranches[branchIndex].GetName()
                    if theRE.search(theBranchName):
                        theInputTree.GetBranch(theBranchName).SetStatus(0)

        theOutputFile = ROOT.TFile(outputFileName, 'RECREATE')

        theCutFlow = theCutManager.createCutFlowHistogram()
        theCutFlow.Write('cutflow', ROOT.TFile.kOverwrite)

        finalCut = theCutManager.createAllCuts()
        print('Performing final tree copy...')

        # Create new tree to hold filtered events
        theOutputTree = theInputTree.CloneTree(0)

        for event in tqdm(theInputTree, total=theInputTree.GetEntries()):
            # Apply event-level cuts
            if not eval(finalCut):  # Replace eval() if finalCut isn't a string expression
                continue

            # Retrieve muons and taus with placeholder selection
            muons = [(event.Muon_eta[i], event.Muon_phi[i], event.Muon_pt[i]) 
                     for i in range(event.nMuon)
                     if self.objectSelection(event.Muon_eta[i], event.Muon_phi[i], event.Muon_pt[i], muonSelection)]
            taus = [(event.Tau_eta[i], event.Tau_phi[i], event.Tau_pt[i]) 
                    for i in range(event.nTau)
                    if self.objectSelection(event.Tau_eta[i], event.Tau_phi[i], event.Tau_pt[i], tauSelection)]

            # Apply delta R cut
            passesDeltaRCut = True
            for muon_eta, muon_phi, _ in muons:
                for tau_eta, tau_phi, _ in taus:
                    if self.deltaR(muon_eta, muon_phi, tau_eta, tau_phi) < deltaR_threshold:
                        passesDeltaRCut = False
                        break
                if not passesDeltaRCut:
                    break

            if passesDeltaRCut:
                theOutputTree.Fill()

        theOutputFile.cd()
        print('Writing all output...')
        theOutputTree.Write('Events', ROOT.TFile.kOverwrite)

        theRunTree.CopyTree('').Write('Runs', ROOT.TFile.kOverwrite)

        theOutputFile.Write()
        theOutputFile.Close()
        theLoadFile.Close()

