import ROOT
import math 
import time
#from plot_dict import *


#Open and count and store the tau properties for the first file

File1 = ROOT.TFile("WJets.root")
File1_tree = File1.Get('Events')
nEntries = File1_tree.GetEntries()
nTaus_File_1 = 0
nboostedTaus_File_1 = 0

tau_pt_1 = []
btau_pt_1 = []




for x in range(nEntries):
    File1_tree.GetEntry(x)
    #nTaus_File_1 = nTaus_File_1 + File1_tree.nTau
    #nboostedTaus_File_1 = nboostedTaus_File_1 + File1_tree.nboostedTau

    for i in range(len(File1_tree.Tau_pt)):
        #if (File1_tree.Tau_idDecayModeNewDMs[i] and (File1_tree.Tau_LooseCombinedIsolationDeltaBetaCorr3Hits[i] or File1_tree.Tau_VVVLooseDeepTau2017v2p1VSjet[i] or (File1_tree.Tau_rawIsodR03[i] < 2.5))):
        if (File1_tree.Tau_idDecayModeNewDMs[i] and (File1_tree.Tau_VVVLooseDeepTau2017v2p1VSjet[i])):
            tau_pt_1.append(File1_tree.Tau_pt[i])
    for i in range(len(File1_tree.boostedTau_pt)):
        if (File1_tree.boostedTau_idDecayModeNewDMs[i] and (File1_tree.boostedTau_VVLooseIsolationMVArun2017v2DBoldDMwLT2017[i] or File1_tree.boostedTau_VVLooseIsolationMVArun2017v2DBoldDMdR0p3wLT2017[i] or File1_tree.boostedTau_VVLooseIsolationMVArun2017v2DBnewDMwLT2017[i])):
            btau_pt_1.append(File1_tree.boostedTau_pt[i])

nTaus_File_1 = len(tau_pt_1)
nboostedTaus_File_1 = len(btau_pt_1)

File1.Close()

nTaus_File_2 = 0
nboostedTaus_File_2 = 0
tau_pt_2 = []
btau_pt_2 = []

File2 = ROOT.TFile("WJets_all.root")
File2_tree = File2.Get('Events')
nEntries = File2_tree.GetEntries()

for x in range(nEntries):
    File2_tree.GetEntry(x)
#    nTaus_File_2 = nTaus_File_2 + File2_tree.nTau
#    nboostedTaus_File_2 = nboostedTaus_File_2 + File2_tree.nboostedTau

    for i in range(len(File2_tree.Tau_pt)):
        tau_pt_2.append(File2_tree.Tau_pt[i])
    for i in range(len(File1_tree.boostedTau_pt)):
        btau_pt_2.append(File2_tree.boostedTau_pt[i])

nTaus_File_2 = len(tau_pt_2)
nboostedTaus_File_2 = len(btau_pt_2)

File2.Close()

print ("Number of taus in the First file = ",nTaus_File_1," Number of bTaus in the First File = ",nboostedTaus_File_1)
print ("Number of taus in the Second file = ",nTaus_File_2," Number of bTaus in the Second File = ",nboostedTaus_File_2)
if (tau_pt_1 == tau_pt_2):
    print("tau pt spectra is the same")
else:
    print("differnet tau pts")

if (btau_pt_1 == btau_pt_2):
    print("btaus pt spectra is the same")
else:
    print("different btau pts")










