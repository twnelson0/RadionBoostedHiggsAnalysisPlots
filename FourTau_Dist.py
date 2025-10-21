import awkward as ak
import uproot
import hist
from hist import intervals
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from coffea import processor, nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import candidate, vector
from math import pi
import numba 
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import vector
import os
import time
from distributed import Client
from dask_jobqueue import HTCondorCluster
from cutflow_producer import cutflow_producer
#import glob


#X509 function (for HTC)
def move_X509():
    try:
        _x509_localpath = (
            [
                line
                for line in os.popen("voms-proxy-info").read().split("\n")
                if line.startswith("path")
            ][0]
            .split(":")[-1]
            .strip()
        )
    except Exception as err:
        raise RuntimeError(
            "x509 proxy could not be parsed, try creating it with 'voms-proxy-init'"
        ) from err
    _x509_path = f'/scratch/{os.environ["USER"]}/{_x509_localpath.split("/")[-1]}'
    os.system(f"cp {_x509_localpath} {_x509_path}")
    return os.path.basename(_x509_localpath)


hep.style.use(hep.style.CMS)
TABLEAU_COLORS = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

#Global Variables
WScaleFactor = 1.21
DYScaleFactor = 1.23
TT_FullLep_BR = 0.1061
TT_SemiLep_BR = 0.4392
TT_Had_BR = 0.4544

#Functions and variables for Luminosity weights
lumi_table_data = {"MC Sample":[], "Luminosity":[], "Cross Section (pb)":[], "Number of Events":[], "Calculated Weight":[]}

#Function to get final state of 4 tau event
def fin_state(n_ele, n_mu):
	state = ""
	for i in range(int(n_ele)):
		state += "e"
	for i in range(int(n_mu)):
		state += "$\\mu$"
	for i in range(4 - int(n_ele) - int(n_mu)):
		state += "$\\tau_h$"
	return state

#Function that gets theoretical branching fraction
def branch_ratio(n_had, n_mu, n_ele):
	return (2/3)**n_had * (1/6)**(n_ele)* (1/6)**(n_mu)

fin_state_vec = np.vectorize(fin_state)
branch_ratio_vec = np.vectorize(branch_ratio)

#Count the number of Z-Bosons from a set of leptons with a total number of lepton pairs given by num_pairs within a certain range of the Z-peak
def Z_Count(leptons, num_pairs, Z_lower = 80, Z_upper = 100):
	if num_pairs == 0: #There exist no possible pairs (either no leptons in event or no OS pairs) (this is must be known apriori)
		return ak.singletons(ak.from_numpy(np.ravel(np.zeros((1,len(leptons))))))
	else:
		#Obtain all pair masses
		for n in range(num_pairs):
			if (n == 0):
				pair_mass = ak.singletons(di_mass(leptons[:,2*(n)], leptons[:,2*n + 1]))
				print(pair_mass)
			else:
				pair_mass = ak.concatenate((pair_mass,ak.singletons(di_mass(leptons[:,2*(n)], leptons[:,2*n + 1]))),axis=1)
		
		#See if masses within width of Z peak
		lower_cond = pair_mass > Z_lower
		upper_cond = pair_mass < Z_upper
		Z_cond = np.bitwise_and(lower_cond,upper_cond)
		Z_masses = pair_mass[Z_cond]
		
		return ak.num(Z_masses,axis=1)  	
		

def delta_phi(vec1,vec2):
	return (vec1.phi - vec2.phi + pi) % (2*pi) - pi	

def MET_delta_phi(part1,MET_obj):
	return (part1.phi - MET_obj.pfMETPhi + pi) % (2*pi) - pi

def deltaR(part1, part2):
	return np.sqrt((part2.eta - part1.eta)**2 + (delta_phi(part1,part2))**2)

def totalCharge(part1,part2):
	return part1.charge + part2.charge

def single_mass(part1):
	return np.sqrt((part1.E)**2 - (part1.Px)**2 - (part1.Py)**2 - (part1.Pz)**2)

def di_mass(part1,part2):
	return np.sqrt((part1.E + part2.E)**2 - (part1.Px + part2.Px)**2 - (part1.Py + part2.Py)**2 - (part1.Pz + part2.Pz)**2)

def four_mass(part_arr): #Four Particle mass assuming each event has 4 particles
	return np.sqrt((part_arr[0].E + part_arr[1].E + part_arr[2].E + part_arr[3].E)**2 - (part_arr[0].Px + part_arr[1].Px + part_arr[2].Px + part_arr[3].Px)**2 - 
		(part_arr[0].Py + part_arr[1].Py + part_arr[2].Py + part_arr[3].Py)**2 - 
		(part_arr[0].Pz + part_arr[1].Pz + part_arr[2].Pz + part_arr[3].Pz)**2)

def bit_mask(in_bits):
	mask = 0
	for bit in in_bits:
		mask += (1 << bit)
	return mask

def bit_or(data):
	cond_1 = np.bitwise_and(data.trigger,bit_mask([39,40])) == bit_mask([39,40])
	cond_2 = np.bitwise_and(data.trigger,bit_mask([39,40])) == bit_mask([39])
	cond_3 = np.bitwise_and(data.trigger,bit_mask([39,40])) == bit_mask([40])
	return np.bitwise_or(cond_1, np.bitwise_or(cond_2,cond_3))

#Dictionary of cross sections 
xSection_Dictionary = {"Signal": 0.01, #Chosen to make plots readable
						#TTBar Background
						"TTTo2L2Nu": 831.76*TT_FullLep_BR, "TTToSemiLeptonic": 831.76*TT_SemiLep_BR, "TTToHadronic": 831.76*TT_Had_BR,
						#DiBoson Background
						"ZZ2l2q": 3.22, "WZ3l1nu": 4.708, "WZ2l2q": 5.595, "WZ1l1nu2q": 10.71, "VV2l2nu": 11.95, "WZ1l3nu": 3.05, #"WZ3l1nu.root" : 27.57,
						#ZZ->4l
						"ZZ4l": 1.212,
						#DiBoson continued
						#"ZZTo2L2Nu_powheg": 0.564, "ZZTo2L2Q_amcNLO": 3.22, "ZZTo4L_powheg": 1.212, "WWTo2L2Nu_powheg": 12.178, "WWTo4Q_powheg": 51.723, "WWTo1LNuQQ_powheg": 49.997, 
						#"WZTo1L3Nu_amcatnloFXFX": 3.033, "WZTo2L2Q_amcNLO": 5.595, "WZTo3LNu_amcNLO": 4.42965, "WZTo1L1Nu2Q_amcNLO": 10.71, "WW1l1nu2q": 49.997, "WZ1l3nu": 3.05,
						#Single Top Background
						"Tbar-tchan": 26.23, "T-tchan": 44.07, "Tbar-tW": 35.6, "T-tW": 35.6, 
						#Drell-Yan Jets
						"DYJetsToLL_Pt-50To100": 387.130778, "DYJetsToLL_Pt-100To250": 89.395097,"DYJetsToLL_Pt-250To400": 3.435181, "DYJetsToLL_Pt-400To650": 0.464024, "DYJetsToLL_Pt-650ToInf": 0.043602,
						#WJets
						"WJetsToLNu_HT-100To200" : 1345*WScaleFactor, "WJetsToLNu_HT-200To400": 359.7*WScaleFactor, "WJetsToLNu_HT-400To600": 48.91*WScaleFactor, "WJetsToLNu_HT-600To800": 12.05*WScaleFactor, "WJetsToLNu_HT-800To1200": 5.501*WScaleFactor, "WJetsToLNu_HT-1200To2500" : 1.329*WScaleFactor, "WJetsToLNu_HT-2500ToInf" : 0.03216*WScaleFactor, 
						#SM Higgs
						"ZH125": 0.7544*0.0621, "ggZHLL125":0.1223 * 0.062 * 3 * 0.033658, "ggZHNuNu125": 0.1223*0.062*0.2,"ggZHQQ125": 0.1223*0.062*0.6991, "toptopH125": 0.5033*0.062, #"ggH125": 48.30* 0.0621, "qqH125": 3.770 * 0.0621, "WPlusH125": 
						#QCD
						"QCD_HT300to500": 347700, "QCD_HT500to700": 32100, "QCD_HT700to1000": 6831, "QCD_HT1000to1500": 1207, "QCD_HT1500to2000": 119.9, "QCD_HT2000toInf": 25.24,
						}	
Lumi_2018 = 59830

#Dictionary of number of events (values specified in main loop)
numEvents_Dict = {}
working_dir = os.getcwd()

def weight_calc(sample,numEvents=1):
	return Lumi_2018*xSection_Dictionary[sample]/numEvents

def pairing_list(maxN):
	out_array = []
	for i in range(maxN):
		out_array.append("pair_%d"%(i + 1))
	return out_array
	
#Use numba to speed up the loop
@numba.njit
def find_Z_Candidates(event_leptons, builder):
	"""
	For a given collection of leptons (with at least 4 leptons) construct and count Z bosons
	"""
	for lepton in event_leptons:
		paired_set = {-1} #Set of already paired indicies
		ZMult = 0
		builder.begin_list()
		for i in range(len(lepton)): #Loop over all leptons
			if i in paired_set: #Avoid double pairings
				continue
			for j in range(i +1, len(lepton)):
				if j in paired_set: #Avoid double pairings
					continue
				if (lepton[i].charge + lepton[j].charge != 0): #Impose 0 eletric charge
					continue
				candidate_mass = np.sqrt((lepton[i].E + lepton[j].E)**2 - (lepton[i].Px + lepton[j].Px)**2 - (lepton[i].Py + lepton[j].Py)**2 - (lepton[i].Pz + lepton[j].Pz)**2)
				if (candidate_mass > 80 and candidate_mass < 100): #Valid Z Boson
					ZMult += 1
					paired_set.add(j) #Add current index to paired set
					break
		
		#Add Z_multiplicity for event to output
		builder.integer(ZMult)
		builder.end_list()
	return builder
	

class FourTauPlotting(processor.ProcessorABC):
	def __init__(self, trigger_bit, trigger_cut = True, offline_cut = False, or_trigger = False, PUWeights = None, PU_weight_bool = False, signal_mass = ""):
		self.trigger_bit = trigger_bit
		self.offline_cut = offline_cut
		self.trigger_cut = trigger_cut
		self.OrTrigger = or_trigger
		self.isData = False #Default assumption is running on MC
		self.PU_bool = PU_weight_bool
		self.PUWeights = PUWeights
		self.massVal = signal_mass
		#pass

	def process(self, events):
		vector.register_awkward()
		#Begin by checking if running on data or sample
		dataset = events.metadata['dataset']
		if ("Data_" in dataset): #Check to see if running on data
			self.isData = True	
		
		event_level = ak.zip(
			{
				#"jet_trigger": events.HLTJet,
				#"mu_trigger": events.HLTEleMuX,
				"METHTMHT_Trigger": events.HLT_PFHT500_PFMET100_PFMHT100_IDTight,
				"Mu_Trigger": events.HLT_Mu50,
				"pfMET": events.MET_pt,
				"pfMETPhi": events.MET_phi,
				"event_weight": ak.ones_like(events.MET_pt), #*0.9,
				"n_electrons": ak.zeros_like(events.MET_pt),
				"n_muons": ak.zeros_like(events.MET_pt),
				"n_tau_electrons": ak.zeros_like(events.MET_pt),
				"n_tau_muons": ak.zeros_like(events.MET_pt),
				"n_tau_hadronic": ak.zeros_like(events.MET_pt),
				"event_num": events.event,
				"run": events.run,
				"Lumi" : events.luminosityBlock,
				#"genWeight": events.genWeight
			},
			with_name="EventArray",
			behavior=candidate.behavior,
		)
		tau = ak.zip( 
			{
				"pt": events.boostedTau_pt,
				"Px": events.boostedTau_pt*np.cos(events.boostedTau_phi),
				"Py": events.boostedTau_pt*np.sin(events.boostedTau_phi),
				"Pz": (events.boostedTau_pt/np.sin(2*np.arctan(np.exp(-events.boostedTau_eta))))*np.cos(2*np.arctan(np.exp(-events.boostedTau_eta))),
				"E": np.sqrt((events.boostedTau_pt/np.sin(2*np.arctan(np.exp(-events.boostedTau_eta))))**2 + events.boostedTau_mass**2),
				"mass": events.boostedTau_mass,
				"eta": events.boostedTau_eta,
				"phi": events.boostedTau_phi,
				"nBoostedTau": events.nboostedTau,
				"charge": events.boostedTau_charge,
				#"iso": events.boostedTau_rawDeepTau2018v2p7VSjet,
				"iso": events.boostedTau_idDeepTau2018v2p7VSjet,
				"DBT": events.boostedTau_rawDeepTau2018v2p7VSjet,
				#"decay": events.boostedTaupfTausDiscriminationByDecayModeFinding,
				"decay": events.boostedTau_idDecayModeOldDMs,
			},
			with_name="TauArray",
			behavior=candidate.behavior,
		)
		electron = ak.zip(
			{
				"pt": events.Electron_pt,
				"eta": events.Electron_eta,
				"phi": events.Electron_phi,
				"charge": events.Electron_charge,
				"Px": events.Electron_pt*np.cos(events.Electron_phi),
				"Py": events.Electron_pt*np.sin(events.Electron_phi),
				"Pz": events.Electron_pt*np.tan(2*np.arctan(np.exp(-events.Electron_eta)))**-1,
				"E": np.sqrt(events.Electron_pt**2 + (events.Electron_pt/np.tan(2*np.arctan(np.exp(-events.Electron_eta))))**2 + events.Electron_mass**2),
				#"SCEta": events.Electron_SCEta,
				"SCEta": events.Electron_deltaEtaSC,
				#"IDMVANoIso": events.Electron_IDMVANoIso,
				"IDMVANoIso": events.Electron_mvaNoIso,
					
			},
			with_name="ElectronArray",
			behavior=candidate.behavior,
			
		)
		muon = ak.zip(
			{
				"pt": events.Muon_pt,
				"eta": events.Muon_eta,
				"phi": events.Muon_phi,
				"charge": events.Muon_charge,
				"Px": events.Muon_pt*np.cos(events.Muon_phi),
				"Py": events.Muon_pt*np.sin(events.Muon_phi),
				"Pz": events.Muon_pt*np.tan(2*np.arctan(np.exp(-events.Muon_eta)))**-1,
				"E": np.sqrt(events.Muon_pt**2 + (events.Muon_pt/np.tan(2*np.arctan(np.exp(-events.Muon_eta))))**2 + events.Muon_mass**2),
				"nMu": events.nMuon,
				#"IDbit": events.muIDbit, #No idea what the nanoAOD analog is for this 
				#"IDbit": events.Muon_IDbit,
                "IDSelec": events.Muon_mediumId,
				"D0": events.Muon_dxy,
				"Dz": events.Muon_dz
					
			},
			with_name="MuonArray",
			behavior=candidate.behavior,
			
		)

		AK8Jet = ak.zip(
			{
				"AK8JetDropMass": events.FatJet_msoftdrop,
				"AK8JetPt": events.FatJet_pt,
				"eta": events.FatJet_eta,
				"phi": events.FatJet_phi,
			},
			with_name="AK8JetArray",
			behavior=candidate.behavior,
		)
		
		Jet = ak.zip(
			{
				"Pt": events.Jet_pt,
				#"PFLooseId": events.JetPFLooseId,
				"PFLooseId": events.Jet_jetId, #Not sure that this is correct
				"eta": events.Jet_eta,
				"phi": events.Jet_phi,
				#"DeepCSVTags_b": events.Jet_DeepCSVTags_b
				"DeepCSVTags_b": events.Jet_btagCSVV2,
			},
			with_name="PFJetArray",
			behavior=candidate.behavior,
		)

		if not(self.isData): #Check to see if this is (or is not) an MC simulation
			Gen_Info = ak.zip({
					"MCId": events.GenPart_pdgId,
					"MotherId": events.GenPart_genPartIdxMother,
					#"GMotherId": events.mcGMomPID,
					"Pt": events.GenPart_pt,
					"Eta": events.GenPart_eta,
					"Phi": events.GenPart_phi,
					"Px": events.GenPart_pt*np.cos(events.GenPart_phi),
					"Py": events.GenPart_pt*np.sin(events.GenPart_phi),
					"Pz": events.GenPart_pt*np.tan(2*np.arctan(np.exp(-events.GenPart_eta)))**-1,
					"E": np.sqrt(events.GenPart_pt**2 + (events.GenPart_pt*np.tan(2*np.arctan(np.exp(-events.GenPart_eta)))**-1)**2 + events.GenPart_mass**2),

				},
				with_name = "GEN_Array",
				behavior=candidate.behavior,
			)


			PU_Info = ak.zip({
				#"puTrue": events.puTrue[:,0] #I have no idea where this info is contained in the nanoAOD
                #"puTrue": events.Pileup_nPU #Possiblity 1
                "puTrue": events.Pileup_nTrueInt #Possiblity 2
                #"puTrue": events.Pileup_sumLOOT #Possiblity 3
				},
				with_name = "PU_Array",
				behavior=candidate.behavior,
			)
            
			#GenTau_Num = ak.num(np.bitwise_and(np.abs(Gen_Info.MCId) == 15, np.abs(Gen_Info.MotherId) != 15),axis=1) #Get the number of Generated taus that decayed from something
			GenTau_Num = ak.num(Gen_Info[np.abs(Gen_Info.MCId) == 15],axis=1)
			#event_level["event_weight"] = event_level.event_weight**GenTau_Num #GenTau_Num #Apply weightings based on Gen Taus
			event_level["event_weight"] = np.multiply(events.genWeight, (event_level.event_weight*0.9)**GenTau_Num) #GenTau_Num #Apply weightings based on Gen Taus
			#event_level["event_weight"] = events.genWeight #GenTau_Num #Apply weightings based on Gen Taus
			dummy_weight = event_level["event_weight"]
			#Debugg/test Gen Tau Weighting
			#for i in range(len(GenTau_Num)):
			#	if (GenTau_Num[i] != len(tau[i].pt)):
			#		print("# of Gen Tau # of Reco Tau Mismatch")
			#		print("# of Gen Taus:%d"%GenTau_Num[i])
			#		print("# of Reco Taus:%d"%len(tau[i].pt))

			    

				#if (GenTau_Num[i] != len(tau[i].pt)):
				#	print("Gen Tau Reco Tau MisMatch")
				#	print("# of Gen Taus: %d"%GenTau_Num[i])
				#	print("# of Reco Taus: %d"%len(tau[i].pt))

			    #if (GenTau_Num[i] > len(tau[i].pt)):
				#	print("!!More Gen Taus than Reco Taus!!")
			    #if not(np.isclose([0.9**GenTau_Num[i]],[event_level.event_weight[i]])):
				#	print("!!Event gen tau weighting mismatch!!")
				#	print("Event weight: %f"%event_level.event_weight[i])
				#	print("Expected weight: %f"%0.9**GenTau_Num[i])
			#event_level["event_weight"] = event_level.event_weight**ak.zeros_like(GenTau_Num) #Force all weights to be 1
			#event_level["event_weight"] = event_level.event_weight * events.genWeight
			#Debugg/test genWeights
			#genWeight = events.genWeight
			#for i in range(len(event_level.event_weight)):
			#	if (dummy_weight[i]*genWeight[i] != event_level.event_weight[i]):
			#		print("!!Event gen weighting mistmatch!!")
			#		print("Event weight: %f"%event_level.event_weight[i])
			#		print("Expected weight: %f"%dummy_weight[i]*genWeight)
			    	#print("!!Event gen weighting mistmatch!!")
				#print("Event weight: %f"%event_leve.event_weight[i])
				#print("Expected weight: %f"%dummy_weight[i]*events.genWeight[i])
			#print(event_level.event_weight)
			dummy_weight = event_level["event_weight"]
			print("Applied gen weights")

			if (self.PU_bool): #Apply PU reweighting scheme
				PU_Arr = np.array(np.rint(ak.flatten(PU_Info.puTrue,axis=-1)),dtype = np.int8)
				PU_Corr = self.PUWeights[PU_Arr] #This may be causing problmes, though I'm not sure, do I need the golden JSON thing to correctly do PU reweighting??? 
				event_level["event_weight"] = np.multiply(event_level.event_weight,PU_Corr) #Is this line screwing things up??
				#Debugg/gest PU reweighting
				#for i in range(len(event_level.event_weight)):
				#	if (dummy_weight[i]*PU_Corr[i] != event_level.event_weight[i]):
				#	    print("!!Event gen weighting mistmatch!!")
				#	    print("Event weight: %f"%event_level.event_weight[i])
				#	    print("Expected weight: %f"%dummy_weight[i]*PU_Corr[i])
				#print(event_level.event_weight)



		
		#tau = tau[ak.argsort(tau.pt,axis=1)] #Force tau pT Ordering
		print("!!!=====Dataset=====!!!!")	
		print(type(dataset))
		print(dataset)


		print("Number of events before selection + Trigger: %d"%ak.num(tau,axis=0))

		#Look at the problem events before anything is applied
		#Construct HT and MHT variables (and give them their own object)
		Jet_MHT = Jet[Jet.Pt > 30]
		Jet_MHT = Jet_MHT[np.abs(Jet_MHT.eta) < 5]
		Jet_MHT = Jet_MHT[Jet_MHT.PFLooseId > 0.5]
		event_level["MHT_x"] = ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False)
		#event_level["MHT_y"] = ak.sum(Jet_MHT.Pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False) #Broken/wrong implementation
		event_level["MHT_y"] = ak.sum(Jet.Pt*np.sin(Jet.phi),axis=1,keepdims=False) #Fixed implementation (I think)
		#Jet_MHT["MHT"] = np.sqrt(Jet_MHT.MHT_x**2 + Jet_MHT.MHT_y**2)
		event_level["MHT"] = np.sqrt(event_level.MHT_x**2 + event_level.MHT_y**2) 
		
		#HT Seleciton (new)
		#tau_temp1,HT_Jet_Cand = ak.unzip(ak.cartesian([tau,Jet_MHT], axis = 1, nested = True))
		Jet_HT = Jet[Jet.Pt > 30]
		Jet_HT = Jet_HT[np.abs(Jet_HT.eta) < 3]
		Jet_HT = Jet_HT[Jet_HT.PFLooseId > 0.5]
		event_level["HT"] = ak.sum(Jet_HT.Pt, axis = 1, keepdims=False)
		
		#Apply trigger weights
		#if not(self.isData):
		#	sum_metmht = event_level["HT"] + event_level["MHT"]
		#	MET = event_level["MET"]
			#Get 2d histograms
		#	with uproot.open("/hdfs/store/user/abdollah/TrgEFF/sf_met_trgEff_2D_2018.root") as f1:
		#		f1["TrgEfficiency2D"]
		#		MET = ak.where(MET > 2000,2000,MET)
		#		sum_metmht = ak.where(sum_metmht > 1500,1500,sum_metmht)

		
		#Triggering logic (This whole thing needs to be changed for NanoAOD)
		trigger_mask = bit_mask([self.trigger_bit])
		if (not(self.isData)):	#MC trigger logic
			if (self.OrTrigger): # and np.pi == np.exp(1)): #Select for both triggers
				print("Both Triggers")
				#event_level_21 = event_level[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				#print(event_level.Mu_Trigger)
				event_level_21 = event_level[event_level.Mu_Trigger]
				#event_level_fail = event_level[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				event_level_fail = event_level[np.bitwise_not(event_level.Mu_Trigger)]
				event_level_39 = event_level_fail[event_level_fail.METHTMHT_Trigger]
				#event_level_27 = event_level_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([27])) == bit_mask([27])]
				#event_level_39 = event_level_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([39])) == bit_mask([39])]

				#Muon ID selection
				#id_cond = np.bitwise_and(muon.IDbit,2) != 0 #Do not delete
				id_cond = muon.IDSelec
				d0_cond = np.abs(muon.D0) < 0.045
				dz_cond = np.abs(muon.Dz) < 0.2
				good_muon_cond = np.bitwise_and(id_cond, np.bitwise_and(d0_cond, dz_cond))
				muon = muon[good_muon_cond]
			
				#Single Muon Trigger	
				tau_21 = tau[event_level.Mu_Trigger]
				tau_fail = tau[np.bitwise_not(event_level.Mu_Trigger)]
				AK8Jet_21 = AK8Jet[event_level.Mu_Trigger]
				AK8Jet_fail = AK8Jet[np.bitwise_not(event_level.Mu_Trigger)]
				Jet_21 = Jet[event_level.Mu_Trigger]
				Jet_fail = Jet[np.bitwise_not(event_level.Mu_Trigger)]
				muon_21 = muon[event_level.Mu_Trigger]
				muon_fail = muon[np.bitwise_not(event_level.Mu_Trigger)]
				electron_21 = electron[event_level.Mu_Trigger]
				electron_fail = electron[np.bitwise_not(event_level.Mu_Trigger)]
				if (not(self.isData)): # and self.isData):
					Gen_Info_21 = Gen_Info[event_level.Mu_Trigger]
					Gen_Info_fail = Gen_Info[np.bitwise_not(event_level.Mu_Trigger)]
					
				

				#Apply offline Single Muon Cut
				tau_21 = tau_21[ak.any(muon_21.nMu > 0, axis = 1)]
				AK8Jet_21 = AK8Jet_21[ak.any(muon_21.nMu > 0, axis = 1)]
				Jet_21 = Jet_21[ak.any(muon_21.nMu > 0, axis = 1)]
				electron_21 = electron_21[ak.any(muon_21.nMu > 0, axis = 1)]
				if (not(self.isData)): # and self.isData):
					Gen_Info_21 = Gen_Info_21[ak.any(muon_21.nMu > 0, axis = 1)]
				muon_21 = muon_21[ak.any(muon_21.nMu > 0, axis = 1)]


				
				#pT
				tau_21 = tau_21[ak.any(muon_21.pt > 52, axis = 1)]
				AK8Jet_21 = AK8Jet_21[ak.any(muon_21.pt > 52, axis = 1)]
				Jet_21 = Jet_21[ak.any(muon_21.pt > 52, axis = 1)]
				event_level_21 = event_level_21[ak.any(muon_21.pt > 52, axis = 1)]
				electron_21 = electron_21[ak.any(muon_21.pt > 52, axis = 1)]
				if (not(self.isData)): # and self.isData):
					Gen_Info_21 = Gen_Info_21[ak.any(muon_21.pt > 52, axis = 1)]
				muon_21 = muon_21[ak.any(muon_21.pt > 52, axis = 1)]

				
				#Apply JetHT_MHT_MET Trigger
				tau_39 = tau_fail[event_level_fail.METHTMHT_Trigger]
				AK8Jet_39 = AK8Jet_fail[event_level_fail.METHTMHT_Trigger]
				Jet_39 = Jet_fail[event_level_fail.METHTMHT_Trigger]
				electron_39 = electron_fail[event_level_fail.METHTMHT_Trigger]
				muon_39 = muon_fail[event_level_fail.METHTMHT_Trigger]
				if (not(self.isData)): # and self.isData):
					Gen_Info_39 = Gen_Info_fail[event_level_fail.METHTMHT_Trigger]
		
				#HT Cut
				tau_39 = tau_39[event_level_39.HT > 550]	
				AK8Jet_39 = AK8Jet_39[event_level_39.HT > 550]	
				Jet_39 = Jet_39[event_level_39.HT > 550]
				muon_39 = muon_39[event_level_39.HT > 550]
				electron_39 = electron_39[event_level_39.HT > 550]
				if (not(self.isData)): # and self.isData):
					Gen_Info_39 = Gen_Info_39[event_level_39.HT > 550]
				event_level_39 = event_level_39[event_level_39.HT > 550]

				#MHT Cut
				tau_39 = tau_39[event_level_39.MHT > 110]	
				AK8Jet_39 = AK8Jet_39[event_level_39.MHT > 110]	
				Jet_39 = Jet_39[event_level_39.MHT > 110]
				muon_39 = muon_39[event_level_39.MHT > 110]
				electron_39 = electron_39[event_level_39.MHT > 110]
				if (not(self.isData)): # and self.isData):
					Gen_Info_39 = Gen_Info_39[event_level_39.MHT > 110]
				event_level_39 = event_level_39[event_level_39.MHT > 110]

				#MET Cut	
				tau_39 = tau_39[event_level_39.pfMET > 110]	
				AK8Jet_39 = AK8Jet_39[event_level_39.pfMET > 110]	
				Jet_39 = Jet_39[event_level_39.pfMET > 110]
				muon_39 = muon_39[event_level_39.pfMET > 110]
				electron_39 = electron_39[event_level_39.pfMET > 110]
				if (not(self.isData)): # and self.isData):
					Gen_Info_39 = Gen_Info_39[event_level_39.pfMET > 110]
				event_level_39 = event_level_39[event_level_39.pfMET > 110]
			
				#Apply JetMHT_MET cut	
				#tau_27 = tau_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([27])) == bit_mask([27])]
				#AK8Jet_27 = AK8Jet_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([27])) == bit_mask([27])]
				#Jet_27 = Jet_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([27])) == bit_mask([27])]
				#muon_27 = muon_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([27])) == bit_mask([27])]
		
				#Apply offline JetHT Cut
				#MET
				#tau_27 = tau_27[event_level_27.pfMET > 130]	
				#AK8Jet_27 = AK8Jet_27[event_level_27.pfMET > 130]	
				#Jet_27 = Jet_27[event_level_27.pfMET > 130]
				#event_level_27 = event_level_27[event_level_27.pfMET > 130]
			
				#MHT
				#tau_27 = tau_27[event_level_27.MHT > 130]	
				#AK8Jet_27 = AK8Jet_27[event_level_27.MHT > 130]	
				#Jet_27 = Jet_27[event_level_27.MHT > 130]
				#event_level_27 = event_level_27[event_level_27.MHT > 130]
				
				#PFLoose ID
				#tau_27 = tau_27[ak.any(Jet_27.PFLooseId, axis=1)]	
				#AK8Jet_27 = AK8Jet_27[ak.any(Jet_27.PFLooseId, axis=1)]	
				#Jet_27 = Jet_27[ak.any(Jet_27.PFLooseId, axis=1)]

				#Recombine 
				#tau = ak.concatenate((tau_21,tau_27))
				#AK8Jet = ak.concatenate((AK8Jet_21, AK8Jet_27))
				#Jet = ak.concatenate((Jet_21,Jet_27))
				#muon = ak.concatenate((muon_21,muon_27))
				tau = ak.concatenate((tau_21,tau_39))
				AK8Jet = ak.concatenate((AK8Jet_21, AK8Jet_39))
				Jet = ak.concatenate((Jet_21,Jet_39))
				muon = ak.concatenate((muon_21,muon_39))
				electron = ak.concatenate((electron_21,electron_39))
				event_level = ak.concatenate((event_level_21, event_level_39))
				if (not(self.isData)): # and self.isData):
					Gen_Info = ak.concatenate((Gen_Info_21,Gen_Info_39))
				
			else: #Single Trigger
				#print("Single Trigger (in theory)")
				if (self.trigger_bit != None and self.OrTrigger == False):
					if (self.trigger_bit == 21): #Single Mu
						print("Single Trigger: Mu Trigger (21)")
						tau = tau[event_level.Mu_Trigger]
						AK8Jet = AK8Jet[event_level.Mu_Trigger]
						Jet = Jet[event_level.Mu_Trigger]
						muon = muon[event_level.Mu_Trigger]
						event_level = event_level[event_level.Mu_Trigger]
						
						#Muon ID selection
						#id_cond = np.bitwise_and(muon.IDbit,2) != 0 #Do not remove line commented out for debugging purposes
						id_cond = muon.IDSelec
						d0_cond = np.abs(muon.D0) < 0.045
						dz_cond = np.abs(muon.Dz) < 0.2
						good_muon_cond = np.bitwise_and(id_cond, np.bitwise_and(d0_cond, dz_cond))
						muon = muon[good_muon_cond]
				
						#Apply offline Single Muon Cut
						#if (np.exp(1) != np.pi):
						tau = tau[ak.any(muon.nMu > 0, axis = 1)]
						AK8Jet = AK8Jet[ak.any(muon.nMu > 0, axis = 1)]
						Jet = Jet[ak.any(muon.nMu > 0, axis = 1)]
						event_level = event_level[ak.any(muon.nMu > 0, axis = 1)]
						electron = electron[ak.any(muon.nMu > 0, axis = 1)]
						muon = muon[ak.any(muon.nMu > 0, axis = 1)]
								
						#pT
						tau = tau[ak.any(muon.pt > 52, axis = 1)]
						AK8Jet = AK8Jet[ak.any(muon.pt > 52, axis = 1)]
						Jet = Jet[ak.any(muon.pt > 52, axis = 1)]
						event_level = event_level[ak.any(muon.pt > 52, axis = 1)]
						electron = electron[ak.any(muon.pt > 52, axis = 1)]
						muon = muon[ak.any(muon.pt > 52, axis = 1)]
					
					if (self.trigger_bit == 27): #Jet HT
						print("Single Trigger: Jet Trigger (27)")
					#	tau = tau[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
					#	AK8Jet = AK8Jet[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
					#	Jet = Jet[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
					#	muon = muon[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
					#	event_level = event_level[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
						
						#pfMET	
					#	tau = tau[event_level.pfMET > 130]	
					#	AK8Jet = AK8Jet[event_level.pfMET > 130]	
					#	Jet = Jet[event_level.pfMET > 130]
					#	event_level = event_level[event_level.pfMET > 130]
					
						#MHT
					#	tau = tau[event_level.MHT > 130]	
					#	AK8Jet = AK8Jet[event_level.MHT > 130]	
					#	Jet = Jet[event_level.MHT > 130]
					#	event_level = event_level[event_level.MHT > 130]
						
						#PFLoose ID
					#	tau = tau[ak.any(Jet.PFLooseId, axis=1)]	
					#	AK8Jet = AK8Jet[ak.any(Jet.PFLooseId, axis=1)]	
					#	Jet = Jet[ak.any(Jet.PFLooseId, axis=1)]
					
					if (self.trigger_bit == 39): #Jet HT
						print("Single Trigger: Jet Trigger (39)")
						tau = tau[event_level.METHTMHT_Trigger]
						AK8Jet = AK8Jet[event_level.METHTMHT_Trigger]
						Jet = Jet[event_level.METHTMHT_Trigger]
						muon = muon[event_level.METHTMHT_Trigger]
						electron = electron[event_level.METHTMHT_Trigger]
						event_level = event_level[event_level.METHTMHT_Trigger]
						print("Number of events after Online Trigger(dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
				
						#Offline Cuts
						#HT Cut
						tau = tau[event_level.HT > 550]
						AK8Jet = AK8Jet[event_level.HT > 550]
						Jet = Jet[event_level.HT > 550]
						muon = muon[event_level.HT > 550]
						electron = electron[event_level.HT > 550]
						event_level = event_level[event_level.HT > 550]

						#pfMET	
						tau = tau[event_level.pfMET > 110]	
						AK8Jet = AK8Jet[event_level.pfMET > 110]	
						Jet = Jet[event_level.pfMET > 110]
						muon = muon[event_level.pfMET > 110]
						electron = electron[event_level.pfMET > 110]
						event_level = event_level[event_level.pfMET > 110]
			
						#MHT
						tau = tau[event_level.MHT > 110]	
						AK8Jet = AK8Jet[event_level.MHT > 110]	
						Jet = Jet[event_level.MHT > 110]
						muon = muon[event_level.MHT > 110]
						electron = electron[event_level.MHT > 110]
						event_level = event_level[event_level.MHT > 110]

			print("Number of events after selection + Trigger: %d"%ak.num(tau,axis=0))
			print("Number of events after Trigger + Selection (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		else:
			 #Skip the trigger (??)
			if ("SingleMuon" in dataset):  #and np.exp(1) == np.pi): #Single Mu
				#Muon ID selection
				#id_cond = np.bitwise_and(muon.IDbit,2) != 0 #Do not delete
				id_cond = muon.IDSelec
				d0_cond = np.abs(muon.D0) < 0.045
				dz_cond = np.abs(muon.Dz) < 0.2
				good_muon_cond = np.bitwise_and(id_cond, np.bitwise_and(d0_cond, dz_cond))
				muon = muon[good_muon_cond]


				print("Single Muon Trigger")
				tau = tau[event_level.Mu_Trigger]	
				AK8Jet = AK8Jet[event_level.Mu_Trigger]	
				Jet = Jet[event_level.Mu_Trigger]	
				muon = muon[event_level.Mu_Trigger]	
				electron = electron[event_level.Mu_Trigger]	
				event_level = event_level[event_level.Mu_Trigger]

				if (self.OrTrigger): #If working on both triggers drop events that passed JetHT trigger
					tau = tau[np.bitwise_not(event_level.METHTMHT_Trigger)]	
					AK8Jet = AK8Jet[np.bitwise_not(event_level.METHTMHT_Trigger)]	
					Jet = Jet[np.bitwise_not(event_level.METHTMHT_Trigger)]	
					muon = muon[np.bitwise_not(event_level.METHTMHT_Trigger)]	
					electron = electron[np.bitwise_not(event_level.METHTMHT_Trigger)]	
					event_level = event_level[np.bitwise_not(event_level.METHTMHT_Trigger)]
					

				#pT
				#if (np.exp(1) != np.pi):
				tau = tau[ak.any(muon.nMu > 0, axis = 1)]
				AK8Jet = AK8Jet[ak.any(muon.nMu > 0, axis = 1)]
				Jet = Jet[ak.any(muon.nMu > 0, axis = 1)]
				electron = electron[ak.any(muon.nMu > 0, axis = 1)]
				muon = muon[ak.any(muon.nMu > 0, axis = 1)]
					
				tau = tau[ak.any(muon.pt > 52, axis = 1)]
				AK8Jet = AK8Jet[ak.any(muon.pt > 52, axis = 1)]
				Jet = Jet[ak.any(muon.pt > 52, axis = 1)]
				event_level = event_level[ak.any(muon.pt > 52, axis = 1)]
				electron = electron[ak.any(muon.pt > 52, axis = 1)]
				muon = muon[ak.any(muon.pt > 52, axis = 1)]
				
			if ("JetHT" in dataset): # and np.exp(1) == np.pi): #HT 
				print("Jet Trigger")
				tau = tau[event_level.METHTMHT_Trigger]	
				AK8Jet = AK8Jet[event_level.METHTMHT_Trigger]	
				Jet = Jet[event_level.METHTMHT_Trigger]	
				muon = muon[event_level.METHTMHT_Trigger]	
				electron = electron[event_level.METHTMHT_Trigger]	
				event_level = event_level[event_level.METHTMHT_Trigger]
				print("Number of events after Online Trigger(dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))

				#Offline Cuts
				#HT Cut
				tau = tau[event_level.HT > 550]
				AK8Jet = AK8Jet[event_level.HT > 550]
				Jet = Jet[event_level.HT > 550]
				muon = muon[event_level.HT > 550]
				electron = electron[event_level.HT > 550]
				event_level = event_level[event_level.HT > 550]

				#pfMET	
				tau = tau[event_level.pfMET > 110]	
				AK8Jet = AK8Jet[event_level.pfMET > 110]	
				Jet = Jet[event_level.pfMET > 110]
				muon = muon[event_level.pfMET > 110]
				electron = electron[event_level.pfMET > 110]
				event_level = event_level[event_level.pfMET > 110]
			
				#MHT
				tau = tau[event_level.MHT > 110]	
				AK8Jet = AK8Jet[event_level.MHT > 110]	
				Jet = Jet[event_level.MHT > 110]
				muon = muon[event_level.MHT > 110]
				electron = electron[event_level.MHT > 110]
				event_level = event_level[event_level.MHT > 110]
				
				#print("Jet HT Trigger")
				#tau = tau[np.bitwise_and(event_level.jet_trigger,bit_mask([27])) == bit_mask([27])]	
				#AK8Jet = AK8Jet[np.bitwise_and(event_level.jet_trigger,bit_mask([27])) == bit_mask([27])]	
				#Jet = Jet[np.bitwise_and(event_level.jet_trigger,bit_mask([27])) == bit_mask([27])]	
				#muon = muon[np.bitwise_and(event_level.jet_trigger,bit_mask([27])) == bit_mask([27])]	
				#event_level = event_level[np.bitwise_and(event_level.jet_trigger,bit_mask([27])) == bit_mask([27])]

				#Offline Cuts
				#pfMET	
				#tau = tau[event_level.pfMET > 130]	
				#AK8Jet = AK8Jet[event_level.pfMET > 130]	
				#Jet = Jet[event_level.pfMET > 130]
				#event_level = event_level[event_level.pfMET > 130]
			
				#MHT
				#tau = tau[event_level.MHT > 130]	
				#AK8Jet = AK8Jet[event_level.MHT > 130]	
				#Jet = Jet[event_level.MHT > 130]
				#event_level = event_level[event_level.MHT > 130]
				
				#PFLoose ID
				#tau = tau[ak.any(Jet.PFLooseId, axis=1)]	
				#AK8Jet = AK8Jet[ak.any(Jet.PFLooseId, axis=1)]	
				#Jet = Jet[ak.any(Jet.PFLooseId, axis=1)]
			
			print("# of events after Trigger + Selection: %d"%ak.num(tau,axis=0))
			print("# of events after Trigger + Selection (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
				
		#Get the number of electrons + muons after trigger
		#Apply electron and muon selections first
		muon = muon[muon.pt > 20]
		electron = electron[electron.pt > 20]
		
		#cond1 = np.bitwise_and(np.abs(electron.SCEta) <= 0.8, electron.IDMVANoIso > 0.837)
		#cond2 = np.bitwise_and(np.bitwise_and(np.abs(electron.SCEta) > 0.8, np.abs(electron.SCEta) <= 1.5), electron.IDMVANoIso > 0.715)
		#cond3 = np.bitwise_and(np.abs(electron.SCEta) >= 1.5, electron.IDMVANoIso > 0.357)
		#good_electron_cond = np.bitwise_or(cond1,np.bitwise_or(cond2,cond3))
		#electron = electron[good_electron_cond]

		#Good muon selection
		#cond
		
		event_level["n_muons"] = ak.singletons(ak.num(muon.pt,axis=1))
		event_level["n_electrons"] = ak.singletons(ak.num(electron.pt,axis=1))
		
		#Produce distribution of e/mu to nearest tau
		electron_fourVec = ak.zip({"x": electron.Px, "y": electron.Py, "z": electron.Pz,"t": electron.E},with_name = "LorentzVector")
		muon_fourVec = ak.zip({"x": muon.Px, "y": muon.Py, "z": muon.Pz,"t": muon.E},with_name = "LorentzVector")
		#if (not(self.isData) and self.isData): #Use gen Leptons 
		#	electron_fourVec = ak.zip({"x": Gen_Info[np.abs(Gen_Info.MCId) == 11].Px, "y": Gen_Info[np.abs(Gen_Info.MCId) == 11].Py, "z": Gen_Info[np.abs(Gen_Info.MCId) == 11].Pz,"t": Gen_Info[np.abs(Gen_Info.MCId) == 11].E},with_name = "LorentzVector")
		#	muon_fourVec = ak.zip({"x": Gen_Info[np.abs(Gen_Info.MCId) == 13].Px, "y": Gen_Info[np.abs(Gen_Info.MCId) == 13].Py, "z": Gen_Info[np.abs(Gen_Info.MCId) == 13].Pz,"t": Gen_Info[np.abs(Gen_Info.MCId) == 13].E},with_name = "LorentzVector")
		
		tau_fourVec = ak.zip({"x": tau.Px, "y": tau.Py, "z": tau.Pz,"t": tau.E},with_name = "LorentzVector")

		#electrons,taus = ak.unzip(ak.cartesian([electron_fourVec,tau_fourVec],axis=1, nested = True))
		#muons,taus = ak.unzip(ak.cartesian([muon_fourVec,tau_fourVec],axis=1, nested = True))
		#electron_dR = ak.zip({"t": electrons.E, "x": electrons.Px, "y": electrons.Py, "z": electrons.Pz}).deltaR(ak.zip({"t": taus.E, "x": taus.Px, "y": taus.Py,"z": taus.Pz}))
		
		#electrons,taus = ak.unzip(ak.cartesian([electron_fourVec, tau_fourVec],axis=1, nested = True))
		#electron_dR = electrons.deltaR(taus)
		#muons,taus = ak.unzip(ak.cartesian([muon_fourVec, tau_fourVec],axis=1, nested = True))
		#muon_dR = muons.deltaR(taus)

		#electrons,taus = ak.unzip(ak.cartesian([electron,tau],axis=1,nested=True))
		#electron_dR = ak.zip({"t":electrons.E,"x": electrons.Px,"y": electrons.Py,"z": electrons.Pz},with_name = "Momentum4D").deltaR(ak.zip({"t": taus.E,"x": taus.Px, "y": taus.Py,"z":taus.Pz},with_name = "Momentum4D"))
		#muons,taus = ak.unzip(ak.cartesian([muon,tau],axis=1,nested=True))
		
		
		min_tau_ele = electron_fourVec.nearest(tau_fourVec)
		min_tau_mu = muon_fourVec.nearest(tau_fourVec)
		
		ele_dR_collection = electron_fourVec.delta_r(min_tau_ele)
		mu_dR_collection = muon_fourVec.delta_r(min_tau_mu)
		electron["tau_min_dR"] = ele_dR_collection 
		muon["tau_min_dR"] = mu_dR_collection
		
		#Set up cutflow dictionary
		cutflow_dict = dict.fromkeys(["No_Selec","Tau_pT","Tau_eta","decay","deepboosted"])
		cutflow_table = hist.Hist.new.Reg(6,0,6,label="Cut flow",underflow = True, overflow = True).Double()
		cutflow_dict["No_Selec"] = ak.num(tau,axis=0) #Initial number of events
		cutflow_table.fill(0,weight = ak.num(tau,axis=0))
		print("Number of events (no selections): %d"%cutflow_dict["No_Selec"])
		print("Should also be Number of events (no selections): %d"%len(np.zeros(ak.num(tau,axis=0))))
		
		#Apply selections
		tau = tau[tau.pt > 30] #pT selection
		
		#Remove events with fewer than 4 taus
		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		event_level = event_level[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		electron = electron[ak.num(tau) >= 4] 
		muon = muon[ak.num(tau) >= 4] 
		if (not(self.isData)): # and self.isData):
			Gen_Info = Gen_Info[ak.num(tau) >= 4] 
		tau = tau[ak.num(tau) >= 4] #4 tau events
		cutflow_dict["Tau_pT"] = ak.num(tau,axis=0) #Initial number of events after pt selection
		cutflow_table.fill(1, weight = ak.num(tau,axis=0))
	
		#if (self.isData or not(self.isData)):
		if (not(self.isData)):
			print("# of events after pT cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		tau = tau[np.abs(tau.eta) < 2.3] #eta selection
		
		#Remove events with fewer than 4 taus	
		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		event_level = event_level[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		electron = electron[ak.num(tau) >= 4] 
		muon = muon[ak.num(tau) >= 4] 
		if (not(self.isData)): # and self.isData):
			Gen_Info = Gen_Info[ak.num(tau) >= 4] 
		tau = tau[ak.num(tau) >= 4] #4 tau events
		cutflow_dict["Tau_eta"] = ak.num(tau,axis=0) #Number of events after eta selection
		cutflow_table.fill(2,weight=ak.num(tau,axis=0))
		#if (self.isData or not(self.isData)):
		if (not(self.isData)):
			print("# of events after eta cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		
		
		#Isolation and decay selections
		tau = tau[tau.decay >= 0.5]
		
		#Remove events with fewer than 4 taus	
		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		event_level = event_level[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		electron = electron[ak.num(tau) >= 4] 
		muon = muon[ak.num(tau) >= 4] 
		if (not(self.isData)): # and self.isData):
			Gen_Info = Gen_Info[ak.num(tau) >= 4] 
		tau = tau[ak.num(tau) >= 4] #4 tau events
		cutflow_dict["decay"] = ak.num(tau,axis=0) #Number of events after deay mode
		cutflow_table.fill(3,weight = ak.num(tau,axis=0))
		#if (self.isData or not(self.isData)):
		#	print("# of events after decay cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		
		tau = tau[tau.iso >= 0.85] #Make loose to ensure high number of statistics
		#Remove events with fewer than 4 taus	
		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		event_level = event_level[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		electron = electron[ak.num(tau) >= 4] 
		muon = muon[ak.num(tau) >= 4] 
		if (not(self.isData)): # and self.isData):
			Gen_Info = Gen_Info[ak.num(tau) >= 4] 
		tau = tau[ak.num(tau) >= 4] #4 tau events
		cutflow_dict["deepboosted"] = ak.num(tau,axis=0) #Number of events after isolation
		cutflow_table.fill(4,weight= ak.num(tau,axis=0))
		#if (self.isData or not(self.isData)):
		if (not(self.isData)):
			print("# of events after isolation cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))


		#Delta R Cut on taus (identifiy and remove jets incorrectly reconstructed as taus)
		#a,b = ak.unzip(ak.cartesian([tau,tau], axis = 1, nested = True)) #Create all di-tau pairs
		#select_arr = np.bitwise_and(deltaR(a,b) < 0.8, deltaR(a,b) != 0)
		#for i in range(5):
		#	print(tau.pt[i]) 
		#	print(select_arr[i])
		#tau["dRCut"] = select_arr
		#tau = tau[ak.any(tau.dRCut, axis = 2) == True]
		#if (self.isData or not(self.isData)):
		#	print("# of events after delta R cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		
		#!!!DO NOT UN COMMENT THIS OUT!!!
		#AK8Jet = AK8Jet[(ak.sum(tau.charge,axis=1) == 0)] #Apply charge conservation cut to AK8Jets
		#event_level = event_level[(ak.sum(tau.charge,axis=1) == 0)]
		#Jet = Jet[(ak.sum(tau.charge,axis=1) == 0)]
		#electron = electron[(ak.sum(tau.charge,axis=1) == 0)]
		#muon = muon[(ak.sum(tau.charge,axis=1) == 0)]
		#tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		#if (self.isData or not(self.isData)):
		#	print("# of events after lepton number cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))

		#Remove all events with 3 or fewer taus (after selections at once)
		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		event_level = event_level[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		electron = electron[ak.num(tau) >= 4] 
		muon = muon[ak.num(tau) >= 4] 
		tau = tau[ak.num(tau) >= 4] #4 tau events
		if (self.isData or not(self.isData)):
			print("# of events after 4-tau cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
	
		print("tau length = %d\nevent_level length = %d"%(ak.num(tau,axis=0),ak.num(event_level,axis=0)))	
		tau = tau[ak.num(tau) > 0] #Handle empty arrays left over
		
		#Z Mutliplticity of taus
		event_level["ZMult_tau"] = find_Z_Candidates(tau,ak.ArrayBuilder()).snapshot()
		
		
		#electron["tau_min_dR"] = ak.where(ak.num(ele_dR_collection,axis=1) != 0, ele_dR_collection, ak.singletons(np.ones(ak.num(ele_dR_collection,axis=0))))
		#muon["tau_min_dR"] = ak.where(ak.num(mu_dR_collection,axis=1) != 0, mu_dR_collection, ak.singletons(np.ones(ak.num(mu_dR_collection,axis=0))))
		#print(electron[0].tau_min_dR)
		#min_tau_mu = tau_fourVec.nearest(muon_fourVec)

		#Add Topology Cuts (only if there's anything left to cut on)
		if (ak.num(event_level.MHT,axis=0) > 0):
			#Remove any events with all same sign
			charge_Arr = totalCharge(tau[:,0],tau)
			good_events = ak.any(charge_Arr == 0,axis = 1) #Drop events with no oposite signs
			bad_indx_list = []
			indx = 0
			#for x in good_events:
			#	if (not(x)):
			#		bad_indx_list.append(indx)
			#		print("Bad Event")
			#	indx+=1
		
			#for i in bad_indx_list:
			#	print(charge_Arr[i])

			#tau = tau[good_events]
			#Jet = Jet[good_events]
			#AK8Jet = AK8Jet[good_events]
			#event_level = event_level[good_events]
			#muon = muon[good_events]
			#electron = electron[good_events]
			#charge_Arr = charge_Arr[good_events]
			#Find all collections of taus with less than 4 events or more than 4 events
			#n_more_4 = 0
			#n_less_4 = 0
			#for x in ak.num(tau.pt,axis=1):
			#	if (x < 4):
			#		n_less_4 += 1
			#	if (x > 4):
			#		n_more_4 += 1
			#print("Number of events with 5 or more taus: %d"%n_more_4)
			#print("Number of events with less than 4 taus: %d"%n_less_4)

	
			#Obtain leading pair
			tau_4vec = ak.zip({"t": tau.E, "x": tau.Px, "y": tau.Py, "z": tau.Pz},with_name="Momentum4D")
			tau_lead,tau_other = ak.unzip(ak.cartesian([tau_4vec[:,0],tau_4vec], axis = 1, nested = False))
			deltaR_Arr = ak.values_astype(tau_lead, np.float64).deltaR(ak.values_astype(tau_4vec, np.float64))

			#Remove leading tau from consideration
			leadingTau_Pair = tau[deltaR_Arr != 0]
			charge_Arr = charge_Arr[deltaR_Arr != 0]
			deltaR_Arr = deltaR_Arr[deltaR_Arr != 0]

			#Select tau that minimizes delta R	
			leadingTau_Pair = leadingTau_Pair[deltaR_Arr == ak.min(deltaR_Arr,axis=1)] #Paired tau is selected as the one that minimized deltaR with leading tau
			charge_Arr = charge_Arr[deltaR_Arr == ak.min(deltaR_Arr,axis=1)]
			deltaR_Arr = ak.min(deltaR_Arr,axis=1)
			pair1_charge = charge_Arr
	
			#Remove any empty/invalid pairings
			tau = tau[ak.num(leadingTau_Pair) > 0]
			tau_4vec = tau_4vec[ak.num(leadingTau_Pair) > 0]
			Jet = Jet[ak.num(leadingTau_Pair) > 0]
			AK8Jet = AK8Jet[ak.num(leadingTau_Pair) > 0]
			muon = muon[ak.num(leadingTau_Pair) > 0]
			electron = electron[ak.num(leadingTau_Pair) > 0]
			event_level = event_level[ak.num(leadingTau_Pair) > 0]
			leadingTau_Pair = leadingTau_Pair[ak.num(leadingTau_Pair) > 0]

			#Find second pair
			tau_lead = ak.singletons(tau[:,0])

			tau_lead_4vec = ak.firsts(ak.zip({"t": tau_lead.E, "x": tau_lead.Px, "y": tau_lead.Py, "z": tau_lead.Pz},with_name="Momentum4D"))
			tau_leadPair_4vec = ak.firsts(ak.zip({"t": leadingTau_Pair.E, "x": leadingTau_Pair.Px, "y": leadingTau_Pair.Py, "z": leadingTau_Pair.Pz},with_name="Momentum4D"))

			leadingPairpT = leadingTau_Pair.pt
			tau_rem = tau[ak.values_astype(tau_lead_4vec, np.float64).deltaR(ak.values_astype(tau_4vec, np.float64)) != 0] #Select leading tau by deltaR
			

			tau_rem_4vec = ak.zip({"t": tau_rem.E, "x": tau_rem.Px, "y": tau_rem.Py, "z": tau_rem.Pz},with_name="Momentum4D")
			tau_rem = tau_rem[ak.values_astype(tau_leadPair_4vec,np.float64).deltaR(ak.values_astype(tau_rem_4vec,np.float64)) != 0]
		
			#Drop events with no oposite signs
			charge_Arr = totalCharge(tau_rem[:,0],tau_rem)
		
			tau_rem_4vec = ak.zip({"t": tau_rem.E, "x": tau_rem.Px, "y": tau_rem.Py, "z": tau_rem.Pz},with_name="Momentum4D")
			tau_nextlead,tau_rem_other = ak.unzip(ak.cartesian([tau_rem_4vec[:,0],tau_rem_4vec], axis = 1, nested = False))
			deltaR_Arr = ak.values_astype(tau_nextlead,np.float64).deltaR(ak.values_astype(tau_rem_4vec,np.float64))

			#Remove leading tau
			leadingTau_NextPair = tau_rem[deltaR_Arr != 0]
			deltaR_temp = deltaR_Arr[deltaR_Arr != 0]
			charge_Arr = charge_Arr[deltaR_Arr != 0]
			
			#Select pair that minimizes delta R	
			leadingTau_NextPair = leadingTau_NextPair[deltaR_temp == ak.min(deltaR_temp,axis=1)]
			charge_Arr = charge_Arr[deltaR_temp == ak.min(deltaR_temp,axis=1)]
			pair2_charge = charge_Arr
			deltaR_temp = deltaR_temp[deltaR_temp == ak.min(deltaR_temp,axis=1)]
			
			#Remove any empty/invalid pairings (this may be vegistal at best but I'm not sure)
			tau_lead = tau_lead[ak.num(leadingTau_NextPair) != 0]
			#print("Here's what the selection array looks like:")
			#print(ak.num(leadingTau_NextPair) != 0)
			leadingTau_Pair = leadingTau_Pair[ak.num(leadingTau_NextPair) != 0]
			tau_rem = tau_rem[ak.num(leadingTau_NextPair) != 0]
			tau = tau[ak.num(leadingTau_NextPair) != 0]
			leadingTau_NextPair = leadingTau_NextPair[ak.num(leadingTau_NextPair) != 0]
			
			#Obtain next/remaining leading tau
			#tau_nextlead = tau_rem[tau_rem.pt == tau_rem[:,0].pt]
			tau_nextlead = ak.singletons(tau_rem[:,0])
			

			#Reconstruct tau object in order of pairings 
			tau = ak.concatenate((tau_lead,leadingTau_Pair),axis=1)
			tau = ak.concatenate((tau,tau_nextlead),axis=1)
			tau = ak.concatenate((tau,leadingTau_NextPair),axis=1)

			#Determine if taus match to lepton (e or mu) and make sutiable replacements
			#print("Number of 4tau events: %d"%ak.num(tau.pt,axis=0))
			##print(ak.num(tau.pt,axis=1))
			#non_4_events = 0
			#for x in ak.num(tau.pt,axis=1):
			#	if (x != 4):
			#		print(x)
			#		non_4_events +=1
			#print("There are %d events that don't have 4 taus"%non_4_events)
			
			lead_tau = ak.zip({"t": tau[:,0].E,"x": tau[:,0].Px, "y": tau[:,0].Py,"z" : tau[:,0].Pz},with_name = "Momentum4D")
			leadingpair_tau = ak.zip({"t": tau[:,1].E,"x": tau[:,1].Px, "y": tau[:,1].Py,"z" : tau[:,1].Pz},with_name = "Momentum4D")
			sublead_tau = ak.zip({"t": tau[:,2].E,"x": tau[:,2].Px, "y": tau[:,2].Py,"z" : tau[:,2].Pz},with_name = "Momentum4D")
			subleadingpair_tau = ak.zip({"t": tau[:,3].E,"x": tau[:,3].Px, "y": tau[:,3].Py,"z" : tau[:,3].Pz},with_name = "Momentum4D")
			tau_fourVec_Arr = ak.Array([lead_tau,leadingpair_tau,sublead_tau,subleadingpair_tau])

			if (self.isData or not(self.isData)):
				electron_fourVec = ak.zip({"t": electron.E, "x": electron.Px, "y": electron.Py, "z": electron.Pz},with_name = "Momentum4D")
				muon_fourVec = ak.zip({"t": muon.E, "x": muon.Px, "y": muon.Py, "z": muon.Pz},with_name = "Momentum4D")
			#if not(self.isData):
			#	electron_fourVec = ak.zip({"x": Gen_Info[np.abs(Gen_Info.MCId) == 11].Px, "y": Gen_Info[np.abs(Gen_Info.MCId) == 11].Py, "z": Gen_Info[np.abs(Gen_Info.MCId) == 11].Pz,"t": Gen_Info[np.abs(Gen_Info.MCId) == 11].E},with_name = "Momentum4D")
			#	muon_fourVec = ak.zip({"x": Gen_Info[np.abs(Gen_Info.MCId) == 13].Px, "y": Gen_Info[np.abs(Gen_Info.MCId) == 13].Py, "z": Gen_Info[np.abs(Gen_Info.MCId) == 13].Pz,"t": Gen_Info[np.abs(Gen_Info.MCId) == 13].E},with_name = "Momentum4D")

			Hadronic_Dict = {0: "LeadingTau_h", 1: "PairedLeadingTau_h", 2: "NextLeadingTau_h", 3: "PairedNextLeadingTau_h"}
			electron_Dict = {0: "LeadingTau_ele", 1: "PairedLeadingTau_ele", 2: "NextLeadingTau_ele", 3: "PairedNextLeadingTau_ele"}
			muon_Dict = {0: "LeadingTau_mu", 1: "PairedLeadingTau_mu", 2: "NextLeadingTau_mu", 3: "PairedNextLeadingTau_mu"}

			temp_electron = electron_fourVec
			temp_muon = muon_fourVec

			min_elec_dR_dict = {}
			
			#Number of taus reconstructed as electrons, muons and taus
			n_electron_array = np.zeros(len(electron.E))
			n_muon_array = np.zeros(len(electron.E))
			n_hadron_array = np.zeros(len(electron.E))

			prev_min_elec_dR = []
			prev_min_muon_dR = []

			if not(self.isData):
				gen_elec_arr = ak.zeros_like(event_level.n_tau_muons)
				gen_muon_arr = ak.zeros_like(event_level.n_tau_muons)
				gen_had_arr = ak.zeros_like(event_level.n_tau_muons)

				Good_GenTau = Gen_Info[np.abs(Gen_Info.MotherId) == 25] #Ensure mother particles are Higgs
				#Good_GenTau = Good_GenTau[Good_GenTau.GMotherId == 25] #Ensure Taus come from Higgs
				#print(len(Good_GenTau))

				#Count muons, electrons and hadrons
				gen_elec_arr = ak.sum(np.abs(Good_GenTau.MCId) == 11,axis = 1)
				gen_muon_arr = ak.sum(np.abs(Good_GenTau.MCId) == 13,axis = 1)
				gen_had_arr = ak.sum(np.bitwise_and(np.abs(Good_GenTau.MCId) != 11,np.abs(Good_GenTau.MCId) != 13),axis=1)
				#gen_elec_arr = ak.where(np.abs(Good_GenTau.MCId) == 11, gen_elec_arr + 1, gen_elec_arr)
				#gen_muon_arr = ak.where(np.abs(Good_GenTau.MCId) == 13, gen_muon_arr + 1, gen_muon_arr)
				#gen_had_arr = ak.where(np.bitwise_and(np.abs(Good_GenTau.MCId) != 13, np.abs(Good_GenTau.MCId) != 11), gen_had_arr + 1, gen_had_arr)

				n_4had_gen = 0
				n_3had_1e_gen = 0
				n_3had_1mu_gen = 0
                
                #Truth debugging print statements
			#	for evnt in range(len(Good_GenTau)):
			#		print(gen_had_arr[evnt])
			#		if (gen_had_arr[evnt] + gen_muon_arr[evnt] + gen_elec_arr[evnt] > 4):
			#			print("!!!Too many gen level particles!!!!!")
			#			print("Electrons from tau: %d"%gen_elec_arr[evnt])
			#			print("Muons from tau: %d"%gen_muon_arr[evnt])
			#			print("Hardonic taus: %d"%gen_had_arr[evnt])
			#		if (gen_had_arr[evnt] + gen_muon_arr[evnt] + gen_elec_arr[evnt] < 4):
			#			print("!!!Too few gen level particles!!!!!!")
			#			print("Electrons from tau: %d"%gen_elec_arr[evnt])
			#			print("Muons from tau: %d"%gen_muon_arr[evnt])
			#			print("Hardonic taus: %d"%gen_had_arr[evnt])
			#		if (gen_had_arr[evnt] == 4):
			#			n_4had_gen += 1
			#		if (gen_had_arr[evnt] == 3 and gen_muon_arr[evnt] == 1):
			#			n_3had_1mu_gen += 1
			#		if (gen_had_arr[evnt] == 3 and gen_elec_arr[evnt] == 1):
			#			n_3had_1e_gen += 1
			#	
			#	print("Fraction of gen 4 hadron decays: %.3f"%(n_4had_gen/len(gen_elec_arr)))	
			#	print("Fraction of gen 3 hadron 1 electron decays: %.3f"%(n_3had_1e_gen/len(gen_elec_arr)))	
			#	print("Fraction of gen 3 hadron 1 muon decays: %.3f"%(n_3had_1mu_gen/len(gen_elec_arr)))	

			
			#Old style implementation (mostly awkward free)
			#print("==========Begin Mostly Awkward Free Algorithm==========")
			#
			#for n in range(len(temp_electron.t)): #Just use event loop
			#	#Keep track of the indicies of previously paired electrons and muons
			#	prev_used_ele = []
			#	prev_used_mu = []
			#	for i in range(4): #Loop over all taus
			#		elec_dR_Array = []
			#		muon_dR_Array = []
			#		tau_fourVec = ak.zip({"t": tau[n][i].E,"x": tau[n][i].Px, "y": tau[n][i].Py,"z" : tau[n][i].Pz},with_name = "Momentum4D")
			#	
			#		#Get lepton tau angular seperations
			#		for j_e in range(len(temp_electron[n])):
			#			elec_dR_Array.append(tau_fourVec.deltaR(temp_electron[n][j_e]))
			#			if (j_e in prev_used_ele): #Skip previously paired electrons
			#				continue
			#		if (len(temp_electron[n]) == 0): #Handle events with no electrons
			#			elec_dR_Array.append(10)
			#		for j_mu in range(len(temp_muon[n])):
			#			muon_dR_Array.append(tau_fourVec.deltaR(temp_muon[n][j_mu]))
			#			if (j_mu in prev_used_mu): #Skip previously paired muons
			#				continue
			#		if (len(temp_muon[n]) == 0): #Handle events with no muons
			#			muon_dR_Array.append(10)
			#		
			#		#Get the smallest delta R Values
			#		min_elec_dR = min(elec_dR_Array)
			#		min_muon_dR = min(muon_dR_Array) 

			#		if (min_elec_dR < min_muon_dR and min_elec_dR < 0.05):
			#			n_electron_array[n] += 1
			#			prev_used_ele.append(elec_dR_Array.index(min_elec_dR))
			#		if (min_muon_dR < min_elec_dR and min_muon_dR < 0.05):
			#			n_muon_array[n] += 1
			#			prev_used_mu.append(muon_dR_Array.index(min_muon_dR))
			#		if (min_elec_dR >= 0.05 and min_muon_dR >= 0.05):
			#			n_hadron_array[n] += 1
			#	#if (i == 0):
			#	#	print(prev_used_ele)
			#	#temp_electron = temp_electron[ak.from_iter(prev_used_ele)]#prev_elec_dR
			#	#temp_muon = temp_muon[ak.from_iter(prev_used_mu)]#prev_elec_dR

			#n_3had_1elec = 0
			#n_3had_1muon = 0
			#n_4had = 0
			#
			#for n in range(len(n_electron_array)):
			#	#Debugging work
			#	if (n_electron_array[n] + n_muon_array[n] + n_hadron_array[n] > 4):
			#		print("==========!!!More particles coming out than going in at event %d!!!=========="%n)

			#	#Count number of 4 hadronic, and 3 hadronic + 1 lepton states
			#	if (n_hadron_array[n] == 4):
			#		n_4had += 1
			#	if (n_electron_array[n] == 1 and n_hadron_array[n] == 3):
			#		n_3had_1elec += 1
			#	if (n_muon_array[n] == 1 and n_hadron_array[n] == 3):
			#		n_3had_1muon += 1

			#print("==========Checking the branching factions==========")
			#print("Observed fraction of 4 hadron fraction: %.3f"%(n_4had/len(n_hadron_array)))
			#print("Expcted fraction of 4 hadron fraction: %.3f"%((2/3)**4))
			#print("Observed fraction of 3 hadron + 1 electron fraction: %.3f"%(n_3had_1elec/len(n_hadron_array)))
			#print("Expcted fraction of 3 hadron + 1 electron fraction: %.3f"%((2/3)**3*(1/6)))
			#print("Observed fraction of 3 hadron + 1 muon fraction: %.3f"%(n_3had_1muon/len(n_hadron_array)))
			#print("Expcted fraction of 3 hadron + 1 muon fraction: %.3f"%((2/3)**3*(1/6)))
			#print("==========End Mostly Awkward Free Algorithm==========")
		
			#Awkward implementation
			for i in range(4):
				print("Tau %d"%i)
				tau_fourVec = ak.zip({"t": tau[:,i].E,"x": tau[:,i].Px, "y": tau[:,i].Py,"z" : tau[:,i].Pz},with_name = "Momentum4D")
				elec_dR = tau_fourVec.deltaR(electron_fourVec)
				muon_dR = tau_fourVec.deltaR(muon_fourVec)

				#Choose leptons with smallest delta Rs such that are < 0.05 
				misId_ele_cond = np.bitwise_and(elec_dR == ak.min(elec_dR,axis=1),elec_dR < 0.05)
				
				misId_ele_cond = ak.fill_none(misId_ele_cond,[False],axis=0) #Find the the smallest dR between electron and tau
				if (i == 0):
					not_prev_id_elec = ak.ones_like(misId_ele_cond)*True
				misId_ele_cond = np.bitwise_and(misId_ele_cond,not_prev_id_elec)

				#Check that there is only one electron
				
				#Update list of paired leptons
				#if (i == 0):
				#	not_prev_id_elec = np.bitwise_not(misId_ele_cond)
				#else:
				#	not_prev_id_elec = np.bitwise_or(not_prev_id_elec,np.bitwise_not(np.bitwise_and(misId_ele_cond)))
				
				min_elec_dR = ak.min(elec_dR[misId_ele_cond],axis=1) 
				min_elec_dR = ak.fill_none(min_elec_dR,10) #Fill Nones with impossibly large values
				#print("================Electron %d====================="%i)
				#for j in range(10):
				#	print("%dth Event:"%j)
				#	print("Minimum Delta R = %f"%min_elec_dR[j])
			
				#print(min_elec_dR)
				#for x in min_elec_dR:
					#print(x)
					#if x > 1:
					#	print("!!!More than 1 min electron (unexpected)!!!")
				
				#misId_mu_cond = valid_mu_dR == ak.min(muon_dR,axis=1)
				misId_mu_cond = np.bitwise_and(muon_dR == ak.min(muon_dR,axis=1),muon_dR < 0.05)
				misId_mu_cond = ak.fill_none(misId_mu_cond,[False],axis=0) #Find the the smallest dR between muon and tau
				if (i == 0):
					not_prev_id_muon = ak.ones_like(misId_mu_cond)*True
				misId_mu_cond = np.bitwise_and(misId_mu_cond,not_prev_id_muon)
				
				#Update list of paired leptons
				#if (i == 0):
				#	not_prev_id_muon = np.bitwise_not(misId_mu_cond)
				#else:
				#	not_prev_id_muon = np.bitwise_or(not_prev_id_muon,np.bitwise_not(misId_mu_cond))
				#if (i != 0):
				#	misId_mu_cond = np.bitwise_and(misId_mu_cond,not_prev_id_muon)

				min_mu_dR = ak.min(muon_dR[misId_mu_cond],axis=1)
				#min_mu_dR = muon_dR[misId_mu_cond]
				min_mu_dR = ak.fill_none(min_mu_dR,10) #Fill Nones with impossibly large values
				
				#Seperate electrons and muons into those matched and those not matched
				misId_ele = electron_fourVec[misId_ele_cond]
				indxNum = 0
				for x in ak.num(misId_ele.t,axis=1):
				    if x > 1:
					    print("!!Multiple electrons matched at event %d!!"%indxNum)
				    indxNum +=1
				misId_mu = muon_fourVec[misId_mu_cond]
				indxNum = 0
				for x in ak.num(misId_mu.t,axis=1):
				    if x > 1:
					    print("!!Multiple muons matched at event %d!!"%indxNum)
				    indxNum +=1
				
				#not_prev_id_elec = np.bitwise_not(misId_ele_cond)
				#not_prev_id_muon = np.bitwise_not(misId_ele_cond)
				#Drop paired leptons from further consideration
				#temp_electron = temp_electron[temp_electron.E != misId_ele.t]
				#temp_muon = temp_muon[temp_muon.E != misId_mu.t]
				#electron_fourVec = electron_fourVec[electron_fourVec.t != misId_ele.t]
				#muon_fourVec = muon_fourVec[muon_fourVec.t != misId_mu.t]

				#Check size of min_elec_dR and min_mu_dR
				#for x in min_elec_dR:

				use_ele = min_elec_dR < min_mu_dR #, min_elec_dR < 0.05) #,not_prev_id_elec)
				use_mu = min_elec_dR > min_mu_dR #, min_mu_dR < 0.05) #,not_prev_id_muon)
				use_had = np.bitwise_not(np.bitwise_or(use_ele,use_mu))
				
				#(Commented out 8 June 2025 for (hopefully) increased speed)
				#for a,b in zip(use_ele,use_mu):
				#	if (a and b):
				#		print("!!!!Use both electron and muon!!!!")

				#Update paired electrons and muons
				#print(len(use_ele))
				#print(len(misId_ele_cond))
				not_prev_id_elec = np.bitwise_and(not_prev_id_elec,np.bitwise_not(misId_ele_cond*ak.ravel(use_ele)))
				not_prev_id_muon = np.bitwise_and(not_prev_id_muon,np.bitwise_not(misId_mu_cond*ak.ravel(use_mu)))
				
				#Count number of taus originating from leptons (Is this logic broken??)
				event_level["n_tau_electrons"] = ak.where(ak.all(ak.singletons(use_ele) == True,axis=1),event_level["n_tau_electrons"] + 1, event_level["n_tau_electrons"])
				event_level["n_tau_muons"] = ak.where(ak.all(ak.singletons(use_mu) == True,axis=1),event_level["n_tau_muons"] + 1, event_level["n_tau_muons"]) 
				event_level["n_tau_hadronic"] = ak.where(ak.all(ak.singletons(use_had) == True,axis=1),event_level["n_tau_hadronic"] + 1,event_level["n_tau_hadronic"])

				#print("Max number of electrons = %d"%ak.max(event_level.n_electrons,axis=0))
				#print("Max number of muons = %d"%ak.max(event_level.n_muons,axis=0))
                

				if (i == 3):
					n_wrong0 = 0
					n_wrong1 = 0
					n_wrong2 = 0
					for j in range(len(event_level.n_tau_electrons)):
						if (event_level[j].n_tau_electrons > 4):
							n_wrong0 += 1
						if (event_level[j].n_tau_muons > 4):
							n_wrong1 += 1
						if (event_level[j].n_tau_muons + event_level[j].n_tau_electrons > 4):
							n_wrong2 += 1
					
					print("%d events have more than 4 electrons"%n_wrong0)
					print("%d events have more than 4 muons"%n_wrong1)
					print("%d events in which electrons + muons > 4"%n_wrong2)

					#Check how many 3 hardonic tau + 1 leptonic tau events there are
					num_3h1mu = 0
					num_3h1e = 0
					num_4h = 0 
					for j in range(len(event_level.n_tau_hadronic)):
						if (event_level[j].n_tau_hadronic == 4):
							num_4h += 1
						if (event_level[j].n_tau_hadronic == 3):
							if (event_level[j].n_tau_electrons == 1):
								num_3h1e += 1
							if (event_level[j].n_tau_muons == 1):
								num_3h1mu += 1
							if (event_level[j].n_tau_muons > 1 or event_level[j].n_tau_electrons > 1):
								print("!!!!=======================Electron and/or muon miscount=======================!!!!")
					print("Fraction of events with 3 hadronic taus and 1 muon: %.3f"%(num_3h1mu/len(event_level.n_tau_hadronic)))
					print("Fraction of events with 4 hadronic taus: %.3f"%(num_4h/len(event_level.n_tau_hadronic)))
					print("Fraction of events with 3 hadronic taus and 1 electron: %.3f"%(num_3h1e/len(event_level.n_tau_hadronic)))


				#Store reco information of taus
				#event_level[Hadronic_Dict[i]] = np.bitwise_not(np.bitwise_or(use_ele,use_mu)) #I think the logic (with bitwise_and) is broken
				#event_level[electron_Dict[i]] = use_ele
				#event_level[muon_Dict[i]] = use_mu

				#Keep these lines meant to swich in electrons and muons when tau misidentified as them
				#tau_fourVec = ak.where(use_ele, misId_ele, tau_fourVec) #Replace lead tau with electron when applicable
				#tau_fourVec = ak.where(use_mu, misId_mu, tau_fourVec) #Replace lead tau with electron when applicable
				
				#if (i == 0):
				#	misId_ele = electron_fourVec[misId_ele_cond]
				#	misId_mu = muon_fourVec[misId_mu_cond]

					#Debugging this implementation
				#	for evnt in range(ak.num(misId_ele,axis=0)):
				#		num_e = ak.num(misId_ele[evnt],axis=0)
				#		num_mu = ak.num(misId_mu[evnt],axis=0)

				#		if (num_e > 1 or num_mu > 1):
				#			print("Lead tau reconstructed to %d electrons and %d muons"%(num_e, num_mu))
				#		if (use_ele[evnt] == use_mu[evnt] and use_ele[evnt]):
				#			print("Leading tau simultaneously matched to an electron and a muon")
				#			print(min_elec_dR[evnt])
				#			print(min_mu_dR[evnt])
				#			print(tau_fourVec[evnt])
				#			print(misId_ele[evnt])
				#			print(misId_mu[evnt])
				#			print("==============================================================")

				#		if (use_ele[evnt]):
				#			if (tau_fourVec[evnt].E != misId_ele[evnt].E):
				#				print("========!!Electron not swapped in!!========")
				#		if (use_mu[evnt]):
				#			if (tau_fourVec[evnt].E != misId_mu[evnt].E):
				#				print("========!!Muon not swapped in!!========")

				#print("Electron four vec size before selection: %d"%ak.num(temp_electron,axis=0))
				#print("Event 0 4-vector: " + str(electron_fourVec[0]))
				#test_size = ak.num(misId_ele_cond)
				#for j in range(len(test_size)):
				#	print(j)
				#	print(temp_electron[j].E)
				#	print(np.bitwise_not(misId_ele_cond[j]))
				

				#temp_electron = temp_electron[np.bitwise_not(misId_ele_cond)]
				#electron_fourVec = electron_fourVec[np.bitwise_not(misId_ele_cond)]
				#print("Electron four vec size after selection: %d"%ak.num(temp_electron,axis=0))
				#print("Event 0 4-vector: " + str(electron_fourVec[0]))
				#print(electron_fourVec.E)
				#for j in range(len(temp_electron.E)):
				#	print("Event %d"%j)
				#	print(temp_electron[j].E)
				#print("Length of muon Id array %d"%ak.num(misId_mu_cond,axis=0))
				#print(type(temp_muon))
				#print(type(misId_mu_cond))
				#print(temp_muon.type)
				#print(misId_mu_cond.type)
				#ak.Array.show(temp_muon)
				#ak.Array.show(misId_mu_cond)
				#misId_mu_cond.show()
				#temp_muon.show()
				#id_num = 0
				#print(ak.num(np.bitwise_not(misId_mu_cond),axis=1))
				#print(ak.num(temp_muon,axis=1))
				#print(ak.num(np.bitwise_not(misId_ele_cond),axis=1))
				#print(ak.num(temp_electron,axis=1))
				#for x,y in zip(np.bitwise_not(misId_mu_cond),temp_muon.E):
				#	print(id_num)
				#	print(x)
				#	print(y)
				#	id_num += 1
				#tmp_arr = np.bitwise_not(misId_mu_cond)
				#print(type(tmp_arr))
				#temp_muon = temp_muon[np.bitwise_not(misId_mu_cond)]
				#muon_fourVec = muon_fourVec[np.bitwise_not(misId_mu_cond)]

				#Update taus
				tau[:,i]["E"] = tau_fourVec.t
				tau[:,i]["Px"] = tau_fourVec.x
				tau[:,i]["Py"] = tau_fourVec.y
				tau[:,i]["Pz"] = tau_fourVec.z
				
			#Obtain di-tau delta R and higgs delta R (if there are any events left)
			if (ak.num(tau,axis=0) > 0):
				tau1 = tau[np.bitwise_and(tau.E == tau[:,0].E,np.bitwise_and(np.bitwise_and(tau.Px == tau[:,0].Px, tau.Py == tau[:,0].Py), tau.Pz == tau[:,0].Pz))]
				tau2 = tau[np.bitwise_and(tau.E == tau[:,1].E,np.bitwise_and(np.bitwise_and(tau.Px == tau[:,1].Px, tau.Py == tau[:,1].Py), tau.Pz == tau[:,1].Pz))]
				tau3 = tau[np.bitwise_and(tau.E == tau[:,2].E,np.bitwise_and(np.bitwise_and(tau.Px == tau[:,2].Px, tau.Py == tau[:,2].Py), tau.Pz == tau[:,2].Pz))]
				tau4 = tau[np.bitwise_and(tau.E == tau[:,3].E,np.bitwise_and(np.bitwise_and(tau.Px == tau[:,3].Px, tau.Py == tau[:,3].Py), tau.Pz == tau[:,3].Pz))]
	
				#Check on the number of taus
				tau_num_arr = ak.num(tau.pt,axis=1)
				
				#Drop the goddman events with anomolous numbers of taus
				#tau = tau[tau_num_arr != 4]
				#Jet = Jet[tau_num_arr != 4]
				#AK8Jet = AK8Jet[tau_num_arr != 4]
				#event_level = event_level[tau_num_arr != 4]
				#electron = electron[tau_num_arr != 4]
				#muon = muon[tau_num_arr != 4]


				#Print statements for debugging events with anomolous numbers of taus
				tau_notFour = tau_num_arr[tau_num_arr != 4]
				print("Number of events with unexpected number of taus: %d"%ak.num(tau_notFour,axis=0))
				print("Number of events with 4 taus: %d"%ak.num(tau_num_arr[tau_num_arr == 4],axis=0))
				print("Number of events with 3 taus: %d"%ak.num(tau_num_arr[tau_num_arr == 3],axis=0))
				print("Number of events with 2 taus: %d"%ak.num(tau_num_arr[tau_num_arr == 2],axis=0))
				print("Number of events with 5 taus: %d"%ak.num(tau_num_arr[tau_num_arr == 5],axis=0))
				print("Number of events with 6 taus: %d"%ak.num(tau_num_arr[tau_num_arr == 6],axis=0))

				#Look at the event with 5 taus to see what's going wrong
				print("=================================Inspection of event with 5 taus===========================================")
				print("Tau Energies: " + str(tau[tau_num_arr == 5].E))
				print("Tau Pt: " + str(tau[tau_num_arr == 5].pt))
				print("Tau Px: " + str(tau[tau_num_arr == 5].Px))
				print("Tau Py: " + str(tau[tau_num_arr == 5].Py))
				print("Tau Pz: " + str(tau[tau_num_arr == 5].Pz))
				print("Tau phi: " + str(tau[tau_num_arr == 5].eta))
				print("Tau eta: " + str(tau[tau_num_arr == 5].phi))
				print("Number of boosted taus " + str(tau[tau_num_arr == 5].nBoostedTau))
				print("Event number:" + str(event_level[tau_num_arr == 5].event_num))
				print("Run:" + str(event_level[tau_num_arr == 5].run))
				print("LumiBlock:" + str(event_level[tau_num_arr == 5].Lumi))
				print("============================================================================")
				
				test_arr3 = ak.num(tau3.eta,axis=1) 
				num_3 = ak.num(test_arr3,axis=0)
				test_arr4 = ak.num(tau4.eta,axis=1)
				num_4 = ak.num(test_arr4,axis=0)
				
				test_sizes = test_arr3 == test_arr4
				test_sizes[test_sizes]
				if (num_3 != num_4):
					print("Different numbers of next leading and paired taus (this is a problem)")
				if (ak.num(test_sizes,axis=0) == num_3):
					print("Issue with dimensions of the taus")
				#End print statments for debugging events with anomolous number of taus



				leading_higgs = ak.zip({
						"x": tau1.Px + tau2.Px,
						"y": tau1.Py + tau2.Py,
						"z": tau1.Pz + tau2.Pz,
						"t": tau1.E + tau2.E
					},with_name="Momentum4D"
				)
				#leading_higgs["phi"] = ak.from_iter(np.arctan2(leading_higgs.Py,leading_higgs.Px))
				#leading_higgs["eta"] = ak.from_iter(np.arcsinh(leading_higgs.Pz)/np.sqrt(leading_higgs.Px**2 + leading_higgs.Py**2 + leading_higgs.Pz**2))
				
				nextleading_higgs = ak.zip({
						"x": tau3.Px + tau4.Px,
						"y": tau3.Py + tau4.Py,
						"z": tau3.Pz + tau4.Pz,
						"t": tau3.E + tau4.E
					},with_name="Momentum4D"
				)
				#nextleading_higgs["phi"] = ak.from_iter(np.arctan2(nextleading_higgs.Py,nextleading_higgs.Px))
				#nextleading_higgs["eta"] = ak.from_iter(np.arcsinh(nextleading_higgs.Pz)/np.sqrt(nextleading_higgs.Px**2 + nextleading_higgs.Py**2 + nextleading_higgs.Pz**2))

				#Why has the delta R thing broken now that I have stopped checking charge??
				#print(leading_higgs.phi)
				#print(nextleading_higgs.eta)
		
				#Visiable Mass selection
				if (ak.num(event_level.MHT,axis=0) > 0):
					#vis_mass1 = ak.concatenate((single_mass(higgs_11),single_mass(higgs_12)),axis=1)
					#for x in range(3):
						#print(vis_mass1[x])
					#vis_mass2 = ak.concatenate((single_mass(higgs_22),single_mass(higgs_21)),axis=1)
					#for x in range(3):
						#print(vis_mass2[x])
					#vis_mass1 = single_mass(leading_higgs)
					#vis_mass2 = single_mass(nextleading_higgs)
					vis_mass1 = leading_higgs.mass
					vis_mass2 = nextleading_higgs.mass
					vis_cond1 = ak.any(vis_mass1 >= 10,axis=1)
					vis_cond2 = ak.any(vis_mass2 >= 10,axis=1)
					vis_cond = np.bitwise_and(vis_cond1, vis_cond2)
					
					tau = tau[vis_cond]
					Jet = Jet[vis_cond]
					AK8Jet = AK8Jet[vis_cond]
					muon = muon[vis_cond]
					electron = electron[vis_cond]
					leading_higgs = leading_higgs[vis_cond]
					nextleading_higgs = nextleading_higgs[vis_cond]
					event_level = event_level[vis_cond]

					cutflow_table.fill(5,weight = ak.num(tau,axis=0))
					if (self.isData or not(self.isData)):
						print("# of events after visible mass cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))


				#higgs_dR = deltaR(leading_higgs, nextleading_higgs) 
				higgs_dR = leading_higgs.deltaR(nextleading_higgs) #Use vector library for delta R calculations

				higgs_cond = ak.all(higgs_dR >= 2.0, axis = 1) #Require Higgs to have seperation deltaR >= 2
				#tau_cond = ak.all(ditau_dR < 0.8,axis = 1)
				#topo_cond = np.bitwise_and(tau_cond, higgs_cond) 
				topo_cond = higgs_cond
		
				#Apply Higgs Topological Condition	
				tau = tau[higgs_cond]
				Jet = Jet[higgs_cond]
				AK8Jet = AK8Jet[higgs_cond]
				muon = muon[higgs_cond]
				electron = electron[higgs_cond]
				event_level = event_level[higgs_cond]
				leading_higgs = leading_higgs[higgs_cond]
				nextleading_higgs = nextleading_higgs[higgs_cond]
				cutflow_table.fill(6,weight = ak.num(tau,axis=0))
				#tau_cond = tau_cond[higgs_cond]
				if (self.isData or not(self.isData)):
					print("# of events after Higgs cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
	
			#Apply Tau topological condition	
			#tau = tau[tau_cond]
			#Jet = Jet[tau_cond]
			#AK8Jet = AK8Jet[tau_cond]
			#event_level = event_level[tau_cond]
			#leading_higgs = leading_higgs[tau_cond]
			#nextleading_higgs = nextleading_higgs[tau_cond]
			#if (self.isData or not(self.isData)):
			#	print("# of events after di-tau delta R cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
			
			#Also apply selection to higgs events to add a visable mass cut
			#leading_higgs = leading_higgs[topo_cond]
			#nextleading_higgs = nextleading_higgs[topo_cond]
			#higgs_11 = higgs_11[topo_cond]
			#higgs_22 = higgs_22[topo_cond]
			#higgs_12 = higgs_12[topo_cond]
			#higgs_21 = higgs_21[topo_cond]
		
			if (self.isData or not(self.isData)):
				print("# of events after topology cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))


		#Apply BJet multiplicity selection
		#Apply pt, eta, loose ID, and deep csv tag cut
		Jet_B = Jet[Jet.PFLooseId > 0.5]
		Jet_B = Jet_B[Jet_B.Pt > 30]
		Jet_B = Jet_B[np.abs(Jet_B.eta) < 2.4]
		Jet_B = Jet_B[Jet_B.DeepCSVTags_b > 0.7527]
		NumBJets = ak.num(Jet_B,axis=1)
		event_level["nBJets"] = NumBJets

		#Z Multiplicity function
		def Z_Mult_Function(lepton,lep_flavor): 
			#Make Good muon selection
			if (lep_flavor == "mu"):
				if (self.trigger_bit == 39):
					#id_cond = np.bitwise_and(lepton.IDbit,2) != 0 #Do not delete
					id_cond = muon.IDSelec 
					d0_cond = np.abs(lepton.D0) < 0.045
					dz_cond = np.abs(lepton.Dz) < 0.2
					good_lepton_cond = np.bitwise_and(id_cond, np.bitwise_and(d0_cond, dz_cond))
					good_lepton = lepton[good_lepton_cond]
				else:
					good_lepton = lepton
			#Make good electron selection
			if (lep_flavor == "ele"):
				cond1 = np.bitwise_and(np.abs(lepton.SCEta) <= 0.8, lepton.IDMVANoIso > 0.837)
				cond2 = np.bitwise_and(np.bitwise_and(np.abs(lepton.SCEta) > 0.8, np.abs(lepton.SCEta) <= 1.5), electron.IDMVANoIso > 0.715)
				cond3 = np.bitwise_and(np.abs(lepton.SCEta) >= 1.5, electron.IDMVANoIso > 0.357)
				good_lepton_cond = np.bitwise_or(cond1,np.bitwise_or(cond2,cond3))
				good_lepton = lepton[good_lepton_cond]
			
			print("Number of lepton filled events before Z-multiplicty building: %d"%ak.num(good_lepton,axis=0))
			Z_Mult = find_Z_Candidates(good_lepton,ak.ArrayBuilder()).snapshot()

			return Z_Mult
			
	
		#Get Z_multiplicity	
		electron_ZMult = Z_Mult_Function(electron,"ele")
		muon_ZMult = Z_Mult_Function(muon,"mu")
		#ZMult_Frozen = muon_ZMult + electron_ZMult
		event_level["ZMult"] = muon_ZMult + electron_ZMult
		#event_level["ZMult"] = event_level["ZMult_tau"]
		event_level["ZMult_e"] = electron_ZMult
		event_level["ZMult_mu"] = muon_ZMult

		tau = tau[ak.num(tau,axis=1) > 0] #Handle empty arrays left by the trigger

		
		#Get the leading Higgs 4-momenta
		PxLeading = tau[:,0].Px + tau[:,1].Px
		PyLeading = tau[:,0].Py + tau[:,1].Py
		PzLeading = tau[:,0].Pz + tau[:,1].Pz
		ELeading = tau[:,0].E + tau[:,1].E
		
		#Get the subleading Higgs 4-momenta
		PxSubLeading = tau[:,2].Px + tau[:,3].Px
		PySubLeading = tau[:,2].Py + tau[:,3].Py
		PzSubLeading = tau[:,2].Pz + tau[:,3].Pz
		ESubLeading = tau[:,2].E + tau[:,3].E

		#Get pair delta R and delta phi Distributions
		leading_dR_Arr = ak.ravel(deltaR(tau[:,0],tau[:,1]))
		leading_dPhi_Arr = ak.ravel(delta_phi(tau[:,0],tau[:,1]))
		subleading_dR_Arr = ak.ravel(deltaR(tau[:,2],tau[:,3]))
		subleading_dPhi_Arr = ak.ravel(delta_phi(tau[:,2],tau[:,3]))
			
		#Reconstructed Higgs Objects
		Higgs_Leading = ak.zip(
			{
				"Px" : ak.from_iter(PxLeading),
				"Py" : ak.from_iter(PyLeading),
				"Pz" : ak.from_iter(PzLeading),
				"E" : ak.from_iter(ELeading)
			}
		)
		Higgs_Leading["phi"] = ak.from_iter(np.arctan2(Higgs_Leading.Py,Higgs_Leading.Px))
		Higgs_Leading["eta"] = ak.from_iter(np.arcsinh(Higgs_Leading.Pz)/np.sqrt(Higgs_Leading.Px**2 + Higgs_Leading.Py**2 + Higgs_Leading.Pz**2))
		Higgs_SubLeading = ak.zip(
			{
				"Px" : ak.from_iter(PxSubLeading),
				"Py" : ak.from_iter(PySubLeading),
				"Pz" : ak.from_iter(PzSubLeading),
				"E" : ak.from_iter(ESubLeading)
			}
		)
		Higgs_SubLeading["phi"] = ak.from_iter(np.arctan2(Higgs_SubLeading.Py,Higgs_SubLeading.Px))
		Higgs_SubLeading["eta"] = ak.from_iter(np.arcsinh(Higgs_SubLeading.Pz)/np.sqrt(Higgs_SubLeading.Px**2 + Higgs_SubLeading.Py**2 + Higgs_SubLeading.Pz**2))


		#Reconstructed Radion
		Radion_Reco = ak.zip(
				{
					"Px": Higgs_Leading.Px + Higgs_SubLeading.Px,
					"Py": Higgs_Leading.Py + Higgs_SubLeading.Py,
					"Pz": Higgs_Leading.Pz + Higgs_SubLeading.Pz,
					"E": Higgs_Leading.E + Higgs_SubLeading.E,
				}
		)
		#Radion_4Vec = vector.LorentzVectov(ak.zip({"t": Radion_Reco.E,"x": Radion_Reco.Px,"y": Radion_Reco.Py,"z": Radion_Reco.Pz},with_name="LorentzVector"))
		Radion_4Vec = ak.zip({"t": Radion_Reco.E,"x": Radion_Reco.Px,"y": Radion_Reco.Py,"z": Radion_Reco.Pz},with_name="Momentum4D")
		Radion_Reco["phi"] = ak.from_iter(np.arctan2(Radion_Reco.Py,Radion_Reco.Px))
		Radion_Reco["eta"] = Radion_4Vec.eta #ak.from_iter(np.arcsinh(Radion_Reco.Pz)/np.sqrt(Radion_Reco.Px**2 + Radion_Reco.Py**2 + Radion_Reco.Pz**2))
		event_level["Radion_Charge"] = tau[:,0].charge + tau[:,1].charge + tau[:,2].charge + tau[:,3].charge
		event_level["LeadingHiggs_Charge"] = tau[:,0].charge + tau[:,1].charge
		event_level["SubleadingHiggs_Charge"] = tau[:,2].charge + tau[:,3].charge
		#print("Leading Charge: " + str(event_level.LeadingHiggs_Charge))
		#print("Subleading Charge: " + str(event_level.SubleadingHiggs_Charge))
		
		if (len(Higgs_Leading.eta) != 0):
			#print("Mass Reconstructed")
			diHiggs_dR_Arr = ak.ravel(deltaR(Higgs_Leading,Higgs_SubLeading))
			LeadingHiggs_mass_Arr = ak.ravel(single_mass(Higgs_Leading))	
			SubLeadingHiggs_mass_Arr = ak.ravel(single_mass(Higgs_SubLeading))
		
			#Obtain delta R between each Higgs and the radion
			leadingHiggs_Rad_dR = ak.ravel(deltaR(Higgs_Leading,Radion_Reco))
			subleadingHiggs_Rad_dR = ak.ravel(deltaR(Higgs_SubLeading,Radion_Reco))
		
			#Obtain Delta phi between MET and Each Higgs
			leadingHiggs_MET_dPhi_Arr = ak.ravel(MET_delta_phi(Higgs_Leading,event_level))
			subleadingHiggs_MET_dPhi_Arr = ak.ravel(MET_delta_phi(Higgs_SubLeading,event_level))
		else:
			#if (self.isData):
			#print("Mass Not Reconstructed")
			diHiggs_dR_Arr = np.array([])
			LeadingHiggs_mass_Arr = np.array([])
			SubLeadingHiggs_mass_Arr = np.array([])
			leadingHiggs_Rad_dR = np.array([])
			subleadingHiggs_Rad_dR = np.array([])
			leadingHiggs_MET_dPhi_Arr = np.array([])
			subleadingHiggs_MET_dPhi_Arr = np.array([])
		
		#Fill Higgs Delta Phi
		phi_leading = np.arctan2(PyLeading,PxLeading)
		phi_subleading = np.arctan2(PySubLeading,PxSubLeading)
		Higgs_DeltaPhi_Arr = ak.ravel((phi_leading - phi_subleading + np.pi) % (2 * np.pi) - np.pi)
		radionPT_HiggsReco = np.sqrt((PxLeading + PxSubLeading)**2 + (PyLeading + PySubLeading)**2)
		radionPT_Arr = ak.ravel(radionPT_HiggsReco)

		#Obtain delat Phi between MET and Radion
		radionMET_dPhi = ak.ravel(MET_delta_phi(Radion_Reco,event_level))

		#print(len(tau))
		FourTau_Mass_Arr =four_mass([tau[:,0],tau[:,1],tau[:,2],tau[:,3]]) #ak.ravel(tau.FourMass)
	
		#Obtain the weight	
		if (self.isData):
			CrossSec_Weight = 1 
		else:
			CrossSec_Weight = weight_calc(dataset,numEvents_Dict[dataset])
			print("=========!!!Weight Debugging!!!=========")
			print(dataset)
			print("Luminosity Weight = %f"%CrossSec_Weight)
			print("Luminosity = %f"%Lumi_2018)
			print("Cross section = %f"%xSection_Dictionary[dataset])
			print("Number of events Processed: %d"%numEvents_Dict[dataset])
		
		#Efficiency Histograms
		if (self.isData):
			crossSecVal = 1
		else:
			crossSecVal = xSection_Dictionary[dataset]

		#Get weight contribution from individual event weights
		ind_event_weight = 1
		if not(self.isData):
			ind_event_weight = ak.prod(event_level.event_weight,axis=0)
		
		#Store data for NN as parquet file
		print(event_level.ZMult)
		print(radionPT_Arr)
		var_nn = ak.zip( #Variables to be exported to .parquet file
			{
				"radion_pt": radionPT_Arr,
				"vis_mass": LeadingHiggs_mass_Arr,
				"vis_mass2": SubLeadingHiggs_mass_Arr,
				"radion_eta": Radion_Reco.eta,
				"higgs1_dr": leading_dR_Arr,
				"higgs2_dr": subleading_dR_Arr,
				"dphi_H1": phi_leading,
				"dphi_H2": phi_subleading,
				"dphi_H1_MET": leadingHiggs_MET_dPhi_Arr,
				"dphi_H2_MET": subleadingHiggs_MET_dPhi_Arr,
				"dr_HH": Higgs_DeltaPhi_Arr,
				"dphi_HH": Higgs_DeltaPhi_Arr,
				"dr_H1_Rad": leadingHiggs_MET_dPhi_Arr,
				"dr_H2_Rad": subleadingHiggs_Rad_dR,
				"dphi_rad_MET": radionMET_dPhi,
				"H1OS": event_level.LeadingHiggs_Charge,
				"H2OS": event_level.SubleadingHiggs_Charge,
				"ZMult": ak.ravel(event_level.ZMult), 
				"numBJet": event_level.nBJets,
				"RecoRadion_Mass": FourTau_Mass_Arr,
				"weight": event_level.event_weight*CrossSec_Weight,
				}
			)

		#if (dataset == "ZZ4l"):
		#	print("radion_pt being stored:")
		#	print(radionPT_Arr)
		#	print("weight being stored:")
		#	print(event_level.event_weight*CrossSec_Weight)

		if not(self.isData):
			file_name = (dataset + ".parquet")
			if (dataset != "Signal"):
				if (mass == "2000"):
					file_name = dataset  + ".parquet"
					print("Background")
			else:
				file_name = dataset + "_mass_" + self.massVal + "GeV.parquet"
				print("Signal")
			ak.to_parquet(var_nn,file_name)
			print("Creating Parquet file (MC)")
		else:
			#event_level.event_weight = event_level.event_weight ** 0 #Force data to have weight 1/ no weighting
			file_name = (dataset + ".parquet")
			if (mass == "2000"):
				if (os.path.isfile(file_name)): #Append to existing parquet file
					file_data = ak.from_parquet(file_name)
					var_nn = ak.concatenate([file_data,var_nn])
					ak.to_parquet(var_nn,file_name)
					print("Appending parquet file (Data)")
				else:
					ak.to_parquet(var_nn,file_name) #Create parquet file
					print("Creating Parquet file (Data)")

		#print(CrossSec_Weight)
		#print(event_level.event_weight)
		print("===================!!!=Weight Debugging!!!====================")
		print(event_level.event_weight*CrossSec_Weight)
		print("===================!!!=Weight Debugging!!!====================")
		#print("===================!!!=Raw Event Count!!!====================")
		#print(ak.num(event_level.event_weight,axis=0))
		#print("===================!!!=Raw Event Count!!!====================")
			
		#Histogram bining
		if (self.trigger_bit == 39): #Use reduced binning for JetHT trigger
			N1 = 6 
			N2 = 6 
		else:
			N1 = 10 
			N2 = 8 
		
		#Create histograms to write out
		h_FourTauMass = hist.Hist.new.Regular(N1,0,3000, label = r"$m_{4\tau}$ [GeV]").Double()
		h_HiggsDeltaPhi = hist.Hist.new.Regular(N1,-pi,pi, label = r"Higgs $\Delta \phi$").Double() 
		h_HiggsDeltaR = hist.Hist.new.Regular(N1,0,5, label = r"Higgs $\Delta$R").Double()
		h_LeadDiTauDeltaR = hist.Hist.new.Regular(N1,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double()
		h_SubLeadingDiTauDeltaR = hist.Hist.new.Regular(N1,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double()
		h_LeadHiggsMass = hist.Hist.new.Regular(N2,0,200, label=r"Leading Higgs Mass (GeV)").Double()
		h_SubleadingHiggsMass = hist.Hist.new.Regular(N2,0,200, label=r"Sub-Leading Higgs Mass (GeV)").Double()
		h_RadionpT = hist.Hist.new.Regular(N1,0,200, label=r"Radion $p_T$ (GeV)").Double()
		h_taupT = hist.Hist.new.Regular(N1,0,400, label=r"$\tau$ $p_T$ (GeV)").Double()
		h_tauEta = hist.Hist.new.Regular(N1,-5,5, label = r"$\tau \ \eta$").Double()
		h_ZMult = hist.Hist.new.Regular(6,0,6, label = r"Z Boson Multiplicity").Double()
		h_ZMultEle = hist.Hist.new.Regular(6,0,6, label = r"Z Boson Multiplicity (electrons only)").Double()
		h_ZMultMu = hist.Hist.new.Regular(6,0,6, label = r"Z Boson Multiplicity (muons only)").Double()
		h_ZMultTau = hist.Hist.new.Regular(6,0,6, label = r"Z Boson Multiplicity (from taus)").Double()
		h_BJetMult = hist.Hist.new.Regular(6,0,6, label = r"BJet Multiplicity").Double()
		h_LeadTaupT = hist.Hist.new.Regular(N1,0,400, label=r"Leading $\tau$ $p_T$ (GeV)").Double()
		h_SubleadTaupT = hist.Hist.new.Regular(N1,0,400, label=r"Subleading $\tau$ $p_T$ (GeV)").Double()
		h_3TaupT = hist.Hist.new.Regular(N1,0,400, label=r"Third leading $\tau$ $p_T$ (GeV)").Double()
		h_4TaupT = hist.Hist.new.Regular(N1,0,400, label=r"Fourth leading $\tau$ $p_T$ (GeV)").Double()
		h_LeadDiTauPhi = hist.Hist.new.Regular(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double() 
		h_SubleadDiTauPhi = hist.Hist.new.Regular(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double() 
		h_RadMETPhi = hist.Hist.new.Regular(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double() 
		h_RadLeadHiggsDeltaR = hist.Hist.new.Regular(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double()
		h_RadSubleadingHiggsDeltaR = hist.Hist.new.Regular(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double()
		h_METLeadHiggsDeltaPhi = hist.Hist.new.Regular(N1,-pi,pi, label = r"Leading Higgs MET $\Delta \phi$").Double() 
		h_METSubleadHiggsDeltaPhi = hist.Hist.new.Regular(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta \phi$").Double()
		h_RadionEta = hist.Hist.new.Regular(N1,-5,5, label = r"Radion $\eta$").Double()
		h_RadionCharge = hist.Hist.new.Regular(10,-5,5,label = r"Radion Electric Charge").Double()
		h_LeadHiggsCharge = hist.Hist.new.Regular(8,-4,4,label = r"Leading Higgs Electric Charge").Double()
		h_SubleadHiggsCharge = hist.Hist.new.Regular(8,-4,4,label = r"Subleading Higgs Electric Charge").Double()
		h_NElec = hist.Hist.new.Regular(8,0,8,label = r"number of electrons").Double()
		h_NMuon = hist.Hist.new.Regular(8,0,8,label = r"number of muons").Double()
		h_MinTauEleDeltaR = hist.Hist.new.Regular(N1,0,1,label = r"Minimized tau to electron $\Delta$R").Double()
		h_MinTauMuonDeltaR = hist.Hist.new.Regular(N1,0,1,label = r"Minimized tau to muon $\Delta$R").Double()
		h_NEleTauID = hist.Hist.new.Regular(5,0,5,label=r"Number of electrons identified as taus").Double()
		h_NMuonTauID= hist.Hist.new.Regular(5,0,5,label=r"Number of muon identified as taus").Double()
		h_weight = hist.Hist.new.Regular(10,-2,2,label=r"Applied Event Weight").Double()

		#Fill histograms
		#if (): #See what the weights and CrossSec_Weight is for the background
		h_FourTauMass.fill(ak.ravel(FourTau_Mass_Arr),weight = event_level.event_weight*CrossSec_Weight)
		h_HiggsDeltaPhi.fill(ak.ravel(Higgs_DeltaPhi_Arr),weight = event_level.event_weight*CrossSec_Weight)
		h_LeadHiggsMass.fill(ak.ravel(LeadingHiggs_mass_Arr),weight = event_level.event_weight*CrossSec_Weight)
		h_SubleadingHiggsMass.fill(ak.ravel(SubLeadingHiggs_mass_Arr),weight = event_level.event_weight*CrossSec_Weight)
		h_HiggsDeltaR.fill(ak.ravel(diHiggs_dR_Arr),weight = event_level.event_weight*CrossSec_Weight)
		h_LeadDiTauDeltaR.fill(ak.ravel(leading_dR_Arr),weight = event_level.event_weight*CrossSec_Weight)
		h_SubLeadingDiTauDeltaR.fill(ak.ravel(subleading_dR_Arr),weight = event_level.event_weight*CrossSec_Weight)
		h_LeadHiggsCharge.fill(ak.ravel(event_level.LeadingHiggs_Charge),weight = event_level.event_weight*CrossSec_Weight)
		h_SubleadHiggsCharge.fill(ak.ravel(event_level.SubleadingHiggs_Charge),weight = event_level.event_weight*CrossSec_Weight)
		
		h_LeadDiTauPhi.fill(ak.ravel(leading_dPhi_Arr),weight = event_level.event_weight*CrossSec_Weight)
		h_SubleadDiTauPhi.fill(ak.ravel(subleading_dPhi_Arr),weight = event_level.event_weight*CrossSec_Weight)
		h_RadMETPhi.fill(ak.ravel(radionMET_dPhi),weight = event_level.event_weight*CrossSec_Weight)
		h_RadLeadHiggsDeltaR.fill(ak.ravel(leadingHiggs_Rad_dR),weight = event_level.event_weight*CrossSec_Weight)
		h_RadSubleadingHiggsDeltaR.fill(ak.ravel(subleadingHiggs_Rad_dR),weight = event_level.event_weight*CrossSec_Weight)
		h_METLeadHiggsDeltaPhi.fill(ak.ravel(leadingHiggs_MET_dPhi_Arr),weight = event_level.event_weight*CrossSec_Weight)
		h_METSubleadHiggsDeltaPhi.fill(ak.ravel(subleadingHiggs_MET_dPhi_Arr),weight = event_level.event_weight*CrossSec_Weight)
		h_RadionEta.fill(ak.ravel(Radion_Reco.eta),weight = event_level.event_weight*CrossSec_Weight)
		h_RadionCharge.fill(ak.ravel(event_level.Radion_Charge),weight = event_level.event_weight*CrossSec_Weight)

		h_RadionpT.fill(ak.ravel(radionPT_Arr),weight = event_level.event_weight*CrossSec_Weight)
		#h_taupT.fill(ak.ravel(tau.pt),weight = event_level.event_weight*CrossSec_Weight)
		h_LeadTaupT.fill(ak.ravel(tau[ak.argsort(tau.pt,axis=-1)][:,3].pt),weight = event_level.event_weight*CrossSec_Weight)
		h_SubleadTaupT.fill(ak.ravel(tau[ak.argsort(tau.pt,axis=-1)][:,2].pt),weight = event_level.event_weight*CrossSec_Weight)
		h_3TaupT.fill(ak.ravel(tau[ak.argsort(tau.pt,axis=-1)][:,1].pt),weight = event_level.event_weight*CrossSec_Weight)
		h_4TaupT.fill(ak.ravel(tau[ak.argsort(tau.pt,axis=-1)][:,0].pt),weight = event_level.event_weight*CrossSec_Weight)
		#h_taupT = h_LeadTaupT + h_SubleadTaupT + h_3TaupT + h_4TaupT
		#h_tauEta.fill(ak.ravel(tau.eta),weight = event_level.event_weight*CrossSec_Weight)
		h_ZMult.fill(ak.ravel(event_level.ZMult),weight = event_level.event_weight*CrossSec_Weight)
		h_ZMultEle.fill(ak.ravel(event_level.ZMult_e),weight = event_level.event_weight*CrossSec_Weight)
		h_ZMultMu.fill(ak.ravel(event_level.ZMult_mu),weight = event_level.event_weight*CrossSec_Weight)
		h_BJetMult.fill(ak.ravel(event_level.nBJets),weight = event_level.event_weight*CrossSec_Weight)
		h_NElec.fill(ak.ravel(event_level.n_electrons),weight = event_level.event_weight*CrossSec_Weight)
		h_NMuon.fill(ak.ravel(event_level.n_muons),weight = event_level.event_weight*CrossSec_Weight)
		#h_MinTauEleDeltaR.fill(ak.ravel(ak.where(ak.num(electron.tau_min_dR,axis=1)!= 0, electron.tau_min_dR, ak.singletons(np.ones(ak.num(electron.tau_min_dR,axis=0))*999))),weight = event_level.event_weight*CrossSec_Weight)
		#h_MinTauMuonDeltaR.fill(ak.ravel(ak.where(ak.num(muon.tau_min_dR,axis=1) != 0, muon.tau_min_dR, ak.singletons(np.ones(ak.num(muon.tau_min_dR,axis=0))*999))),weight = event_level.event_weight*CrossSec_Weight)
		h_NEleTauID.fill(ak.ravel(event_level.n_tau_electrons),weight = event_level.event_weight*CrossSec_Weight)
		h_NMuonTauID.fill(ak.ravel(event_level.n_tau_muons),weight = event_level.event_weight*CrossSec_Weight)
		h_weight.fill(ak.ravel(event_level.event_weight*CrossSec_Weight))

		return{
			dataset: {
				#"Weight": CrossSec_Weight,
				"Weight_Val": CrossSec_Weight,
				"Weight": ak.to_list(event_level.event_weight*CrossSec_Weight), 
				"FourTau_Mass_Arr": h_FourTauMass,
				"HiggsDeltaPhi_Arr": h_HiggsDeltaPhi,
				"LeadingHiggs_mass": h_LeadHiggsMass,
				"SubLeadingHiggs_mass": h_SubleadingHiggsMass,
				"Higgs_DeltaR_Arr": h_HiggsDeltaR,
				"leading_dR_Arr": h_LeadDiTauDeltaR,
				"subleading_dR_Arr": h_SubLeadingDiTauDeltaR,
				"LeadingHiggsSgn_Arr" : h_LeadHiggsCharge,
				"SubleadingHiggsSgn_Arr" : h_SubleadHiggsCharge,
				
				"leading_dPhi_Arr": h_LeadDiTauPhi,
				"subleading_dPhi_Arr": h_SubleadDiTauPhi,
				"radionMET_dPhi_Arr": h_RadMETPhi,
				"leadingHiggs_Rad_dR_Arr": h_RadLeadHiggsDeltaR,
				"subleadingHiggs_Rad_dR_Arr": h_RadSubleadingHiggsDeltaR,
				"leadingHiggs_MET_dPhi_Arr": h_METLeadHiggsDeltaPhi,
				"subleadingHiggs_MET_dPhi_Arr": h_METSubleadHiggsDeltaPhi,
				"Radion_eta_Arr": h_RadionEta,
				"Radion_Charge_Arr": h_RadionCharge,
				
				"radionPT_Arr" : h_RadionpT,
				#"tau_pt_Arr": h_taupT,
				"tau_lead_pt_Arr":h_LeadTaupT ,
				"tau_sublead_pt_Arr":h_SubleadTaupT ,
				"tau_3rdlead_pt_Arr": h_3TaupT,
				"tau_4thlead_pt_Arr": h_4TaupT,
				#"tau_eta_Arr": h_tauEta,
				"ZMult_Arr": h_ZMult,
				"ZMult_ele_Arr": h_ZMultEle,
				"ZMult_mu_Arr": h_ZMultMu,
				"BJet_Arr": h_BJetMult,
				"Lumi_Val": Lumi_2018,
				"CrossSec_Val": crossSecVal,
				"NEvent_Val": numEvents_Dict[dataset],
				"Num_Electrons_Arr": h_NElec,
				"Num_Muons_Arr": h_NMuon,
				#"Electron_tau_dR_Arr": h_MinTauEleDeltaR,
				#"Muon_tau_dR_Arr": h_MinTauMuonDeltaR, #muon.tau_min_dR
                "num_electron_tau_Arr": h_NEleTauID,
                "num_muon_tau_Arr": h_NMuonTauID,
				"cutflow_table": cutflow_table,
				"num_events": ak.num(event_level.event_weight,axis=0),
				"weight_Hist": h_weight
			}
		}
	
	def postprocess(self, accumulator):
		pass	

if __name__ == "__main__":
	#mass_str_arr = ["1000","2000","3000"]
	mass_str_arr = ["2000"]
	
	#Functions and variables for Luminosity weights
	lumi_table_data = {"MC Sample":[], "Luminosity":[], "Cross Section (pb)":[], "Number of Events":[], "Calculated Weight":[]}

	#Set up dictionary of all possible final states (in the least efficient way but I just don't care anymore)
	template_array_1 = []
	template_array_2 = []
	template_array_3 = []
	for i in range(5):
		for j in range(5):
			for k in range(5):
				if (i + j + k == 4):
					template_array_1.append(i) #electron
					template_array_2.append(j) #muon
					template_array_3.append(k) #Hadron

	final_state_dict_signal = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),0)
	final_state_dict_data = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),0)
	final_state_dict_background = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),0)
	final_state_dict_signal_error = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),0)
	final_state_dict_data_error = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),0)
	final_state_dict_background_error = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),0)
	final_state_dict_data_full = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),[])
	#final_state_dict_signal_full = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),[])
	#final_state_dict_background_full = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),[])
	#final_state_dict_theory = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),branch_ratio_vec(template_array_3,template_array_2,template_array_1))
	final_state_dict_theory = dict(zip(fin_state_vec(template_array_1,template_array_2),branch_ratio_vec(template_array_3,template_array_2,template_array_1)))
	background_state_array = []

	
	#Trigger dictionaries
	#trigger_dict = {"Mu50": (21,False), "PFMET120_PFMHT120_IDTight": (27,False), "EitherOr_Trigger": (41,True)}
	#trigger_dict = {"Mu50": (21,False), "PFMET120_PFMHT120_IDTight": (27,False), "EitherOr_Trigger": (41,True)}
	trigger_dict = {"EitherOr_Trigger": (41,True)}
	#trigger_dict = {"Mu50": (21,False)}
	#trigger_dict = {"PFHT500_PFMET100_PFMHT100_IDTight": (39,False)} #,"EitherOr_Trigger": (41,True)}
	#trigger_dict = {"Mu50": (21,False), "EitherOr_Trigger": (41,True)}
	#trigger_dict = {"Mu50": (21,False), "PFHT500_PFMET100_PFMHT100_IDTight": (39,False), "EitherOr_Trigger": (41,True)}
	#trigger_dict = {"Mu50": (21,False), "PFHT500_PFMET100_PFMHT100_IDTight": (39,False)} #,"EitherOr_Trigger": (41,True)}
	#trigger_dict = {"No_Trigger": (0,False)}
	#trigger_dict = {"PFHT500_PFMET100_PFMHT100_IDTight": (39,False), "AK8PFJet400_TrimMass30": (40,False), "EitherOr_Trigger": (41,True)}
	
	#Locations of files
	signal_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/GluGluToRadionToHHTo4T_M-"
	#background_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"	
	#background_base = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/" #ZZTo4L_25February25_0413_skim__skim_Feb25/ #NanoAOD files
	#background_base = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/"
	#background_base = "root://cmsxrootd.hep.wisc.edu:1094//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/" #ZZTo4L_25February25_0413_skim__skim_Feb25/ #NanoAOD files
	background_base = "root://cmsxrootd.hep.wisc.edu//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/" #ZZTo4L_25February25_0413_skim__skim_Feb25/ #NanoAOD files
	background_loc = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/" #ZZTo4L_25February25_0413_skim__skim_Feb25/ #NanoAOD files
	#background_base = "" #For testing nanoAOD just dumped one ZZ4l root file into here, not scalable though 
	#data_loc = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"
	
	#signal_base = "hdfs/store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/GluGluToRadionToHHTo4T_M-"
	#background_base = "hdfs/store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"	
	#data_loc = "hdfs/store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"
	
	#signal_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2/GluGluToRadionToHHTo4T_M-"
	#background_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2/"	
	#data_loc = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2/" (miniAOD)
	data_loc = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/Data/"
	data_base = "root://cmsxrootd.hep.wisc.edu//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/Data/" 

	#Xrootd crap
	_x509_path = move_X509()
	print(f"x509 path: {_x509_path}")
	#Condor related stuff
	os.environ["CONDOR_CONFIG"] = "/etc/condor/condor_config"
	#htc_log_err_dir = "/scratch/twnelson/ControlPlot_HTC/Run_" + str(time.localtime()[0]) + "_" + str(time.localtime()[1]) + "_" + str(time.localtime()[2]) + "_" + str(time.localtime()[3]) + f".{time.localtime()[4]:02d}"
	#os.makedirs(htc_log_err_dir)

    #DO NOT DELETE THESE LINES CONDOR BROKEN???
	cluster = ""
#	cluster = HTCondorCluster(
#            cores=1,
#			 memory="6 GB",
#            disk="3 GB",
#            death_timeout = '60',
#            job_extra_directives={
#                "+JobFlavour": '"tomorrow"',
#                "log": "dask_job_output.$(PROCESS).$(CLUSTER).log",
#                "output": "dask_job_output.$(PROCESS).$(CLUSTER).out",
#                "error": "dask_job_output.$(PROCESS).$(CLUSTER).err",
#                "should_transfer_files": "yes",
#                "when_to_transfer_ouput": "ON_EXIT_OR_EVICT",
#                #"transfer_output_remaps": working_dir,
#				#"transfer_output_files": os.getcwd(), #Dump parquet files in current directory
#                "transfer_executable": "false",
#                "+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest-py3.10"',
#                #"+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.25-py3.10"',
#                "Requirements": "HasSingularityJobStart",
#                "InitialDir": f'/scratch/{os.environ["USER"]}',
#                'transfer_input_files': f"{_x509_path}",
#
#            },
#            job_script_prologue = [
#                "export XRD_RUNFORKHANDLER=1",
#                f"export X509_USER_PROXY={_x509_path}"
#            ]
#    )
#	cluster.adapt(minimum=1, maximum=500)

	run_on_condor = False
	
	if (run_on_condor):
		print("Run on Condor")
		iterative_runner = processor.Runner(
			#executor = processor.DaskExecutor(client=Client(cluster)),
			executor = processor.DaskExecutor(client=Client(cluster),status=False),
			schema=BaseSchema,
			skipbadfiles=True,
			xrootdtimeout=1000,
			#executor = processor.DaskExecutor(client=cowtools.GetCondorClient(container_image="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.25-py3.10")),
			#executor = processor.DaskExecutor(client=cowtools.GetCondorClient(cluster,container_image="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.25-py3.10")),
		#executor = processor.IterativeExecutor(compression=None), #This needs to be changed
		)
	else:
		#iterative_runner = processor.Runner(executor = processor.FuturesExecutor(), schema=BaseSchema)
		iterative_runner = processor.Runner(executor = processor.IterativeExecutor(), schema=BaseSchema)
	#four_tau_hist_list = ["FourTau_Mass_Arr","HiggsDeltaPhi_Arr", "Higgs_DeltaR_Arr","leading_dR_Arr","subleading_dR_Arr","LeadingHiggs_mass","SubLeadingHiggs_mass", "radionPT_Arr", "tau_pt_Arr", 
	#		"tau_eta_Arr","ZMult_Arr", "BJet_Arr", "tau_lead_pt_Arr", "tau_sublead_pt_Arr", "tau_3rdlead_pt_Arr", "tau_4thlead_pt_Arr", "leading_dPhi_Arr", "subleading_dPhi_Arr", 
	#		"radionMET_dPhi_Arr","leadingHiggs_Rad_dR_Arr","subleadingHiggs_Rad_dR_Arr","leadingHiggs_MET_dPhi_Arr","subleadingHiggs_MET_dPhi_Arr","Radion_eta_Arr", "Radion_Charge_Arr"]
	four_tau_hist_list = ["FourTau_Mass_Arr","HiggsDeltaPhi_Arr", "Higgs_DeltaR_Arr","leading_dR_Arr","subleading_dR_Arr","LeadingHiggs_mass","SubLeadingHiggs_mass", "radionPT_Arr", 
			"ZMult_Arr", "BJet_Arr", "tau_lead_pt_Arr", "tau_sublead_pt_Arr", "tau_3rdlead_pt_Arr", "tau_4thlead_pt_Arr", "leading_dPhi_Arr", "subleading_dPhi_Arr", 
			"radionMET_dPhi_Arr","leadingHiggs_Rad_dR_Arr","subleadingHiggs_Rad_dR_Arr","leadingHiggs_MET_dPhi_Arr","subleadingHiggs_MET_dPhi_Arr","Radion_eta_Arr", "Radion_Charge_Arr",
			"LeadingHiggsSgn_Arr", "SubleadingHiggsSgn_Arr","Num_Electrons_Arr","Num_Muons_Arr","cutflow_table"] #,"num_electron_tau_Arr","num_muon_tau_Arr"] #,"Electron_tau_dR_Arr","Muon_tau_dR_Arr"] (Removed num_electron_tau and num_muon_tau for now)
	#four_tau_hist_list = ["Num_Electrons_Arr","Num_Muons_Arr","Electron_tau_dR_Arr","Muon_tau_dR_Arr"]
	#four_tau_hist_list = ["ZMult_Arr"] #,"ZMult_ele_Arr","ZMult_mu_Arr", "ZMult_tau_Arr"]
	#four_tau_hist_list = ["leading_dR_Arr"] #Only make 1 histogram for brevity/debugging purposes
	hist_name_dict = {"FourTau_Mass_Arr": r"Reconstructed 4-$\tau$ invariant mass", "HiggsDeltaPhi_Arr": r"Reconstructed Higgs $\Delta \phi$", "Higgs_DeltaR_Arr": r"Reconstructed Higgs $\Delta R$",
					"leading_dR_Arr": r"$\Delta R$ of leading di-$\tau$ pair", "subleading_dR_Arr": r"$\Delta R$ of subleading di-$\tau$ pair", 
					"LeadingHiggs_mass": r"Leading di-$\tau$ pair invariant mass", "SubLeadingHiggs_mass": r"Subleading di-$\tau$ pair invariant mass", "radionPT_Arr": r"Reconstructed Radion $p_T$",
					"tau_pt_Arr": r"$\tau$ $p_T$", "tau_eta_Arr": r"$\tau \ \eta$", "ZMult_Arr": r"Z Boson Multiplicity", "ZMult_mu_Arr": r"Z Boson Multiplicity (muons only)", 
                    "ZMult_ele_Arr": r"Z Boson Multiplicity (electrons only)", "ZMult_tau_Arr" : r"Z Boson Multiplicity (from taus)", "BJet_Arr": r"B-Jet Multiplicity", 
                    "tau_lead_pt_Arr": r"Leadng $\tau$ $p_T$", "tau_sublead_pt_Arr": r"Subleading $\tau$ $p_T$", "tau_3rdlead_pt_Arr": r"Third leading $\tau$ $p_T$", 
					"tau_4thlead_pt_Arr": r"Fourth leading $\tau$ $p_T$", "leading_dPhi_Arr": r"Leading di-$\tau$ $\Delta \phi$", "subleading_dPhi_Arr": r"Subleading di-$\tau$ $\Delta \phi$",
					"radionMET_dPhi_Arr": r"Radion MET $\Delta \phi$", "leadingHiggs_Rad_dR_Arr": r"Leading Higgs Radion $\Delta$R", 
					"subleadingHiggs_Rad_dR_Arr": r"Subleading Higgs Radion $\Delta$R", "leadingHiggs_MET_dPhi_Arr": r"Leading Higgs MET $\Delta \phi$", 
					"subleadingHiggs_MET_dPhi_Arr": r"Subleading Higgs MET $\Delta \phi$", "Radion_eta_Arr": r"Radion $\eta$", "Radion_Charge_Arr": r"Radion Charge",
					"Num_Electrons_Arr": r"Number of $e$","Num_Muons_Arr": r"Number of $\mu$","Electron_tau_dR_Arr" : r"Electron $\tau$ minimized $\Delta$R",
					"Muon_tau_dR_Arr": r"$\mu$ $\tau$ minimized $\Delta$R","num_electron_tau_Arr": r"Number of e reconstructed as $\tau$","num_muon_tau_Arr": r"Number of $\mu$ reconstructed as $\tau$"}
    #four_tau_hist_list = ["HiggsDeltaPhi_Arr","Pair_DeltaPhi_Hist"]

	#Get PU Weighting information
	PUWeight = np.array([])
	with uproot.open("pu_distributions_mc_2018.root") as f1:
		with uproot.open("pu_distributions_data_2018.root") as f2:
			mc = f1["pileup"].values()
			data = f2["pileup"].values()
			HistoPUMC = np.divide(mc, ak.sum(mc))
			HistoPUData = np.divide(data, ak.sum(data))
			PUWeight = np.divide(HistoPUData, HistoPUMC) #	
			#PUWeight = np.ones(len(mc))

	#Look at the PU Weight values obtained
	#print("==========!!PU Weight debugging!!==========")
	#for pu in PUWeight:
	#	print(pu)

	#Loop over all mass points
	for mass in mass_str_arr:
		print("====================Radion Mass = " + mass[0] + "." + mass[1] + " TeV====================")
		#print(np.char.replace(np.array( os.listdir(background_loc + "ZZTo4L_25February25_0413_skim__skim_Feb25/")), "", background_loc + "ZZTo4L_25February25_0413_skim__skim_Feb25/",1).tolist())
		file_dict_test = { #Reduced files to run over
			#"ZZ4l": [background_base + "ZZTo4L_26August25_0757_skim_Newskim/ZZTo4L.root"], 
			"TTToSemiLeptonic": [background_base + "TTToSemiLeptonic_35August25_0448_skim_Newskim/TTToSemiLeptonic" + str(j) + ".root" for j in range(10)], 
			"TTTo2L2Nu": [background_base + "TTTo2L2Nu_26August25_0719_skim_Newskim/TTTo2L2Nu.root"], 
			"TTToHadronic": [background_base + "TTToHadronic_35August25_0419_skim_Newskim/TTToHadronic" + str(j) + ".root" for j in range(10)],
			"Data_SingleMuon": [data_base + "SingleMu_Run2018A_27August25_0551_skim_Newskim/SingleMu_Run2018A.root"], 
				#data_base + "SingleMu_Run2018B_27August25_0529_skim_Newskim/SingleMu_Run2018B.root", data_base + "SingleMu_Run2018C_27August25_0540_skim_Newskim/SingleMu_Run2018C.root", 
				#data_base + "SingleMu_Run2018D_27August25_0613_skim_Newskim/SingleMu_Run2018D.root"],
			"Data_JetHT": [data_base + "JetHT_2018_27August25_0655_skim_Newskim/JetHT_2018.root"]#, data_base + "JetHT_Other_2018_27August25_0522_skim_Newskim/JetHT_Other_2018.root"]
        }

		#file_dict["ZZ4l"].remove("root://cms-xrd-global.cern.ch//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/Hadd_ZZTo4L/ZZTo4L_Hadd_9.root") #Remove file 9 to fix errors (maybe?)
		file_dict_debug = {"WJetsToLNu_HT-1200To2500":[background_base + "WJetsToLNu_HT-1200To2500_OtherPart_28February25_1012_skim__skim_Feb25/singleFileSkimForSubmission-NANO_NANO_80.root"]}
		file_dict_signal_only = {
		#file_dict = {
			"Signal": [signal_base + mass + ".root"]
		}
		
		#Grand Unified Background + Signal + Data Dictionary links file name to location of root file
		file_dict = {
			"TTToSemiLeptonic": [background_base + "TTToSemiLeptonic_35August25_0448_skim_Newskim/TTToSemiLeptonic" + str(j) + ".root" for j in range(10)], 
			"TTTo2L2Nu": [background_base + "TTTo2L2Nu_26August25_0719_skim_Newskim/TTTo2L2Nu.root"], 
			"TTToHadronic": [background_base + "TTToHadronic_35August25_0419_skim_Newskim/TTToHadronic" + str(j) + ".root" for j in range(10)],
			"ZZ4l": [background_base + "ZZTo4L_26August25_0757_skim_Newskim/ZZTo4L.root"], 
			"VV2l2nu": [background_base + "WWTo2L2Nu_26August25_1040_skim_Newskim/WWTo2L2Nu.root"], 
			"WZ1l3nu": [background_base + "WZTo1L3Nu_4f_26August25_1016_skim_Newskim/WZTo1L3Nu_4f.root"], 
			"WZ3l1nu": [background_base + "WZTo3L1Nu_4f_26August25_1032_skim_Newskim/WZTo3L1Nu_4f.root"],  
			"ZZ2l2q": [background_base + "ZZTo2Q2L_26August25_1034_skim_Newskim/ZZTo2Q2L.root"],
			"WZ2l2q": [background_base + "WZTo2L2Q_26August25_0926_skim_Newskim/WZTo2L2Q.root"],
			"WZ1l1nu2q" : [background_base + "WZTo1L1Nu2Q_26August25_0840_skim_Newskim/WZTo1L1Nu2Q.root"],
			"DYJetsToLL_Pt-50To100": [background_base + "DYJetsToLL_LHEFilterPtZ-50To100_MatchEWPDG20_26August25_1018_skim_Newskim/DYJetsToLL_LHEFilterPtZ-50To100_MatchEWPDG20.root"],
			"DYJetsToLL_Pt-100To250": [background_base + "DYJetsToLL_LHEFilterPtZ-100To250_MatchEWPDG20_26August25_0917_skim_Newskim/DYJetsToLL_LHEFilterPtZ-100To250_MatchEWPDG20.root"], 
			"DYJetsToLL_Pt-250To400": [background_base + "DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20_26August25_0748_skim_Newskim/DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20.root"], 
			"DYJetsToLL_Pt-400To650": [background_base + "DYJetsToLL_LHEFilterPtZ-400To650_MatchEWPDG20_26August25_1042_skim_Newskim/DYJetsToLL_LHEFilterPtZ-400To650_MatchEWPDG20.root"], 
			"DYJetsToLL_Pt-650ToInf": [background_base + "DYJetsToLL_LHEFilterPtZ-650ToInf_MatchEWPDG20_26August25_0842_skim_Newskim/DYJetsToLL_LHEFilterPtZ-650ToInf_MatchEWPDG20.root"],
			"T-tchan": [background_base + "ST_t-channel_top_4f_InclusiveDecays_26August25_0843_skim_Newskim/ST_t-channel_top_4f_InclusiveDecays.root"], 
			"Tbar-tchan": [background_base + "ST_t-channel_antitop_4f_InclusiveDecays_26August25_0821_skim_Newskim/ST_t-channel_antitop_4f_InclusiveDecays.root"], 
			"T-tW": [background_base + "ST_tW_top_5f_inclusiveDecays_26August25_0753_skim_Newskim/ST_tW_top_5f_inclusiveDecays.root"], 
			"Tbar-tW": [background_base + "ST_tW_antitop_5f_inclusiveDecays_26August25_1030_skim_Newskim/ST_tW_antitop_5f_inclusiveDecays.root"],
			"WJetsToLNu_HT-100To200": [background_base + "WJetsToLNu_HT-100To200_26August25_0810_skim_Newskim/WJetsToLNu_HT-100To200.root"],
			"WJetsToLNu_HT-200To400": [background_base + "WJetsToLNu_HT-200To400_26August25_0709_skim_Newskim/WJetsToLNu_HT-200To400.root"], 
			"WJetsToLNu_HT-400To600": [background_base + "WJetsToLNu_HT-400To600_26August25_1014_skim_Newskim/WJetsToLNu_HT-400To600.root", background_base +"WJetsToLNu_HT-400To600_OtherPart_26August25_1032_skim_Newskim/WJetsToLNu_HT-400To600_OtherPart.root"], 
			"WJetsToLNu_HT-600To800": [background_base + "WJetsToLNu_HT-600To800_26August25_0755_skim_Newskim/WJetsToLNu_HT-600To800.root", background_base + "WJetsToLNu_HT-600To800_OtherPart_26August25_0752_skim_Newskim/WJetsToLNu_HT-600To800_OtherPart.root"],
			"WJetsToLNu_HT-800To1200": [background_base + "WJetsToLNu_HT-800To1200_26August25_0708_skim_Newskim/WJetsToLNu_HT-800To1200.root", background_base + "WJetsToLNu_HT-800To1200_OtherPart_26August25_0925_skim_Newskim/WJetsToLNu_HT-800To1200_OtherPart.root"],
			"WJetsToLNu_HT-1200To2500": [background_base + "WJetsToLNu_HT-1200To2500_26August25_1016_skim_Newskim/WJetsToLNu_HT-1200To2500.root", background_base + "WJetsToLNu_HT-1200To2500_OtherPart_26August25_1041_skim_Newskim/WJetsToLNu_HT-1200To2500_OtherPart.root"],
			"WJetsToLNu_HT-2500ToInf": [background_base + "WJetsToLNu_HT-2500ToInf_26August25_1047_skim_Newskim/WJetsToLNu_HT-2500ToInf.root", background_base + "WJetsToLNu_HT-2500ToInf_OtherPart_26August25_1043_skim_Newskim/WJetsToLNu_HT-2500ToInf_OtherPart.root"],
			#"Signal": [signal_base + mass + ".root"],
			"Data_SingleMuon": [data_base + "SingleMu_Run2018A_27August25_0551_skim_Newskim/SingleMu_Run2018A.root", data_base + "SingleMu_Run2018B_27August25_0529_skim_Newskim/SingleMu_Run2018B.root", 
                                data_base + "SingleMu_Run2018C_27August25_0540_skim_Newskim/SingleMu_Run2018C.root", data_base + "SingleMu_Run2018D_27August25_0613_skim_Newskim/SingleMu_Run2018D.root"],
			"Data_JetHT": [data_base + "JetHT_2018_27August25_0655_skim_Newskim/JetHT_2018.root", data_base + "JetHT_Other_2018_27August25_0522_skim_Newskim/JetHT_Other_2018.root"]
			#"Data_JetHT": [data_loc + "JetHT_Run2018A-17Sep2018-v1.root", data_loc + "JetHT_Run2018B-17Sep2018-v1.root", data_loc + "JetHT_Run2018C-17Sep2018-v1.root",data_loc + "JetHT_Run2018D-PromptReco-v2.root"]
		}

		#Generate dictionary of number of processed events This logic needs fixing
		#print("About to obtain number of events being proscessed")
		sumEvents_Dict = {}
		for key_name, file_array in file_dict.items(): 
			print(key_name)
			if (key_name != "Data_JetHT" and key_name != "Data_SingleMuon"): #This logic needs to be fixed
				numEvents_Dict[key_name] = 0 #Initialize the number of events dictionary
				sumEvents_Dict[key_name] = 0 #Initialize the number of events dictionary
				#print("Background:")
				#print(key_name)
				#print(file_array)
				for file in file_array:
					with uproot.open(file) as tempFile:
						print(file)
						#print("Current number of events: " + str(numEvents_Dict[key_name]))
						#print("Number of events being added: " + str(tempFile['Runs/genEventCount'].array()[0]))
						#numEvents_Dict[key_name] += np.sum(tempFile['Runs/genEventCount'].array()) #Fixed for nanoAOD (!!This line may cause issues!!)
						numEvents_Dict[key_name] += np.sum(tempFile['Runs/genEventSumw'].array()) #Fixed for nanoAOD (!!This line may cause issues!!)
					#numEvents_Dict[key_name] = tempFile['hEvents'].member('fEntries')/2
					#numEvents_Dict[key_name] = tempFile['hcount'].member('fEntries')/2 #This is only good for miniAOD

			else: #Ignore data files
				numEvents_Dict[key_name] = 1

		#break	
		background_list = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"]
		#background_list = [r"$ZZ \rightarrow 4l$"]
		#background_list = [r"$t\bar{t}$"]
		#background_list = [r"$t\bar{t}$", r"$ZZ \rightarrow 4l$"]
		signal_list = [r"MC Sample $m_\phi$ = %s TeV"%mass[0]]
		background_plot_names = {r"$t\bar{t}$" : "_ttbar_", r"Drell-Yan+Jets": "_DYJets_", "Di-Bosons" : "_DiBosons_", "Single Top": "_SingleTop_", "QCD" : "_QCD_", "W+Jets" : "_WJets_", r"$ZZ \rightarrow 4l$" : "_ZZ4l_"} #For file names
		
		background_dict = {r"$t\bar{t}$" : ["TTToSemiLeptonic","TTTo2L2Nu","TTToHadronic"], 
				r"Drell-Yan+Jets": ["DYJetsToLL_Pt-50To100","DYJetsToLL_Pt-100To250","DYJetsToLL_Pt-250To400","DYJetsToLL_Pt-400To650","DYJetsToLL_Pt-650ToInf"], 
				"Di-Bosons": ["WZ3l1nu","WZ2l2q","WZ1l1nu2q","ZZ2l2q", "WZ1l3nu", "VV2l2nu"], "Single Top": ["Tbar-tchan","T-tchan","Tbar-tW","T-tW"], 
				"W+Jets": ["WJetsToLNu_HT-100To200","WJetsToLNu_HT-200To400","WJetsToLNu_HT-400To600","WJetsToLNu_HT-600To800","WJetsToLNu_HT-800To1200","WJetsToLNu_HT-1200To2500","WJetsToLNu_HT-2500ToInf"],
				r"$ZZ \rightarrow 4l$" : ["ZZ4l"]
		}
		

		for trigger_name, trigger_pair in trigger_dict.items(): #Run over all triggers/combinations of interest
			#Dictionaries of histograms for background, signal and data
			hist_dict_background = dict.fromkeys(four_tau_hist_list)
			hist_dict_signal = dict.fromkeys(four_tau_hist_list)
			hist_dict_data = dict.fromkeys(four_tau_hist_list)
			
			#Dictinary with file names
			four_tau_names = {"FourTau_Mass_Arr": "FourTauMass_" + mass + "-" + trigger_name, "HiggsDeltaPhi_Arr": "HiggsDeltaPhi_" + mass + "-" + trigger_name, 
				"Pair_DeltaPhi_Hist": "TauPair_DeltaPhi_" + mass + "-" + trigger_name, "RadionPTComp_Hist": "pTReco_Comp_"+mass+ "-"+ trigger_name,
				"Higgs_DeltaR_Arr": "Higgs_DeltaR_" + mass + "-" + trigger_name, "leading_dR_Arr": "leading_diTau_DeltaR_" + mass + "-" + trigger_name, 
				"subleading_dR_Arr": "subleading_diTau_DeltaR_" + mass + "-" + trigger_name, "LeadingHiggs_mass" : "LeadingHiggs_Mass_"+ mass + "-" + trigger_name, 
				"SubLeadingHiggs_mass" : "SubLeadingHiggs_Mass_" + mass + "-" + trigger_name, "radionPT_Arr": "radion_pT_Mass_" + mass + "-" + trigger_name,
				"tau_pt_Arr": "TaupT_Mass_" + mass + "-" + trigger_name, "tau_eta_Arr": "Taueta_Mass_" + mass + "-" + trigger_name, "ZMult_Arr": "ZMult_Mass_" + mass + "-" + trigger_name, 
				"ZMult_ele_Arr": "ele_ZMult_Mass_" + mass + "-" + trigger_name, "ZMult_mu_Arr": "mu_ZMult_Mass_" + mass + "-" + trigger_name, 
				"ZMult_tau_Arr": "tau_ZMult_Mass_" + mass + "-" + trigger_name, "BJet_Arr": "BJetMult_Mass_" + mass + "-" + trigger_name, 
				"tau_lead_pt_Arr": r"LeadTau_pT_Mass_" + mass + "-" + trigger_name, "tau_sublead_pt_Arr": r"SubleadTau_pT_Mass_" + mass + "-" + trigger_name, 
				"tau_3rdlead_pt_Arr": r"ThirdleadTau_pT_Mass_" + mass + "-" + trigger_name,"tau_4thlead_pt_Arr": r"FourthleadTau_pT_Mass_" + mass + "-" + trigger_name,
				"leading_dPhi_Arr": "LeadingdiTau_DeltaPhi_Mass_" + mass + "-" + trigger_name, "subleading_dPhi_Arr": "SubleadingdiTau_DeltaPhi_Mass_" + mass + "-" + trigger_name,
				"radionMET_dPhi_Arr": "Radion_MET_DeltaPhi_Mass_" + mass + "-" + trigger_name, "leadingHiggs_Rad_dR_Arr": "LeadingHiggs_Radion_DeltaR_Mass_" + mass + "-" + trigger_name, 
				"subleadingHiggs_Rad_dR_Arr": "SubLeadingHiggs_Radion_DeltaR_Mass_" + mass + "-" + trigger_name, 
				"leadingHiggs_MET_dPhi_Arr": "LeadingHiggs_MET_DeltaPhi_Mass_" + mass + "-" + trigger_name, 
				"subleadingHiggs_MET_dPhi_Arr": "SubLeadingHiggs_MET_DeltaPhi_Mass_" + mass + "-" + trigger_name, "Radion_eta_Arr": "Radion_eta_Mass_" + mass + "-" + trigger_name,
				"Radion_Charge_Arr": "Radion_Charge_Mass" + mass + "-" + trigger_name,
				"LeadingHiggsSgn_Arr": "LeadingHiggs_Charge_Mass" + mass + "-" + trigger_name,
				"SubleadingHiggsSgn_Arr": "SubleadingHiggs_Charge_Mass" + mass + "-" + trigger_name,"Num_Electrons_Arr": "NumberOf_Electrons_Mass" + mass + "-" + trigger_name,
				"Num_Muons_Arr" : "NumberOf_Muons_Mass" + mass + "-" + trigger_name, "Electron_tau_dR_Arr": "min_dR_Tau_Electron_Mass" + mass + "-" + trigger_name,
				"Muon_tau_dR_Arr": "min_dR_Tau_Muon_Mass" + mass + "-" + trigger_name,
				"num_electron_tau_Arr": "Num_electrons_as_tau_Mass" + mass + "-" + trigger_name,
				"num_muon_tau_Arr": "Num_muons_as_tau_Mass" + mass + "-" + trigger_name,
				"cutflow_table": "Cutflow_Table_Mass" + mass + "-" + trigger_name
			}
			
			#fourtau_out = iterative_runner(file_dict, treename="Events", processor_instance=FourTauPlotting(trigger_bit=trigger_pair[0], or_trigger=trigger_pair[1],PUWeights = PUWeight, PU_weight_bool =True, signal_mass = mass)) #Modified for NanoAOD (changd treename)
			print("About to run iterative runner")
			fourtau_out = iterative_runner(file_dict, treename="Events", processor_instance=FourTauPlotting(trigger_bit=trigger_pair[0], or_trigger=trigger_pair[1],PUWeights = PUWeight, PU_weight_bool = False, signal_mass = mass)) #Modified for NanoAOD (changd treename)
			print("Ran iterative runner")
			for hist_name in four_tau_hist_list: #Loop over all histograms

				#back_hist_dict = {} #Dictionary of all histogram backgrounds for 
				#if (hist_name != "Pair_DeltaPhi_Hist" and hist_name != "RadionPTComp_Hist"):
				temp_hist_dict = dict.fromkeys(background_list) # create dictionary of histograms for each background type
						
				if (hist_name == "cutflow_table"): #Get the weight distributions from data
					fig_data, ax_data = plt.subplots()
					data_hist = fourtau_out["Data_SingleMuon"]["cutflow_table"]
					data_hist += fourtau_out["Data_JetHT"]["cutflow_table"]
					data_hist.plot1d(ax = ax_data)
					plt.title("Data Weights")
					plt.savefig("SingleBackground_Data_WeightTable")
					plt.close()
				
				for background_type in background_list:
					print("Background type %s"%background_type)
					background_array = []
					backgrounds = background_dict[background_type]
						
					#Loop over all backgrounds
					for background in backgrounds:
						print("%s"%background)
						if (mass == "2000"): #Only need to generate single background once
							
							#Plot the cutflow for each background
							if (hist_name == "cutflow_table"):
								#print(fourtau_out[background]["cutflow_table"].axes)
								if (background == backgrounds[0]):
									cutflow_hist = fourtau_out[background]["cutflow_table"]
								else:
									cutflow_hist += fourtau_out[background]["cutflow_table"]
								
								if (background == backgrounds[-1]):
									fig2p5, ax2p5 = plt.subplots()
									cutflow_hist.plot1d(ax=ax2p5)
									plt.title(background_type + " Cutflow Table")
									ax2p5.set_yscale('log')
									plt.savefig("SingleBackground" + background_plot_names[background_type] + "CutFlowTable")
									plt.close()
							
								#Plot the weights for each background
								if (background == backgrounds[0]):
									weight_hist = fourtau_out[background]["weight_Hist"]
								else:
									weight_hist += fourtau_out[background]["weight_Hist"]
								if (background == backgrounds[-1]):
									figweight, axweight = plt.subplots()
									weight_hist.plot1d(ax=axweight)
									plt.title(background_type + " Weight Histogram")
									plt.savefig("SingleBackground" + background_plot_names[background_type] + "Weight")
									plt.close()
							
							if (hist_name == "Radion_Charge_Arr"):
								lumi_table_data["MC Sample"].append(background)
								lumi_table_data["Luminosity"].append(fourtau_out[background]["Lumi_Val"])
								lumi_table_data["Cross Section (pb)"].append(fourtau_out[background]["CrossSec_Val"])
								lumi_table_data["Number of Events"].append(fourtau_out[background]["NEvent_Val"])
								lumi_table_data["Calculated Weight"].append(fourtau_out[background]["Weight_Val"])
							

							if (hist_name != "Electron_tau_dR_Arr" and hist_name != "Muon_tau_dR_Arr"):
								if (background == backgrounds[0]):
									crnt_hist = fourtau_out[background][hist_name]
									print("Background: " + background)
									print("Sum of entries: %f"%fourtau_out[background][hist_name].sum())
								else:
									crnt_hist += fourtau_out[background][hist_name]
									print("Background: " + background)
									print("Sum of entries: %f"%fourtau_out[background][hist_name].sum())
								if (background == backgrounds[-1]):
									fig2, ax2 = plt.subplots()
									temp_hist_dict[background_type] = crnt_hist #Try to fix stacking bug
									crnt_hist.plot1d(ax=ax2)
									#if (hist_name == "FourTau_Mass_Arr"):
									print("Background: " + background_type)
									print("Sum of entries: %f"%crnt_hist.sum())
									#print("Number of Entries: %d"%fourtau_out[background]["num_events"])
									plt.title(background_type)
									plt.savefig("SingleBackground" + background_plot_names[background_type] + four_tau_names[hist_name])
									plt.close()

							else: #lepton-tau delta R 
								fig2, ax2 = plt.subplots()
								fourtau_out[background][hist_name].plot1d(ax=ax2)
								ax2.set_yscale('log')
								plt.title(background_type)
								plt.savefig("SingleBackground" + background_plot_names[background_type] + four_tau_names[hist_name])
								plt.close()
						
						#Could there be issues here in terms of how the backgrounds are being combined???
					#	if (hist_name != "Electron_tau_dR_Arr"): # and hist_name != "Muon_tau_dR_Arr"): #Skip the lepton-tau delta R
					#		#print(fourtau_out[background]["Weight"])
					#		if (temp_hist_dict[background_type] == None): #Combine distirbutions of like background types together
					#			#temp_hist_dict[background_type] = fourtau_out[background][hist_name]
					#			print("First histogram added")
					#		else:
					#			#temp_hist_dict[background_type] += fourtau_out[background][hist_name]
					#			print("Additional Histogram added")

					#		#hist_dict_background[hist_name].fill(background_type,fourtau_out[background][hist_name],weight = fourtau_out[background]["Weight"]) #Obtain background distributions
					#		print("Background %s added"%background)
					#		print("Showing histogram:" + hist_name)
					#		#hist_dict_background[hist_name].show(background_type)
					#		#print()
					#		#center_arr = hist_dict_background[hist_name].axes.centers[0]
					#		#count_arr = hist_dict_background[hist_name].counts()
					#		#for n in range(hist_dict_background[hist_name].axes.size[0]):
					#		#	print(f"Bin #{n}, center at {center_arr[n]}: {count_arr[n]}")
					#	if (hist_name == "num_electron_tau_Arr"): # and np.pi == np.exp(1)): #Count final states
					#		background_state_array += fin_state_vec(fourtau_out[background]["num_electron_tau_Arr"],fourtau_out[background]["num_muon_tau_Arr"]).tolist()


				#Combine the backgrounds together
				hist_dict_background[hist_name] = hist.Stack.from_dict(temp_hist_dict) #This could be causing the problems 



							
					
				#	if (hist_name != "Electron_tau_dR_Arr" and hist_name != "Muon_tau_dR_Arr"): #Skip the lepton-tau delta R 
				#		hist_dict_signal[hist_name].fill("Signal",fourtau_out["Signal"][hist_name],weight = fourtau_out["Signal"]["Weight"]) #Obtain signal distribution
				#		hist_dict_signal[hist_name] = fourtau_out["Signal"][hist_name] #Obtain signal distribution
				#		if (hist_name == "num_electron_tau_Arr"): # and np.exp(1) == np.pi): # and np.pi == np.exp(1)): #Count final states
				#			print("Getting final states for signal MC")
				#			final_state_array = fin_state_vec(fourtau_out["Signal"]["num_electron_tau_Arr"],fourtau_out["Signal"]["num_muon_tau_Arr"])
				#			for state in final_state_array:
				#				final_state_dict_signal[state] += 1/len(fourtau_out["Signal"]["num_electron_tau_Arr"]) #Why is this getting me non sensical results??

				#			#Obtain uncertainties
				#			for state in final_state_dict_signal:
				#				final_state_dict_signal_error[state] = np.sqrt(final_state_dict_signal[state]*len(fourtau_out["Signal"]["num_electron_tau_Arr"]))
				#				final_state_dict_signal_error[state] /= len(fourtau_out["Signal"]["num_electron_tau_Arr"])
                        

					
				#Obtain data distributions
				print("==================Hist %s================"%hist_name)
				#print("Total amount of data = %d"%(len(fourtau_out["Data_SingleMuon"][hist_name]) + len(fourtau_out["Data_JetHT"][hist_name])))
				#print("Total amount of data = %d"%(len(fourtau_out["Data_SingleMuon"][hist_name])))
				#print("Total amount of data = %d"%(len(fourtau_out["Data_JetHT"][hist_name])))
				if (trigger_name == "Mu50"):
					print("Mu50 Only")
					hist_dict_data[hist_name] = fourtau_out["Data_SingleMuon"][hist_name] #.fill("Data",fourtau_out["Data_SingleMuon"][hist_name]) 
				if (trigger_name == "PFHT500_PFMET100_PFMHT100_IDTight"):
					print("JetHTMHTMET Only")
					hist_dict_data[hist_name] = fourtau_out["Data_JetHT"][hist_name]#.fill("Data",fourtau_out["Data_JetHT"][hist_name]) 
				if (trigger_name == "EitherOr_Trigger"):
					print("Both Triggers")
					hist_dict_data[hist_name] = fourtau_out["Data_SingleMuon"][hist_name]
					hist_dict_data[hist_name] += fourtau_out["Data_JetHT"][hist_name]
						
#							if (hist_name == "num_electron_tau_Arr"):  #and np.pi == np.exp(1)):
#							    print("Getting final states for data")
#							    final_state_array_Mu = fin_state_vec(fourtau_out["Data_SingleMuon"]["num_electron_tau_Arr"],fourtau_out["Data_SingleMuon"]["num_muon_tau_Arr"])
#							    final_state_array_Jet = fin_state_vec(fourtau_out["Data_JetHT"]["num_electron_tau_Arr"],fourtau_out["Data_JetHT"]["num_muon_tau_Arr"])
#							    for state in final_state_array_Mu:
#								    final_state_dict_data[state] += 1/(len(fourtau_out["Data_SingleMuon"]["num_electron_tau_Arr"]) + len(fourtau_out["Data_JetHT"]["num_electron_tau_Arr"]))
#							    for state in final_state_array_Jet:
#								    final_state_dict_data[state] += 1/(len(fourtau_out["Data_SingleMuon"]["num_electron_tau_Arr"]) + len(fourtau_out["Data_JetHT"]["num_electron_tau_Arr"]))
#
#								#Obtain uncertanties
#							    for state in final_state_dict_data:
#								    final_state_dict_data_error[state] = np.sqrt(final_state_dict_data[state]*(len(fourtau_out["Data_SingleMuon"]["num_electron_tau_Arr"]) + len(fourtau_out["Data_JetHT"]["num_electron_tau_Arr"])))
#								    final_state_dict_data_error[state] /= (len(fourtau_out["Data_SingleMuon"]["num_electron_tau_Arr"]) + len(fourtau_out["Data_JetHT"]["num_electron_tau_Arr"]))

					#print("Number of Jet HT entries: %d"%len(fourtau_out["Data_JetHT"][hist_name]))
								
				#Plot the weights for the data
				figweight_data, axweight_data = plt.subplots()
				weight_hist_data = fourtau_out["Data_SingleMuon"]["weight_Hist"]
				weight_hist_data += fourtau_out["Data_JetHT"]["weight_Hist"]

				weight_hist_data.plot1d(ax=axweight_data)
				plt.title("Data Weight Histogram")
				plt.savefig("SingleBackground_Data_Weight")
				plt.close()
			
				#Put histograms into stacks and arrays for plotting purposes (is the issue arising here??) (This logic may be outdated the .stack(name) may not be needed anymore)
				background_stack = hist_dict_background[hist_name] #hist_dict_background[hist_name].stack("background")
				#signal_stack = hist_dict_signal[hist_name].stack("signal")
				data_stack = hist_dict_data[hist_name] #.stack("data")    
				#signal_array = [signal_stack["Signal"]]
				data_array = [data_stack] #["Data"]]
				
				for background in background_list:
					background_array.append(background_stack[background]) #Is this line fucking up your scaling??
					#if (hist_name == "FourTau_Mass_Arr"):
					print("Background: " + background)
					print("Sum of stacked histogram: %f"%background_stack[background].sum())
			
				#Stack background distributions and plot signal + data distribution
				fig,ax = plt.subplots()
				hep.histplot(background_array,ax=ax,stack=True,histtype="fill",label=background_list,facecolor=TABLEAU_COLORS[:len(background_list)],edgecolor=TABLEAU_COLORS[:len(background_list)])
				#hep.histplot(signal_array,ax=ax,stack=True,histtype="step",label=signal_list,edgecolor=TABLEAU_COLORS[len(background_list)+1],linewidth=2.95)
				hep.histplot(data_array,ax=ax,stack=False,histtype="errorbar", yerr=True,label=["Data"],marker="o",color = "k") #,facecolor='black',edgecolor='black') #,mec='k')
				hep.cms.text("Preliminary",loc=0,fontsize=13)
				#ax.set_title(hist_name_dict[hist_name],loc = "right")
				ax.set_title("2018 Data",loc = "right")
				ax.legend(fontsize=10, loc='upper right')
				plt.savefig(four_tau_names[hist_name])
				plt.close()
	








#Store final states in tables (Commented out on 2 September 2025, do not delete yet)
#	for state in background_state_array:
#		final_state_dict_background[state] += 1/len(background_state_array)
#
#	#Obtain error bars
#	for state in final_state_dict_background:
#		final_state_dict_background_error[state] = np.sqrt(final_state_dict_background[state]*len(background_state_array))
#		final_state_dict_background_error[state] /= len(background_state_array)
#	
#	#for state in final_state_dict_signal_full:
#	#	final_state_dict_signal_full[state].append(final_state_dict_signal[state])
#	#for state in final_state_dict_data_full:
#	#	final_state_dict_data_full[state].append(final_state_dict_data[state])
#	#for state in final_state_dict_background_full:
#	#	final_state_dict_background_full[state].append(final_state_dict_background[state])
#    
#	#print(final_state_dict_signal)	
#	#print(final_state_dict_data)	
#	#print(final_state_dict_background)
#	
#    #print("Number of Signal events: %d"%len(fourtau_out["Signal"]["num_electron_tau_Arr"]))
#	#print("Number of Data events: %d"%(len(fourtau_out["Data_SingleMuon"]["num_electron_tau_Arr"]) + len(fourtau_out["Data_JetHT"]["num_electron_tau_Arr"])))
#	print("Number of Data events: %d"%(len(fourtau_out["Data_SingleMuon"]["num_electron_tau_Arr"])))
#	print("Number of Background events: %d"%len(background_state_array))
#
#
#	#Store information about taus in tex table
#	store_tau_states = True
#	if (store_tau_states):
#		#file = open("Final_State_Table_Gen.tex","w")
#		file = open("Final_State_Table_Reco_errorbars_05.tex","w")
#
#		#Set up the tex document
#		file.write("\\documentclass{article} \n")
#		file.write("\\usepackage{multirow} \n")
#		file.write("\\usepackage{multirow} \n")
#		file.write("\\usepackage{lscape}\n")
#		file.write("\\begin{document} \n")
#		file.write("\\begin{landscape} \n")
#		file.write("\\centering \n")
#
#		#Set up the table
#		file.write("\\begin{tabular}{|p{4.5cm}|p{3cm}|p{3cm}|p{3cm}|}")
#		file.write("\\hline \n")
#		file.write("\\multicolumn{4}{|c|}{Final State Table (Reco \\(\\Delta R < 0.05\\))} \\\\ \n")
#		file.write("\\hline \n")
#		file.write("4$\\tau$ Channel & 2 TeV Signal & Drell-Yan + Jets & Theory \\\\ \n")
#		file.write("\\hline \n")
#		for state in final_state_dict_signal:
#			file.write(state + " & %.3f"%final_state_dict_signal[state] + " $\\pm$ %.3f"%final_state_dict_signal_error[state] + 
#					" & %.3f"%final_state_dict_background[state] + "$\\pm$ %.3f"%final_state_dict_background_error[state] +
#					" & %.3f"%(final_state_dict_theory[state]) + "\\\\")
#					#" & %.3f"%final_state_dict_data[state] + " $\\pm$ %.3f"%final_state_dict_data_error[state] + "\\\\")
#			file.write("\n")
#			file.write("\\hline \n")
#		file.write("\\end{tabular} \n")
#		file.write("\\end{landscape} \n")
#		file.write("\\end{document}")
#		file.close()
	
