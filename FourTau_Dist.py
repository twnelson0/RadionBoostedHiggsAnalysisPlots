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
vector.register_awkward()

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
		state += "m"
	for i in range(4 - int(n_ele) - int(n_mu)):
		state += "h"
	return state

fin_state_vec = np.vectorize(fin_state)

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
		#Begin by checking if running on data or sample
		dataset = events.metadata['dataset']
		if ("Data_" in dataset): #Check to see if running on data
			self.isData = True	
		
		event_level = ak.zip(
			{
				"jet_trigger": events.HLTJet,
				"mu_trigger": events.HLTEleMuX,
				"pfMET": events.pfMET,
				"pfMETPhi": events.pfMETPhi,
				"event_weight": ak.ones_like(events.pfMET)*0.9,
                "n_electrons": ak.zeros_like(events.pfMET),
                "n_muons": ak.zeros_like(events.pfMET),
                "n_tau_electrons": ak.zeros_like(events.pfMET),
                "n_tau_muons": ak.zeros_like(events.pfMET)
				#"n_muons": events.nEle,
				#"n_electrons": events.nMu,
			},
			with_name="EventArray",
			behavior=candidate.behavior,
		)
		tau = ak.zip( 
			{
				"pt": events.boostedTauPt,
				"E": events.boostedTauEnergy,
				"Px": events.boostedTauPx,
				"Py": events.boostedTauPy,
				"Pz": events.boostedTauPz,
				"mass": events.boostedTauMass,
				"eta": events.boostedTauEta,
				"phi": events.boostedTauPhi,
				"nBoostedTau": events.nBoostedTau,
				"charge": events.boostedTauCharge,
				"iso": events.boostedTauByIsolationMVArun2v1DBoldDMwLTrawNew,
				"decay": events.boostedTaupfTausDiscriminationByDecayModeFinding,
			},
			with_name="TauArray",
			behavior=candidate.behavior,
		)
		electron = ak.zip(
			{
				"pt": events.elePt,
				"E": events.eleEn,
				"eta": events.eleEta,
				"phi": events.elePhi,
				"charge": events.eleCharge,
				"Px": events.elePt*np.cos(events.elePhi),
				"Py": events.elePt*np.sin(events.elePhi),
				"Pz": events.elePt*np.tan(2*np.arctan(np.exp(-events.eleEta)))**-1,
				"SCEta": events.eleSCEta,
				"IDMVANoIso": events.eleIDMVANoIso,
					
			},
			with_name="ElectronArray",
			behavior=candidate.behavior,
			
		)
		muon = ak.zip(
			{
				"pt": events.muPt,
				"E": events.muEn,
				"eta": events.muEta,
				"phi": events.muPhi,
				"charge": events.muCharge,
				"Px": events.muPt*np.cos(events.muPhi),
				"Py": events.muPt*np.sin(events.muPhi),
				"Pz": events.muPt*np.tan(2*np.arctan(np.exp(-events.muEta)))**-1,
				"nMu": events.nMu,
				"IDbit": events.muIDbit,
				"D0": events.muD0,
				"Dz": events.muDz
					
			},
			with_name="MuonArray",
			behavior=candidate.behavior,
			
		)

		AK8Jet = ak.zip(
			{
				"AK8JetDropMass": events.AK8JetSoftDropMass,
				"AK8JetPt": events.AK8JetPt,
				"eta": events.AK8JetEta,
				"phi": events.AK8JetPhi,
			},
			with_name="AK8JetArray",
			behavior=candidate.behavior,
		)
		
		Jet = ak.zip(
			{
				"Pt": events.jetPt,
				"PFLooseId": events.jetPFLooseId,
				"eta": events.jetEta,
				"phi": events.jetPhi,
				"DeepCSVTags_b": events.jetDeepCSVTags_b
			},
			with_name="PFJetArray",
			behavior=candidate.behavior,
		)

		if not(self.isData): #Check to see if this is (or is not) an MC simulation
			Gen_Info = ak.zip({
					"MCId": events.mcPID,
					"MotherId": events.mcMomPID,
					"Pt": events.mcPt,
					"Eta": events.mcEta,
					"Phi": events.mcPhi,
					"E": events.mcE,
					"Px": events.mcPt*np.cos(events.mcPhi),
					"Py": events.mcPt*np.sin(events.mcPhi),
					"Pz": events.mcPt*np.tan(2*np.arctan(np.exp(-events.mcEta)))**-1,

				},
				with_name = "GEN_Array",
				behavior=candidate.behavior,
			)

			PU_Info = ak.zip({
				"puTrue": events.puTrue[:,0]
				},
				with_name = "PU_Array",
				behavior=candidate.behavior,
			)
            
			#GenTau_Num = ak.num(np.bitwise_and(np.abs(Gen_Info.MCId) == 15, np.abs(Gen_Info.MotherId) != 15),axis=1) #Get the number of Generated taus that decayed from something
			GenTau_Num = ak.num(Gen_Info[np.abs(Gen_Info.MCId) == 15],axis=1)
			print("=============================================================================")
			print(event_level.event_weight)
			event_level["event_weight"] = event_level.event_weight**GenTau_Num #GenTau_Num #Apply weightings based on Gen Taus
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
			print(event_level.event_weight)
			dummy_weight = event_level["event_weight"]
			print("Applied gen weights")

			if (self.PU_bool): #Apply PU reweighting scheme
				PU_Arr = np.array(np.rint(ak.flatten(PU_Info.puTrue,axis=-1)),dtype = np.int8)
				PU_Corr = self.PUWeights[PU_Arr]
				event_level["event_weight"] = np.multiply(event_level.event_weight,PU_Corr) #Is this line screwing things up??
				#Debugg/gest PU reweighting
				#for i in range(len(event_level.event_weight)):
				#	if (dummy_weight[i]*PU_Corr[i] != event_level.event_weight[i]):
				#	    print("!!Event gen weighting mistmatch!!")
				#	    print("Event weight: %f"%event_level.event_weight[i])
				#	    print("Expected weight: %f"%dummy_weight[i]*PU_Corr[i])
				print(event_level.event_weight)



		
		#Histograms 4 tau
		#TauDeltaPhi_all_hist = (
		#	hist.Hist.new
		#	.StrCat(["Leading pair","Subleading pair"], name = "delta_phi")
		#	.Reg(50, -pi, pi, name="delta_phi_all", label=r"$\Delta \phi$") 
        #   .Double()
		#)

        #Force taus to be ordered via transverse momenta (if they are not already)
		tau = tau[ak.argsort(tau.pt,axis=1)]

		print("!!!=====Dataset=====!!!!")	
		print(type(dataset))
		print(dataset)


		print("Number of events before selection + Trigger: %d"%ak.num(tau,axis=0))

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
		
		#Triggering logic
		trigger_mask = bit_mask([self.trigger_bit])
		if (not(self.isData)):	#MC trigger logic
			if (self.OrTrigger): # and np.pi == np.exp(1)): #Select for both triggers
				print("Both Triggers")
				event_level_21 = event_level[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				event_level_fail = event_level[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				event_level_27 = event_level_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([27])) == bit_mask([27])]
				event_level_39 = event_level_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([39])) == bit_mask([39])]

				#Muon ID selection
				id_cond = np.bitwise_and(muon.IDbit,2) != 0
				d0_cond = np.abs(muon.D0) < 0.045
				dz_cond = np.abs(muon.Dz) < 0.2
				good_muon_cond = np.bitwise_and(id_cond, np.bitwise_and(d0_cond, dz_cond))
				muon = muon[good_muon_cond]
			
				#Single Muon Trigger	
				tau_21 = tau[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				tau_fail = tau[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				AK8Jet_21 = AK8Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				AK8Jet_fail = AK8Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				Jet_21 = Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				Jet_fail = Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				muon_21 = muon[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				muon_fail = muon[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				electron_21 = electron[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				electron_fail = electron[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				if (not(self.isData) and self.isData):
					Gen_Info_21 = Gen_Info[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
					Gen_Info_fail = Gen_Info[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
					
				

				#Apply offline Single Muon Cut
				tau_21 = tau_21[ak.any(muon_21.nMu > 0, axis = 1)]
				AK8Jet_21 = AK8Jet_21[ak.any(muon_21.nMu > 0, axis = 1)]
				Jet_21 = Jet_21[ak.any(muon_21.nMu > 0, axis = 1)]
				electron_21 = electron_21[ak.any(muon_21.nMu > 0, axis = 1)]
				if (not(self.isData) and self.isData):
					Gen_Info_21 = Gen_Info_21[ak.any(muon_21.nMu > 0, axis = 1)]
				muon_21 = muon_21[ak.any(muon_21.nMu > 0, axis = 1)]


				
				#pT
				tau_21 = tau_21[ak.any(muon_21.pt > 52, axis = 1)]
				AK8Jet_21 = AK8Jet_21[ak.any(muon_21.pt > 52, axis = 1)]
				Jet_21 = Jet_21[ak.any(muon_21.pt > 52, axis = 1)]
				event_level_21 = event_level_21[ak.any(muon_21.pt > 52, axis = 1)]
				electron_21 = electron_21[ak.any(muon_21.pt > 52, axis = 1)]
				if (not(self.isData) and self.isData):
					Gen_Info_21 = Gen_Info_21[ak.any(muon_21.pt > 52, axis = 1)]
				muon_21 = muon_21[ak.any(muon_21.pt > 52, axis = 1)]

				
				#Apply JetHT_MHT_MET Trigger
				tau_39 = tau_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([39])) == bit_mask([39])]
				AK8Jet_39 = AK8Jet_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([39])) == bit_mask([39])]
				Jet_39 = Jet_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([39])) == bit_mask([39])]
				electron_39 = electron_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([39])) == bit_mask([39])]
				muon_39 = muon_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([39])) == bit_mask([39])]
				if (not(self.isData) and self.isData):
					Gen_Info_39 = Gen_Info_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([39])) == bit_mask([39])]
		
				#HT Cut
				tau_39 = tau_39[event_level_39.HT > 550]	
				AK8Jet_39 = AK8Jet_39[event_level_39.HT > 550]	
				Jet_39 = Jet_39[event_level_39.HT > 550]
				muon_39 = muon_39[event_level_39.HT > 550]
				electron_39 = electron_39[event_level_39.HT > 550]
				if (not(self.isData) and self.isData):
					Gen_Info_39 = Gen_Info_39[event_level_39.HT > 550]
				event_level_39 = event_level_39[event_level_39.HT > 550]

				#MHT Cut
				tau_39 = tau_39[event_level_39.MHT > 110]	
				AK8Jet_39 = AK8Jet_39[event_level_39.MHT > 110]	
				Jet_39 = Jet_39[event_level_39.MHT > 110]
				muon_39 = muon_39[event_level_39.MHT > 110]
				electron_39 = electron_39[event_level_39.MHT > 110]
				if (not(self.isData) and self.isData):
					Gen_Info_39 = Gen_Info_39[event_level_39.MHT > 110]
				event_level_39 = event_level_39[event_level_39.MHT > 110]

				#MET Cut	
				tau_39 = tau_39[event_level_39.pfMET > 110]	
				AK8Jet_39 = AK8Jet_39[event_level_39.pfMET > 110]	
				Jet_39 = Jet_39[event_level_39.pfMET > 110]
				muon_39 = muon_39[event_level_39.pfMET > 110]
				electron_39 = electron_39[event_level_39.pfMET > 110]
				if (not(self.isData) and self.isData):
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
				if (not(self.isData) and self.isData):
					Gen_Info = ak.concatenate((Gen_Info_21,Gen_Info_39))
				
			else: #Single Trigger
				#print("Single Trigger (in theory)")
				if (self.trigger_bit != None and self.OrTrigger == False):
					if (self.trigger_bit == 21): #Single Mu
						print("Single Trigger: Mu Trigger (21)")
						tau = tau[np.bitwise_and(event_level.mu_trigger,trigger_mask) == trigger_mask]
						AK8Jet = AK8Jet[np.bitwise_and(event_level.mu_trigger,trigger_mask) == trigger_mask]
						Jet = Jet[np.bitwise_and(event_level.mu_trigger,trigger_mask) == trigger_mask]
						muon = muon[np.bitwise_and(event_level.mu_trigger,trigger_mask) == trigger_mask]
						event_level = event_level[np.bitwise_and(event_level.mu_trigger,trigger_mask) == trigger_mask]
						
						#Muon ID selection
						id_cond = np.bitwise_and(muon.IDbit,2) != 0
						d0_cond = np.abs(muon.D0) < 0.045
						dz_cond = np.abs(muon.Dz) < 0.2
						good_muon_cond = np.bitwise_and(id_cond, np.bitwise_and(d0_cond, dz_cond))
						muon = muon[good_muon_cond]
				
						#Apply offline Single Muon Cut
						#if (np.exp(1) != np.pi):
						tau = tau[ak.any(muon.nMu > 0, axis = 1)]
						AK8Jet = AK8Jet[ak.any(muon.nMu > 0, axis = 1)]
						Jet = Jet[ak.any(muon.nMu > 0, axis = 1)]
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
						tau = tau[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
						AK8Jet = AK8Jet[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
						Jet = Jet[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
						muon = muon[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
						electron = electron[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
						event_level = event_level[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
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
			if ("SingleMuon" in dataset): #Single Mu
				#Muon ID selection
				id_cond = np.bitwise_and(muon.IDbit,2) != 0
				d0_cond = np.abs(muon.D0) < 0.045
				dz_cond = np.abs(muon.Dz) < 0.2
				good_muon_cond = np.bitwise_and(id_cond, np.bitwise_and(d0_cond, dz_cond))
				muon = muon[good_muon_cond]


				print("Single Muon Trigger")
				tau = tau[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]	
				AK8Jet = AK8Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]	
				Jet = Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]	
				muon = muon[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]	
				electron = electron[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]	
				event_level = event_level[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]

				if (self.OrTrigger): #If working on both triggers drop events that passed JetHT trigger
					tau = tau[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) != bit_mask([39])]	
					AK8Jet = AK8Jet[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) != bit_mask([39])]	
					Jet = Jet[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) != bit_mask([39])]	
					muon = muon[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) != bit_mask([39])]	
					electron = electron[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) != bit_mask([39])]	
					event_level = event_level[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) != bit_mask([39])]
					

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
				
			if ("JetHT" in dataset): #HT 
				print("Jet Trigger")
				tau = tau[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) == bit_mask([39])]	
				AK8Jet = AK8Jet[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) == bit_mask([39])]	
				Jet = Jet[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) == bit_mask([39])]	
				muon = muon[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) == bit_mask([39])]	
				electron = electron[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) == bit_mask([39])]	
				event_level = event_level[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) == bit_mask([39])]
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
		
		cond1 = np.bitwise_and(np.abs(electron.SCEta) <= 0.8, electron.IDMVANoIso > 0.837)
		cond2 = np.bitwise_and(np.bitwise_and(np.abs(electron.SCEta) > 0.8, np.abs(electron.SCEta) <= 1.5), electron.IDMVANoIso > 0.715)
		cond3 = np.bitwise_and(np.abs(electron.SCEta) >= 1.5, electron.IDMVANoIso > 0.357)
		good_electron_cond = np.bitwise_or(cond1,np.bitwise_or(cond2,cond3))
		electron = electron[good_electron_cond]
		
		event_level["n_muons"] = ak.singletons(ak.num(muon.pt,axis=1))
		event_level["n_electrons"] = ak.singletons(ak.num(electron.pt,axis=1))
		
		#Produce distribution of e/mu to nearest tau
		electron_fourVec = ak.zip({"x": electron.Px, "y": electron.Py, "z": electron.Pz,"t": electron.E},with_name = "LorentzVector")
		muon_fourVec = ak.zip({"x": muon.Px, "y": muon.Py, "z": muon.Pz,"t": muon.E},with_name = "LorentzVector")
		#if (not(self.isData) and self.isData): #Use gen Leptons 
		#	electron_fourVec = ak.zip({"x": Gen_Info[np.abs(Gen_Info.MCId) == 11].Px, "y": Gen_Info[np.abs(Gen_Info.MCId) == 11].Py, "z": Gen_Info[np.abs(Gen_Info.MCId) == 11].Pz,"t": Gen_Info[np.abs(Gen_Info.MCId) == 11].E},with_name = "LorentzVector")
		#	muon_fourVec = ak.zip({"x": Gen_Info[np.abs(Gen_Info.MCId) == 13].Px, "y": Gen_Info[np.abs(Gen_Info.MCId) == 13].Py, "z": Gen_Info[np.abs(Gen_Info.MCId) == 13].Pz,"t": Gen_Info[np.abs(Gen_Info.MCId) == 13].E},with_name = "LorentzVector")
		
		tau_fourVec = ak.zip({"x": tau.Px, "y": tau.Py, "z": tau.Pz,"t": tau.E},with_name = "LorentzVector")
		#electron_fourVec = ak.zip({})

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
		
		for j in range(ak.num(tau,axis=0)): #Debugging work
			if (len(min_tau_ele[j]) != len(electron_fourVec[j])):
				print("!!Electron electron-tau dR mismatch!!")
				print("Min Electron-Tau dR length: %d"%len(min_tau_ele[j]))
				print("Electron length: %d"%len(electron_fourVec[j]))
			if (len(min_tau_mu[j]) != len(muon_fourVec[j])):
				print("!!Electron electron-tau dR mismatch!!")
				print("Min Muon-tau dR length: %d"%len(min_tau_mu[j]))
				print("Moun length: %d"%len(muon_fourVec[j]))

		print("There are %d electrons in total"%ak.sum(ak.num(electron.pt,axis=1)))
		print("There are %d muons in total"%ak.sum(ak.num(muon.pt,axis=1)))

		ele_dR_collection = electron_fourVec.delta_r(min_tau_ele)
		mu_dR_collection = muon_fourVec.delta_r(min_tau_mu)
		#if (self.isData):
		electron["tau_min_dR"] = ele_dR_collection 
		muon["tau_min_dR"] = mu_dR_collection
		#else:
		#	electron["tau_min_dR"] = ak.zeros_like(electron.pt) 
		#	muon["tau_min_dR"] = ak.zeros_like(muon.pt)
            
		
		#Apply selections
		tau = tau[tau.pt > 20] #pT selection
		
		#Remove events with fewer than 4 taus
		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		event_level = event_level[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		electron = electron[ak.num(tau) >= 4] 
		muon = muon[ak.num(tau) >= 4] 
		tau = tau[ak.num(tau) >= 4] #4 tau events
		if (self.isData or not(self.isData)):
			print("# of events after pT cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		tau = tau[np.abs(tau.eta) < 2.3] #eta selection
		
		#Remove events with fewer than 4 taus	
		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		event_level = event_level[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		electron = electron[ak.num(tau) >= 4] 
		muon = muon[ak.num(tau) >= 4] 
		tau = tau[ak.num(tau) >= 4] #4 tau events
		if (self.isData or not(self.isData)):
			print("# of events after eta cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		
		
		#Isolation and decay selections
		tau = tau[tau.decay >= 0.5]
		
		#Remove events with fewer than 4 taus	
		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		event_level = event_level[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		electron = electron[ak.num(tau) >= 4] 
		muon = muon[ak.num(tau) >= 4] 
		tau = tau[ak.num(tau) >= 4] #4 tau events
		if (self.isData or not(self.isData)):
			print("# of events after decay cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		
		tau = tau[tau.iso >= 0.0] #Make loose to ensure high number of statistics
		#Remove events with fewer than 4 taus	
		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		event_level = event_level[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		electron = electron[ak.num(tau) >= 4] 
		muon = muon[ak.num(tau) >= 4] 
		tau = tau[ak.num(tau) >= 4] #4 tau events
		if (self.isData or not(self.isData)):
			print("# of events after isolation cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))


		#Delta R Cut on taus (identifiy and remove jets incorrectly reconstructed as taus)
		a,b = ak.unzip(ak.cartesian([tau,tau], axis = 1, nested = True)) #Create all di-tau pairs
		select_arr = np.bitwise_and(deltaR(a,b) < 0.8, deltaR(a,b) != 0)
		#for i in range(5):
		#	print(tau.pt[i]) 
		#	print(select_arr[i])
		#tau["dRCut"] = select_arr
		#tau = tau[ak.any(tau.dRCut, axis = 2) == True]
		#if (self.isData or not(self.isData)):
		#	print("# of events after delta R cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		
		#AK8Jet = AK8Jet[(ak.sum(tau.charge,axis=1) == 0)] #Apply charge conservation cut to AK8Jets
		#event_level = event_level[(ak.sum(tau.charge,axis=1) == 0)]
		#Jet = Jet[(ak.sum(tau.charge,axis=1) == 0)]
		#electron = electron[(ak.sum(tau.charge,axis=1) == 0)]
		#muon = muon[(ak.sum(tau.charge,axis=1) == 0)]
		#tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		#if (self.isData or not(self.isData)):
			#print(ak.sum(tau.charge,axis=1))
		#	for q in ak.sum(tau.charge,axis=1):
		#		if q != 0:
		#			print("Somethign went wrong")
		#	print("# of events after lepton number cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))

		#Remove all events with 3 or fewer taus (after selections at once)
		#AK8Jet = AK8Jet[ak.num(tau) >= 4]
		#event_level = event_level[ak.num(tau) >= 4]
		#Jet = Jet[ak.num(tau) >= 4]
		#electron = electron[ak.num(tau) >= 4] 
		#muon = muon[ak.num(tau) >= 4] 
		#tau = tau[ak.num(tau) >= 4] #4 tau events
		#if (self.isData or not(self.isData)):
		#	print("# of events after 4-tau cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
	
		#print("tau length = %d\nevent_level length = %d"%(ak.num(tau,axis=0),ak.num(event_level,axis=0)))	
		#tau = tau[ak.num(tau) > 0] #Handle empty arrays left over
		
		#Z Mutliplticity of taus
		event_level["ZMult_tau"] = find_Z_Candidates(tau,ak.ArrayBuilder()).snapshot()
		


		
        #print(ele_dR_collection)
		#print(mu_dR_collection)

		
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
	
			#Obtain leading pair
			tau_4vec = ak.zip({"t": tau.E, "x": tau.Px, "y": tau.Py, "z": tau.Pz},with_name="Momentum4D")
			tau_lead,tau_other = ak.unzip(ak.cartesian([tau_4vec[:,0],tau_4vec], axis = 1, nested = False))
			#deltaR_Arr = deltaR(tau_lead,tau) #Delta R Between leading tau and all other taus in event
			deltaR_Arr = tau_lead.deltaR(tau_4vec)
			#deltaphi_Arr = delta_phi(tau_lead,tau) #Debugg the delta phi values

			#Construct all delta Rs from all possible pairings
			#tau_plus,tau_minus = ak.unzip(ak.cartesian(tau[tau.charge > 0],tau[tau.charge < 0]))
			#tau1,tau2 = ak.unzip(ak.cartesian([tau,tau],axis = 1, nested = False))
			#deltaR_Full = deltaR(tau1,tau2)
			#deltaR_Full = deltaR_Full[deltaR_Full != 0] #Remove 0s
			#dummyIndx = 10
			#if (len(deltaR_Full) < 10):
			#	test_range = len(deltaR_Full)
			#else:
			#	test_range = 10
				#test_range = len(deltaR_Full)
			#print("Delta R of all possible pairings")
			#for i in range(test_range):
			#	print(deltaR_Full[i])
			
			#Select only taus with oposite charge
			#leadingTau_Pair = tau[charge_Arr == 0]
			#deltaR_Arr = deltaR_Arr[charge_Arr == 0]
			#deltaphi_Arr = deltaphi_Arr[charge_Arr == 0]

			#Remove leading tau from consideration
			leadingTau_Pair = tau[deltaR_Arr != 0]
			#deltaphi_Arr = deltaphi_Arr[deltaR_Arr != 0]
			charge_Arr = charge_Arr[deltaR_Arr != 0]
			deltaR_Arr = deltaR_Arr[deltaR_Arr != 0]

			#Look at the delta Phi and delta R values
			#for i in range(len(deltaphi_Arr)):
			#	for dphi in deltaphi_Arr[i]:
			#		if (np.abs(dphi) > pi):
			#			print("!!!Outside Expected Range!!!")
			#			print(deltaphi_Arr[i])
			#			print(dphi)
			#			print(tau[:,0][i].phi)
			#			print(leadingTau_Pair[i].phi)

			#print("Charges of all possible pairings")
			#for i in range(test_range):
			#	print(charge_Arr[i])
		
			#Select OS tau that minimizes delta R	
			leadingTau_Pair = leadingTau_Pair[deltaR_Arr == ak.min(deltaR_Arr,axis=1)] #Paired tau is selected as the one that minimized deltaR with leading tau
			#deltaphi_Arr = deltaphi_Arr[deltaR_Arr == ak.min(deltaR_Arr,axis=1)]
			charge_Arr = charge_Arr[deltaR_Arr == ak.min(deltaR_Arr,axis=1)]
			deltaR_Arr = ak.min(deltaR_Arr,axis=1)
			pair1_charge = charge_Arr
	
			#print("Leading pair Minimized delta R:")
			#for i in range(test_range):
			#	print(deltaR_Arr[i])

			#print("Charge of leading Piar")
			#for i in range(test_range):
			#	print(pair1_charge[i])
			
			zerocharge = pair1_charge[pair1_charge == 0]
			zerocharge = zerocharge[ak.num(zerocharge,axis=1) > 0]
			nonZerocharge = pair1_charge[pair1_charge != 0]
			nonZerocharge = nonZerocharge[ak.num(nonZerocharge,axis=1) > 0]

			numZeroCharge_lead = ak.num(zerocharge,axis=0)
			numNonZeroCharge_lead = ak.num(nonZerocharge,axis=0)

			#print("There are %d leading pairs with an electric charge of 0"%numZeroCharge_lead)
			#print("There are %d leading pairs with an electric charge of +/- 2"%numNonZeroCharge_lead)

			#Remove any empty/invalid pairings
			tau = tau[ak.num(leadingTau_Pair) > 0]
			Jet = Jet[ak.num(leadingTau_Pair) > 0]
			AK8Jet = AK8Jet[ak.num(leadingTau_Pair) > 0]
			muon = muon[ak.num(leadingTau_Pair) > 0]
			electron = electron[ak.num(leadingTau_Pair) > 0]
			event_level = event_level[ak.num(leadingTau_Pair) > 0]
			leadingTau_Pair = leadingTau_Pair[ak.num(leadingTau_Pair) > 0]

			#Find second pair
			tau_lead = tau[tau.pt == tau[:,0].pt]
			leadingpT = ak.ravel(tau_lead.pt)

			leadingPairpT = leadingTau_Pair.pt
			tau_rem = tau[tau.pt != leadingpT]
			tau_rem = tau_rem[tau_rem.pt != ak.ravel(leadingPairpT)] #Get Remaining taus

			#Drop events with no oposite signs
			charge_Arr = totalCharge(tau_rem[:,0],tau_rem)
		
			tau_rem_4vec = ak.zip({"t": tau_rem.E, "x": tau_rem.Px, "y": tau_rem.Py, "z": tau_rem.Pz},with_name="Momentum4D")
			tau_nextlead,tau_rem_other = ak.unzip(ak.cartesian([tau_rem_4vec[:,0],tau_rem_4vec], axis = 1, nested = False))
			#deltaR_Arr = deltaR(tau_nextlead,tau_rem)
			deltaR_Arr = tau_nextlead.deltaR(tau_rem_4vec)

			#Look at all pairings left
			#tau3,tau4 = ak.unzip(ak.cartesian([tau_rem,tau_rem], axis=1, nested = False))
			#deltaR_rem_Full = deltaR(tau3,tau4)
			#deltaR_rem_Full = deltaR_rem_Full[deltaR_rem_Full != 0]
			
			#print("Remaining pairings")
			#for i in range(test_range):
			#	print(deltaR_rem_Full[i])

			#Remove leading tau
			leadingTau_NextPair = tau_rem[deltaR_Arr != 0]
			deltaR_temp = deltaR_Arr[deltaR_Arr != 0]
			charge_Arr = charge_Arr[deltaR_Arr != 0]
			
			#print("Charges of all possible pairings")
			#for i in range(test_range):
			#	print(charge_Arr[i])
			
			#Select pair that minimizes delta R	
			leadingTau_NextPair = leadingTau_NextPair[deltaR_temp == ak.min(deltaR_temp,axis=1)]
			charge_Arr = charge_Arr[deltaR_temp == ak.min(deltaR_temp,axis=1)]
			pair2_charge = charge_Arr
			deltaR_temp = deltaR_temp[deltaR_temp == ak.min(deltaR_temp,axis=1)]
			
			#print("Subleading Pair minimized delta R:")
			#for i in range(test_range):
			#	print(deltaR_temp[i])
			
			#print("Charge of subleading Piar")
			#for i in range(test_range):
			#	print(pair2_charge[i])

			zerocharge = pair2_charge[pair2_charge == 0]
			zerocharge = zerocharge[ak.num(zerocharge,axis=1) > 0]
			nonZerocharge = pair2_charge[pair2_charge != 0]
			nonZerocharge = nonZerocharge[ak.num(nonZerocharge,axis=1) > 0]
			
			numZeroCharge_sublead = ak.num(zerocharge,axis=0)
			numNonZeroCharge_sublead = ak.num(nonZerocharge,axis=0)

			#print("There are %d subleading pairs with an electric charge of 0"%numZeroCharge_sublead)
			#print("There are %d subleading pairs with an electric charge of +/- 2"%numNonZeroCharge_sublead)

			#Determine what is signal and what is faking signal (if sum and product of total charge of both pairs are 0 than it's signal otherwise it's a fake)
			#print("Tau length for debugging (pre signal condition)")
			#print(len(tau))
			signal_cond = np.bitwise_and(ak.all(pair1_charge + pair2_charge,axis=1) == 0, ak.all(pair1_charge * pair2_charge,axis=1) == 0)
    
			#Drop all but signal (OSOS Higgs)
			#tau_rem = tau_rem[signal_cond]
			#tau_lead = tau_lead[signal_cond]
			#leadingTau_Pair = leadingTau_Pair[signal_cond]
			#leadingTau_NextPair = leadingTau_NextPair[signal_cond]
			#charge_Arr = charge_Arr[signal_cond]
			#tau = tau[signal_cond]
			#Jet = Jet[signal_cond]
			#AK8Jet = AK8Jet[signal_cond]
			#event_level = event_level[signal_cond]
			#muon = muon[signal_cond]
			#electron = electron[signal_cond]
            
			#print("Tau length for debugging (post signal condition)")
			#print(len(tau))
			
			#Remove any empty/invalid pairings (this may be vegistal at best but I'm not sure)
			tau_lead = tau_lead[ak.num(leadingTau_NextPair) != 0]
			leadingTau_Pair = leadingTau_Pair[ak.num(leadingTau_NextPair) != 0]
			tau_rem = tau_rem[ak.num(leadingTau_NextPair) != 0]
			tau = tau[ak.num(leadingTau_NextPair) != 0]
			leadingTau_NextPair = leadingTau_NextPair[ak.num(leadingTau_NextPair) != 0]
			
			#Obtain next/remaining leading tau
			tau_nextlead = tau_rem[tau_rem.pt == tau_rem[:,0].pt]

			#Check size of taus
			#print("Leading Tau %d"%len(tau_lead))
			#print(tau_lead.pt)
			#print("Leading Tau Pair %d"%len(leadingTau_Pair))
			#print(leadingTau_Pair.pt)
			#print("Next Leading Tau %d"%len(tau_nextlead))
			#print(tau_nextlead.pt)
			#print("Next Leading Tau Pair %d"%len(leadingTau_NextPair))
			#print(leadingTau_NextPair.pt)

			#Reconstruct tau object in order of pairings 
			tau = ak.concatenate((tau_lead,leadingTau_Pair),axis=1)
			tau = ak.concatenate((tau,tau_nextlead),axis=1)
			tau = ak.concatenate((tau,leadingTau_NextPair),axis=1)
		
			#Determine if taus match to lepton (e or mu) and make sutiable replacements
			lead_tau = ak.zip({"t": tau[:,0].E,"x": tau[:,0].Px, "y": tau[:,0].Py,"z" : tau[:,0].Pz},with_name = "Momentum4D")
			leadingpair_tau = ak.zip({"t": tau[:,1].E,"x": tau[:,1].Px, "y": tau[:,1].Py,"z" : tau[:,1].Pz},with_name = "Momentum4D")
			sublead_tau = ak.zip({"t": tau[:,2].E,"x": tau[:,2].Px, "y": tau[:,2].Py,"z" : tau[:,2].Pz},with_name = "Momentum4D")
			subleadingpair_tau = ak.zip({"t": tau[:,3].E,"x": tau[:,3].Px, "y": tau[:,3].Py,"z" : tau[:,3].Pz},with_name = "Momentum4D")
			tau_fourVec_Arr = ak.Array([lead_tau,leadingpair_tau,sublead_tau,subleadingpair_tau])

			electron_fourVec = ak.zip({"t": electron.E, "x": electron.Px, "y": electron.Py, "z": electron.Pz},with_name = "Momentum4D")
			muon_fourVec = ak.zip({"t": muon.E, "x": muon.Px, "y": muon.Py, "z": muon.Pz},with_name = "Momentum4D")

			Hadronic_Dict = {0: "LeadingTau_h", 1: "PairedLeadingTau_h", 2: "NextLeadingTau_h", 3: "PairedNextLeadingTau_h"}
			electron_Dict = {0: "LeadingTau_ele", 1: "PairedLeadingTau_ele", 2: "NextLeadingTau_ele", 3: "PairedNextLeadingTau_ele"}
			muon_Dict = {0: "LeadingTau_mu", 1: "PairedLeadingTau_mu", 2: "NextLeadingTau_mu", 3: "PairedNextLeadingTau_mu"}
 
			for i in range(4):
				print(i)
				tau_fourVec = ak.zip({"t": tau[:,i].E,"x": tau[:,i].Px, "y": tau[:,i].Py,"z" : tau[:,i].Pz},with_name = "Momentum4D")
				elec_dR = tau_fourVec.deltaR(electron_fourVec)
				muon_dR = tau_fourVec.deltaR(muon_fourVec)

				#Choose leptons with smallest delta Rs such that are < 0.1 
				valid_elec_dR = elec_dR[elec_dR < 0.1]
				misId_ele_cond = elec_dR == ak.min(elec_dR,axis=1)
				misId_ele_cond = ak.fill_none(misId_ele_cond,[False],axis=0) #Find the the smallest dR between electron and tau

				min_elec_dR = ak.min(elec_dR,axis=1)
				min_elec_dR = ak.fill_none(min_elec_dR,10) #Fill Nones with impossibly large values
				
				valid_mu_dR = muon_dR[muon_dR < 0.1] 
				misId_mu_cond = muon_dR == ak.min(muon_dR,axis=1)
				misId_mu_cond = ak.fill_none(misId_mu_cond,[False],axis=0) #Find the the smallest dR between muon and tau

				min_mu_dR = ak.min(muon_dR,axis=1)
				min_mu_dR = ak.fill_none(min_mu_dR,10) #Fill Nones with impossibly large values
				
				#Seperate electrons and muons into those matched and those not matched
				misId_ele = electron_fourVec[misId_ele_cond]
				misId_mu = muon_fourVec[misId_mu_cond]

				use_ele = min_elec_dR < min_mu_dR
				use_mu = min_elec_dR > min_mu_dR


				#This logic may be broken!!
				event_level["n_tau_electrons"] = ak.where(ak.all(ak.singletons(use_ele) == True,axis=1),event_level["n_tau_electrons"] + 1, event_level["n_tau_electrons"])
				event_level["n_tau_muons"] = ak.where(ak.all(ak.singletons(use_mu) == True,axis=1),event_level["n_tau_muons"] + 1, event_level["n_tau_muons"]) 

				#print("Max number of electrons = %d"%ak.max(event_level.n_electrons,axis=0))
				#print("Max number of muons = %d"%ak.max(event_level.n_muons,axis=0))
                
				n_wrong0 = 0
				n_wrong1 = 0
				print("Tau %d"%i)
				for j in range(len(event_level.n_tau_electrons)):
					if (event_level[j].n_tau_electrons > 4):
						n_wrong0 += 1
					if (event_level[j].n_tau_muons > 4):
						n_wrong1 += 1
                
				print("%d events have more than 4 electrons"%n_wrong0)
				print("%d events have more than 4 muons"%n_wrong1)

				#Store reco information of taus
				event_level[Hadronic_Dict[i]] = np.bitwise_not(np.bitwise_and(use_ele,use_mu))
				event_level[electron_Dict[i]] = use_ele
				event_level[muon_Dict[i]] = use_mu

				tau_fourVec = ak.where(use_ele, misId_ele, tau_fourVec) #Replace lead tau with electron when applicable
				tau_fourVec = ak.where(use_mu, misId_mu, tau_fourVec) #Replace lead tau with electron when applicable
				
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

				#electron_fourVec = electron_fourVec[np.bitwise_not(misId_ele_cond)]
				#muon_fourVec = muon_fourVec[np.bitwise_not(misId_mu_cond)]

				#Update taus
				tau[:,i]["E"] = tau_fourVec.t
				tau[:,i]["Px"] = tau_fourVec.x
				tau[:,i]["Py"] = tau_fourVec.y
				tau[:,i]["Pz"] = tau_fourVec.z
				
			
			#Check taus are not pairing to the same leptons
			for evnt in range(ak.num(tau,axis=0)):
				if (tau[:,0][evnt].E == tau[:,1][evnt].E):
					print("!!!Leading tau and paired tau appear to be matching to the same lepton!!!")
				if (tau[:,2][evnt].E == tau[:,3][evnt].E):
					print("!!!Subleading tau and paired tau appear to be matching to the same lepton!!!")

				

			
            #reduced_deltaR = deltaR_Arr[deltaR_Arr != 0]			
			#for x in reduced_deltaR:
			#	print(x)

			#tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau[tau.charge > 0],2))
			#tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau[tau.charge < 0],2))

			#for t1,t2 in zip(ak.ravel(tau_plus1.pt), ak.ravel(tau_plus2.pt)):
			#	if (t1 < t2):
			#		print("Fundamental Assumption broken")
			#for t1,t2 in zip(ak.ravel(tau_minus1.pt), ak.ravel(tau_minus2.pt)):
			#	if (t1 < t2):
			#		print("Fundamental Assumption broken")

			#Obtain di-tau delta R and higgs delta R (if there are any events left)
			if (ak.num(tau,axis=0) > 0):
				tau1 = tau[tau.pt == tau[:,0].pt]
				tau2 = tau[tau.pt == tau[:,1].pt]
				tau3 = tau[tau.pt == tau[:,2].pt]
				tau4 = tau[tau.pt == tau[:,3].pt]
	
				#print("Tau length for debugging")
				#print(len(tau))
				#for t in tau:
				#	if len(t) != 4:
				#		print("Event does not have 4 events")
				
				leading_deltaR = deltaR(tau1,tau2)
				next_deltaR = deltaR(tau3,tau4)
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
					if (self.isData or not(self.isData)):
						print("# of events after visible mass cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))


				#higgs_dR = deltaR(leading_higgs, nextleading_higgs) 
				higgs_dR = leading_higgs.deltaR(nextleading_higgs) #Use vector library for delta R calculations
				#ditau_dR = ak.concatenate((leading_deltaR, next_deltaR),axis=1)

				higgs_cond = ak.all(higgs_dR >= 2.0, axis = 1)
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
				#tau_cond = tau_cond[higgs_cond]
				if (self.isData or not(self.isData)):
					print("# of events after Higgs cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
	
			#Apply Tau topolotical condition	
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

		#Cut events with non-zero b jets
		#tau = tau[event_level.nBJets == 0]
		#Jet = Jet[event_level.nBJets == 0]
		#AK8Jet = AK8Jet[event_level.nBJets == 0]
		#electron = electron[event_level.nBJets == 0]
		#muon = muon[event_level.nBJets == 0]
		#event_level = event_level[event_level.nBJets == 0]
		if (self.isData or not(self.isData)):
			print("# of events after b-jet cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		#print(len(muon))
		#print(len(tau))
		#print(len(event_level))

		#Z Multiplicity function
		def Z_Mult_Function(lepton,lep_flavor): 
			#Make Good muon selection
			if (lep_flavor == "mu"):
				if (self.trigger_bit == 39):
					id_cond = np.bitwise_and(lepton.IDbit,2) != 0
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
			weight_Val = 1 
		else:
			weight_Val = weight_calc(dataset,numEvents_Dict[dataset])
			print("=========!!!Weight Debugging!!!=========")
			print(dataset)
			print("Luminosity Weight = %f"%weight_Val)
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
		var_nn = ak.zip(
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
				"numBJet": event_level.nBJets,
				"weight": event_level.event_weight*weight_Val,
				}
			)

		if not(self.isData):
			file_name = (dataset + ".parquet")
			ak.to_parquet(var_nn,file_name)
		else:
			if (dataset != "signal"):
				file_name = (dataset  + ".parquet")
			else:
				file_name = (dataset + "_mass_" + self.massVal + "GeV.parquet")
			if (os.path.isfile(file_name)): #Append to existing parquet file
				file_data = ak.from_parquet(file_name)
				var_nn = ak.concatenate([file_data,var_nn])
				ak.to_parquet(var_nn,file_name)
			else:
				ak.to_parquet(var_nn,file_name) #Create parquet file

		#print(weight_Val)
		#print(event_level.event_weight)
		print("===================!!!=Weight Debugging!!!====================")
		print(event_level.event_weight*weight_Val)
		print("===================!!!=Weight Debugging!!!====================")
		return{
			dataset: {
				#"Weight": weight_Val,
				"Weight_Val": weight_Val,
				"Weight": ak.to_list(event_level.event_weight*weight_Val), 
				"FourTau_Mass_Arr": ak.to_list(FourTau_Mass_Arr),
				"HiggsDeltaPhi_Arr": ak.to_list(Higgs_DeltaPhi_Arr),
				"LeadingHiggs_mass": ak.to_list(LeadingHiggs_mass_Arr),
				"SubLeadingHiggs_mass": ak.to_list(SubLeadingHiggs_mass_Arr),
				"Higgs_DeltaR_Arr": ak.to_list(diHiggs_dR_Arr),
				"leading_dR_Arr": ak.to_list(leading_dR_Arr),
				"subleading_dR_Arr": ak.to_list(subleading_dR_Arr),
				"LeadingHiggsSgn_Arr" : ak.to_list(event_level.LeadingHiggs_Charge),
				"SubleadingHiggsSgn_Arr" : ak.to_list(ak.ravel(event_level.SubleadingHiggs_Charge)),
				
				"leading_dPhi_Arr": ak.to_list(leading_dPhi_Arr),
				"subleading_dPhi_Arr": ak.to_list(subleading_dPhi_Arr),
				"radionMET_dPhi_Arr": ak.to_list(radionMET_dPhi),
				"leadingHiggs_Rad_dR_Arr":ak.to_list(leadingHiggs_Rad_dR),
				"subleadingHiggs_Rad_dR_Arr": ak.to_list(subleadingHiggs_Rad_dR),
				"leadingHiggs_MET_dPhi_Arr": ak.to_list(leadingHiggs_MET_dPhi_Arr),
				"subleadingHiggs_MET_dPhi_Arr": ak.to_list(subleadingHiggs_MET_dPhi_Arr),
				"Radion_eta_Arr": ak.to_list(ak.ravel(Radion_Reco.eta)),
				"Radion_Charge_Arr": ak.to_list(event_level.Radion_Charge),
				
				"radionPT_Arr" : ak.to_list(radionPT_Arr),
				"tau_pt_Arr": ak.to_list(ak.ravel(tau.pt)),
				"tau_lead_pt_Arr": ak.to_list(ak.ravel(tau[ak.argsort(tau.pt,axis=-1)][:,3].pt)),
				"tau_sublead_pt_Arr": ak.to_list(ak.ravel(tau[ak.argsort(tau.pt,axis=-1)][:,2].pt)),
				"tau_3rdlead_pt_Arr": ak.to_list(ak.ravel(tau[ak.argsort(tau.pt,axis=-1)][:,1].pt)),
				"tau_4thlead_pt_Arr": ak.to_list(ak.ravel(tau[ak.argsort(tau.pt,axis=-1)][:,0].pt)),
				"tau_eta_Arr": ak.to_list(ak.ravel(tau.eta)),
				"ZMult_Arr": ak.to_list(ak.ravel(event_level.ZMult)),
				"ZMult_ele_Arr": ak.to_list(ak.ravel(event_level.ZMult_e)),
				"ZMult_mu_Arr": ak.to_list(ak.ravel(event_level.ZMult_mu)),
				"ZMult_tau_Arr": ak.to_list(ak.ravel(event_level.ZMult_tau)),
				"BJet_Arr": ak.to_list(ak.ravel(event_level.nBJets)),
				"Lumi_Val": Lumi_2018,
				"CrossSec_Val": crossSecVal,
				"NEvent_Val": numEvents_Dict[dataset],
				"Num_Electrons_Arr": ak.to_list(ak.ravel(event_level.n_electrons)),
				"Num_Muons_Arr": ak.to_list(ak.ravel(event_level.n_muons)),
				"Electron_tau_dR_Arr": ak.to_list(ak.ravel(ak.where(ak.num(electron.tau_min_dR,axis=1)!= 0, electron.tau_min_dR, ak.singletons(np.ones(ak.num(electron.tau_min_dR,axis=0))*999)))),
				"Muon_tau_dR_Arr": ak.to_list(ak.ravel(ak.where(ak.num(muon.tau_min_dR,axis=1) != 0, muon.tau_min_dR, ak.singletons(np.ones(ak.num(muon.tau_min_dR,axis=0))*999)))), #muon.tau_min_dR
                "num_electron_tau_Arr": ak.to_list(ak.ravel(event_level.n_tau_electrons)),
                "num_muon_tau_Arr": ak.to_list(ak.ravel(event_level.n_tau_muons)),
				#"LeadTau_h": ak.to_list(ak.ravel(event_level.LeadingTau_h)),	
				#"PairLeadTau_h": ak.to_list(ak.ravel(event_level.PairedLeadingTau_h)),	
				#"NextLeadTau_h": ak.to_list(ak.ravel(event_level.NextLeadingTau_h)),	
				#"PairNextLeadTau_h": ak.to_list(ak.ravel(event_level.PairedNextLeadingTau_h)),	
				#"LeadTau_ele": ak.to_list(ak.ravel(event_level.LeadingTau_ele)),	
				#"PairLeadTau_ele": ak.to_list(ak.ravel(event_level.PairedLeadingTau_ele)),	
				#"NextLeadTau_ele": ak.to_list(ak.ravel(event_level.NextLeadingTau_ele)),	
				#"PairNextLeadTau_ele": ak.to_list(ak.ravel(event_level.PairedNextLeadingTau_ele)),	
				#"LeadTau_mu": ak.to_list(ak.ravel(event_level.LeadingTau_mu)),	
				#"PairLeadTau_mu": ak.to_list(ak.ravel(event_level.PairedLeadingTau_mu)),	
				#"NextLeadTau_mu": ak.to_list(ak.ravel(event_level.NextLeadingTau_mu)),	
				#"PairNextLeadTau_mu": ak.to_list(ak.ravel(event_level.PairedNextLeadingTau_mu)),	
				#"Electron_tau_dR_Arr": electron.tau_min_dR,
				#"Muon_tau_dR_Arr": muon.tau_min_dR
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
	for i in range(5):
		for j in range(5):
			for k in range(5):
				if (i + j + k == 4):
					template_array_1.append(i)
					template_array_2.append(j)

	final_state_dict_signal = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),0)
	final_state_dict_data = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),0)
	final_state_dict_background = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),0)
	final_state_dict_data_full = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),[])
	final_state_dict_signal_full = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),[])
	final_state_dict_background_full = dict.fromkeys(fin_state_vec(template_array_1,template_array_2),[])
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
	background_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"	
	data_loc = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"
	
	#signal_base = "hdfs/store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/GluGluToRadionToHHTo4T_M-"
	#background_base = "hdfs/store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"	
	#data_loc = "hdfs/store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"
	
	#signal_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2/GluGluToRadionToHHTo4T_M-"
	#background_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2/"	
	#data_loc = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2/"


	iterative_runner = processor.Runner(
		executor = processor.IterativeExecutor(compression=None),
		schema=BaseSchema
	)
	#four_tau_hist_list = ["FourTau_Mass_Arr","HiggsDeltaPhi_Arr", "Higgs_DeltaR_Arr","leading_dR_Arr","subleading_dR_Arr","LeadingHiggs_mass","SubLeadingHiggs_mass", "radionPT_Arr", "tau_pt_Arr", 
	#		"tau_eta_Arr","ZMult_Arr", "BJet_Arr", "tau_lead_pt_Arr", "tau_sublead_pt_Arr", "tau_3rdlead_pt_Arr", "tau_4thlead_pt_Arr", "leading_dPhi_Arr", "subleading_dPhi_Arr", 
	#		"radionMET_dPhi_Arr","leadingHiggs_Rad_dR_Arr","subleadingHiggs_Rad_dR_Arr","leadingHiggs_MET_dPhi_Arr","subleadingHiggs_MET_dPhi_Arr","Radion_eta_Arr", "Radion_Charge_Arr"]
	four_tau_hist_list = ["FourTau_Mass_Arr","HiggsDeltaPhi_Arr", "Higgs_DeltaR_Arr","leading_dR_Arr","subleading_dR_Arr","LeadingHiggs_mass","SubLeadingHiggs_mass", "radionPT_Arr", 
			"ZMult_Arr", "BJet_Arr", "tau_lead_pt_Arr", "tau_sublead_pt_Arr", "tau_3rdlead_pt_Arr", "tau_4thlead_pt_Arr", "leading_dPhi_Arr", "subleading_dPhi_Arr", 
			"radionMET_dPhi_Arr","leadingHiggs_Rad_dR_Arr","subleadingHiggs_Rad_dR_Arr","leadingHiggs_MET_dPhi_Arr","subleadingHiggs_MET_dPhi_Arr","Radion_eta_Arr", "Radion_Charge_Arr",
			"LeadingHiggsSgn_Arr", "SubleadingHiggsSgn_Arr","Num_Electrons_Arr","Num_Muons_Arr","num_electron_tau_Arr","num_muon_tau_Arr"] #,"Electron_tau_dR_Arr","Muon_tau_dR_Arr"]
	#four_tau_hist_list = ["Num_Electrons_Arr","Num_Muons_Arr","Electron_tau_dR_Arr","Muon_tau_dR_Arr"]
	#four_tau_hist_list = ["ZMult_Arr","ZMult_ele_Arr","ZMult_mu_Arr", "ZMult_tau_Arr"]
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
			PUWeight = np.divide(HistoPUData, HistoPUMC)	


	#Loop over all mass points
	for mass in mass_str_arr:
		print("====================Radion Mass = " + mass[0] + "." + mass[1] + " TeV====================")
		file_dict_test = { #Reduced files to run over
			#"ZZ4l": [background_base + "ZZ4l.root"],
			"DYJetsToLL_Pt-50To100": [background_base + "DYJetsToLL_Pt-50To100.root"] ,
			"DYJetsToLL_Pt-100To250": [ background_base + "DYJetsToLL_Pt-100To250.root"], 
			"DYJetsToLL_Pt-250To400": [ background_base + "DYJetsToLL_Pt-250To400.root"], 
			"DYJetsToLL_Pt-400To650": [ background_base + "DYJetsToLL_Pt-400To650.root"], 
			"DYJetsToLL_Pt-650ToInf": [background_base + "DYJetsToLL_Pt-650ToInf.root"],
			"Signal": [signal_base + mass + ".root"],
			"Data_SingleMuon": [data_loc + "SingleMu_Run2018A.root"], #, data_loc + "SingleMu_Run2018B.root", data_loc + "SingleMu_Run2018C.root", data_loc + "SingleMu_Run2018D.root"],
			"Data_JetHT": [data_loc + "JetHT_Run2018A-17Sep2018-v1.root"] #, data_loc + "JetHT_Run2018B-17Sep2018-v1.root", data_loc + "JetHT_Run2018C-17Sep2018-v1.root",data_loc + "JetHT_Run2018D-PromptReco-v2.root"]
			#"Data_SingleMuon": [data_loc + "SingleMu_Run2018A.root", data_loc + "SingleMu_Run2018B.root", data_loc + "SingleMu_Run2018C.root", data_loc + "SingleMu_Run2018D.root"],
			#"Data_JetHT": [data_loc + "JetHT_Run2018A-17Sep2018-v1.root", data_loc + "JetHT_Run2018B-17Sep2018-v1.root", data_loc + "JetHT_Run2018C-17Sep2018-v1.root",data_loc + "JetHT_Run2018D-PromptReco-v2.root"]
  
        }

		file_dict_signal_only = {
		#file_dict = {
			"Signal": [signal_base + mass + ".root"]
		}
		
		#Grand Unified Background + Signal + Data Dictionary links file name to location of root file
		file_dict = {
			"TTToSemiLeptonic": [background_base + "TTToSemiLeptonic.root"], "TTTo2L2Nu": [background_base + "TTTo2L2Nu.root"], "TTToHadronic": [background_base + "TTToHadronic.root"],
			"ZZ4l": [background_base + "ZZ4l.root"],  
			"VV2l2nu" : [background_base + "VV2l2nu.root"], 
			"WZ1l3nu" : [background_base + "WZ1l3nu.root"], 
			"WZ3l1nu" : [background_base + "WZ3l1nu.root"], 
			"ZZ2l2q" : [background_base + "ZZ2l2q.root"], 
			"WZ2l2q" : [background_base + "WZ2l2q.root"], 
			"WZ1l1nu2q" : [background_base + "WZ1l1nu2q.root"],
			"DYJetsToLL_Pt-50To100": [background_base + "DYJetsToLL_Pt-50To100.root"] ,
			"DYJetsToLL_Pt-100To250": [ background_base + "DYJetsToLL_Pt-100To250.root"], 
			"DYJetsToLL_Pt-250To400": [ background_base + "DYJetsToLL_Pt-250To400.root"], 
			"DYJetsToLL_Pt-400To650": [ background_base + "DYJetsToLL_Pt-400To650.root"], 
			"DYJetsToLL_Pt-650ToInf": [background_base + "DYJetsToLL_Pt-650ToInf.root"],
			"Tbar-tchan" : [background_base + "Tbar-tchan.root"], 
			"T-tchan" : [background_base + "T-tchan.root"], 
			"Tbar-tW" : [background_base + "Tbar-tW.root"], 
			"T-tW" : [background_base + "T-tW.root"],
			"WJetsToLNu_HT-100To200" : [background_base + "WJetsToLNu_HT-100To200.root"],
			"WJetsToLNu_HT-200To400" : [background_base + "WJetsToLNu_HT-200To400.root"], 
			"WJetsToLNu_HT-400To600" : [background_base + "WJetsToLNu_HT-400To600.root"], 
			"WJetsToLNu_HT-600To800" : [background_base + "WJetsToLNu_HT-600To800.root"],
			"WJetsToLNu_HT-800To1200" : [background_base + "WJetsToLNu_HT-800To1200.root"],
			"WJetsToLNu_HT-1200To2500" : [background_base + "WJetsToLNu_HT-1200To2500.root"],
			"WJetsToLNu_HT-2500ToInf" : [background_base + "WJetsToLNu_HT-2500ToInf.root"],
			"Signal": [signal_base + mass + ".root"],
			"Data_SingleMuon": [data_loc + "SingleMu_Run2018A.root", data_loc + "SingleMu_Run2018B.root", data_loc + "SingleMu_Run2018C.root", data_loc + "SingleMu_Run2018D.root"],
			"Data_JetHT": [data_loc + "JetHT_Run2018A-17Sep2018-v1.root", data_loc + "JetHT_Run2018B-17Sep2018-v1.root", data_loc + "JetHT_Run2018C-17Sep2018-v1.root",data_loc + "JetHT_Run2018D-PromptReco-v2.root"]
		}
		
		#Generate dictionary of number of processed events
		for key_name, file_name in file_dict.items(): 
			if (file_name != "Data_JetHT" or file_name != "Data_SingleMuon"):
				tempFile = uproot.open(file_name[0]) #Get file
				#numEvents_Dict[key_name] = tempFile['hEvents'].member('fEntries')/2
				numEvents_Dict[key_name] = tempFile['hcount'].member('fEntries')/2
			else: #Ignore data files
				continue

		
		background_list = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"]
		#background_list = ["Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"]
		#background_list = [r"$ZZ \rightarrow 4l$"]
		#background_list = [r"Drell-Yan+Jets"]
		signal_list = [r"MC Sample $m_\phi$ = %s TeV"%mass[0]]
		background_plot_names = {r"$t\bar{t}$" : "_ttbar_", r"Drell-Yan+Jets": "_DYJets_", "Di-Bosons" : "_DiBosons_", "Single Top": "_SingleTop+", "QCD" : "_QCD_", "W+Jets" : "_WJets_", r"$ZZ \rightarrow 4l$" : "_ZZ4l_"} #For file names
		background_dict = {r"$t\bar{t}$" : ["TTToSemiLeptonic","TTTo2L2Nu","TTToHadronic"], 
				r"Drell-Yan+Jets": ["DYJetsToLL_Pt-50To100","DYJetsToLL_Pt-100To250","DYJetsToLL_Pt-250To400","DYJetsToLL_Pt-400To650","DYJetsToLL_Pt-650ToInf"], 
				"Di-Bosons": ["WZ3l1nu","WZ2l2q","WZ1l1nu2q","ZZ2l2q", "WZ1l3nu", "VV2l2nu"], "Single Top": ["Tbar-tchan","T-tchan","Tbar-tW","T-tW"], 
				"W+Jets": ["WJetsToLNu_HT-100To200","WJetsToLNu_HT-200To400","WJetsToLNu_HT-400To600","WJetsToLNu_HT-600To800","WJetsToLNu_HT-800To1200","WJetsToLNu_HT-1200To2500","WJetsToLNu_HT-2500ToInf"],
				r"$ZZ \rightarrow 4l$" : ["ZZ4l"]
		}
		#background_dict = {r"$ZZ \rightarrow 4l$" : ["ZZ4l"]}
		#background_dict = {r"Drell-Yan+Jets" : ["DYJetsToLL_Pt-50To100","DYJetsToLL_Pt-100To250","DYJetsToLL_Pt-250To400","DYJetsToLL_Pt-400To650","DYJetsToLL_Pt-650ToInf"]}

		for trigger_name, trigger_pair in trigger_dict.items(): #Run over all triggers/combinations of interest
			#Histogram bining
			if (trigger_pair[0] == 39): #Use reduced binning for JetHT trigger
				N1 = 6 
				N2 = 6 
			else:
				N1 = 10 
				N2 = 8 
			
			#Dictionaries of histograms for background, signal and data
			hist_dict_background = {
					"FourTau_Mass_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Regular(N1,0,3000, label = r"$m_{4\tau}$ [GeV]").Double(), 
					"HiggsDeltaPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Higgs $\Delta \phi$").Double(), 
				"Higgs_DeltaR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,5, label = r"Higgs $\Delta$R").Double(), 
				"leading_dR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(), 
				"subleading_dR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double(), 
				"LeadingHiggs_mass" : hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N2,0,120, label=r"Leading di-$\tau$ Mass (GeV)").Double(), 
				"SubLeadingHiggs_mass" : hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N2,0,120, label=r"Sub-Leading di-$\tau$ Mass (GeV)").Double(), 
				"radionPT_Arr" : hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,500, label=r"Radion $p_T$ (GeV)").Double(), 
				"tau_pt_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,400, label=r"$\tau$ $p_T$ (GeV)").Double(),
				"tau_eta_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-5,5, label = r"$\tau \ \eta$").Double(),
				"ZMult_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(6,0,6, label = r"Z Boson Multiplicity").Double(),
				"ZMult_ele_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(6,0,6, label = r"Z Boson Multiplicity (electrons only)").Double(),
				"ZMult_mu_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(6,0,6, label = r"Z Boson Multiplicity (muons only)").Double(),
				"ZMult_tau_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(6,0,6, label = r"Z Boson Multiplicity (from taus)").Double(),
				"BJet_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(6,0,6, label = r"B Jet Multiplicity").Double(),
				"tau_lead_pt_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,400, label=r"Leading $\tau$ $p_T$ (GeV)").Double(),
				"tau_sublead_pt_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,400, label=r"Subleading $\tau$ $p_T$ (GeV)").Double(),
				"tau_3rdlead_pt_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,400, label=r"Third Leading $\tau$ $p_T$ (GeV)").Double(),
				"tau_4thlead_pt_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,400, label=r"Fourth Leading $\tau$ $p_T$ (GeV)").Double(),
				
				"leading_dPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(), 
				"subleading_dPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(), 
				"radionMET_dPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(), 
				"leadingHiggs_Rad_dR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
				"subleadingHiggs_Rad_dR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
				"leadingHiggs_MET_dPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Leading Higgs MET $\Delta \phi$").Double(), 
				"subleadingHiggs_MET_dPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta \phi$").Double(),
				"Radion_eta_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-5,5, label = r"Radion $\eta$").Double(),
				"Radion_Charge_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name = "background").Reg(10,-5,5,label = r"Radion Electric Charge").Double(),
				"LeadingHiggsSgn_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name = "background").Reg(8,-4,4,label = r"Leading Higgs Electric Charge").Double(),
				"SubleadingHiggsSgn_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name = "background").Reg(8,-4,4,label = r"Subleading Higgs Electric Charge").Double(),
				"Num_Electrons_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name = "background").Reg(8,0,8,label = r"number of electrons").Double(),
				"Num_Muons_Arr" : hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name = "background").Reg(8,0,8,label = r"number of muons").Double(),
				"Electron_tau_dR_Arr" : hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name = "background").Reg(N1,0,1,label = r"Minimized tau to electron").Double(),
				"Muon_tau_dR_Arr" : hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name = "background").Reg(N1,0,1,label = r"Minimized tau to muon").Double(),
				"num_electron_tau_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name = "background").Reg(5,0,5,label=r"Number of electrons identified as taus").Double(),
				"num_muon_tau_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name = "background").Reg(5,0,5,label=r"Number of muon identified as taus").Double(),

			}
			
			hist_dict_signal = {
				"FourTau_Mass_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Regular(N1,0,3000, label = r"$m_{4\tau}$ [GeV]").Double(),
				"HiggsDeltaPhi_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Higgs $\Delta \phi$").Double(), 
				"Higgs_DeltaR_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,5, label = r"Higgs $\Delta$R").Double(),
				"leading_dR_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
				"subleading_dR_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double(),
				"LeadingHiggs_mass" : hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N2,0,120, label=r"Leading di-$\tau$ Mass (GeV)").Double(),
				"SubLeadingHiggs_mass" : hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N2,0,120, label=r"Sub-Leading di-$\tau$ Mass (GeV)").Double(),
				"radionPT_Arr" : hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,500, label=r"Radion $p_T$ (GeV)").Double(),
				"tau_pt_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,400, label=r"$\tau$ $p_T$ (GeV)").Double(),
				"tau_eta_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-5,5, label = r"$\tau \ \eta$").Double(),
				"ZMult_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(6,0,6, label = r"Z Boson Multiplicity").Double(),
				"ZMult_ele_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(6,0,6, label = r"Z Boson Multiplicity (electrons only)").Double(),
				"ZMult_mu_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(6,0,6, label = r"Z Boson Multiplicity (muons only)").Double(),
				"ZMult_tau_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(6,0,6, label = r"Z Boson Multiplicity (from taus)").Double(),
				"BJet_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(6,0,6, label = r"B Jet Multiplicity").Double(),
				"tau_lead_pt_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,400, label=r"Leading $\tau$ $p_T$ (GeV)").Double(),
				"tau_sublead_pt_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,400, label=r"Subleading $\tau$ $p_T$ (GeV)").Double(),
				"tau_3rdlead_pt_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,400, label=r"Third leading $\tau$ $p_T$ (GeV)").Double(),
				"tau_4thlead_pt_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,400, label=r"Fourth leading $\tau$ $p_T$ (GeV)").Double(),
				"leading_dPhi_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(), 
				"subleading_dPhi_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(), 
				"radionMET_dPhi_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(), 
				"leadingHiggs_Rad_dR_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
				"subleadingHiggs_Rad_dR_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
				"leadingHiggs_MET_dPhi_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Leading Higgs MET $\Delta \phi$").Double(), 
				"subleadingHiggs_MET_dPhi_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta \phi$").Double(),
				"Radion_eta_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-5,5, label = r"Radion $\eta$").Double(),
				"Radion_Charge_Arr": hist.Hist.new.StrCat(["Signal"],name = "signal").Reg(10,-5,5,label = r"Radion Electric Charge").Double(),
				"LeadingHiggsSgn_Arr": hist.Hist.new.StrCat(["Signal"],name = "signal").Reg(8,-4,4,label = r"Leading Higgs Electric Charge").Double(),
				"SubleadingHiggsSgn_Arr": hist.Hist.new.StrCat(["Signal"],name = "signal").Reg(8,-4,4,label = r"Subleading Higgs Electric Charge").Double(),
				"Num_Electrons_Arr": hist.Hist.new.StrCat(["Signal"],name = "signal").Reg(8,0,8,label = r"number of electrons").Double(),
				"Num_Muons_Arr" : hist.Hist.new.StrCat(["Signal"],name = "signal").Reg(8,0,8,label = r"number of muons").Double(),
				"Electron_tau_dR_Arr" : hist.Hist.new.StrCat(["Signal"],name = "signal").Reg(N1,0,1,label = r"Minimized tau to electron").Double(),
				"Muon_tau_dR_Arr" : hist.Hist.new.StrCat(["Signal"],name = "signal").Reg(N1,0,1,label = r"Minimized tau to muon").Double(),
				"num_electron_tau_Arr": hist.Hist.new.StrCat(["Signal"],name = "signal").Reg(5,0,5,label=r"Number of electrons identified as taus").Double(),
				"num_muon_tau_Arr": hist.Hist.new.StrCat(["Signal"],name = "signal").Reg(5,0,5,label=r"Number of muon identified as taus").Double(),


			}
			hist_dict_data = {
				"FourTau_Mass_Arr": hist.Hist.new.StrCat(["Data"],name="data").Regular(N1,0,3000, label = r"$m_{4\tau}$ [GeV]").Double(),
				"HiggsDeltaPhi_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Higgs $\Delta \phi$").Double(), 
				"Higgs_DeltaR_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,5, label = r"Higgs $\Delta$R").Double(),
				"leading_dR_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
				"subleading_dR_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double(),
				"LeadingHiggs_mass" : hist.Hist.new.StrCat(["Data"],name="data").Reg(N2,0,120, label=r"Leading di-$\tau$ Mass (GeV)").Double(),
				"SubLeadingHiggs_mass" : hist.Hist.new.StrCat(["Data"],name="data").Reg(N2,0,120, label=r"Sub-Leading di-$\tau$ Mass (GeV)").Double(),
				"radionPT_Arr" : hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,500, label=r"Radion $p_T$ (GeV)").Double(),
				"tau_pt_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,400, label=r"$\tau$ $p_T$ (GeV)").Double(),
				"tau_eta_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-5,5, label = r"$\tau \ \eta$").Double(),
				"ZMult_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(6,0,6, label = r"Z Boson Multiplicity").Double(),
				"ZMult_ele_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(6,0,6, label = r"Z Boson Multiplicity (electrons only)").Double(),
				"ZMult_mu_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(6,0,6, label = r"Z Boson Multiplicity (muons only)").Double(),
				"ZMult_tau_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(6,0,6, label = r"Z Boson Multiplicity (from taus)").Double(),
				"BJet_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(6,0,6, label = r"B Jet Multiplicity").Double(),
				"tau_lead_pt_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,400, label=r"Leading $\tau$ $p_T$ (GeV)").Double(),
				"tau_sublead_pt_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,400, label=r"Subleading $\tau$ $p_T$ (GeV)").Double(),
				"tau_3rdlead_pt_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,400, label=r"Third leading $\tau$ $p_T$ (GeV)").Double(),
				"tau_4thlead_pt_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,400, label=r"Fourth leading $\tau$ $p_T$ (GeV)").Double(),
				"leading_dPhi_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(), 
				"subleading_dPhi_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(), 
				"radionMET_dPhi_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(), 
				"leadingHiggs_Rad_dR_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
				"subleadingHiggs_Rad_dR_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
				"leadingHiggs_MET_dPhi_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Leading Higgs MET $\Delta \phi$").Double(), 
				"subleadingHiggs_MET_dPhi_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta \phi$").Double(),
				"Radion_eta_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-5,5, label = r"Radion $\eta$").Double(),
				"Radion_Charge_Arr": hist.Hist.new.StrCat(["Data"],name = "data").Reg(10,-5,5,label = r"Radion Electric Charge").Double(),
				"LeadingHiggsSgn_Arr": hist.Hist.new.StrCat(["Data"],name = "data").Reg(8,-4,4,label = r"Leading Higgs Electric Charge").Double(),
				"SubleadingHiggsSgn_Arr": hist.Hist.new.StrCat(["Data"],name = "data").Reg(8,-4,4,label = r"Subleading Higgs Electric Charge").Double(),
				"Num_Electrons_Arr": hist.Hist.new.StrCat(["Data"],name = "data").Reg(8,0,8,label = r"number of electrons").Double(),
				"Num_Muons_Arr" : hist.Hist.new.StrCat(["Data"],name = "data").Reg(8,0,8,label = r"number of muons").Double(),
				"Electron_tau_dR_Arr" : hist.Hist.new.StrCat(["Data"],name = "data").Reg(N1,0,1,label = r"Minimized tau to electron").Double(),
				"Muon_tau_dR_Arr" : hist.Hist.new.StrCat(["Data"],name = "data").Reg(N1,0,1,label = r"Minimized tau to muon").Double(),
				"num_electron_tau_Arr": hist.Hist.new.StrCat(["Data"],name = "data").Reg(5,0,5,label=r"Number of electrons identified as taus").Double(),
				"num_muon_tau_Arr": hist.Hist.new.StrCat(["Data"],name = "data").Reg(5,0,5,label=r"Number of muon identified as taus").Double(),
			}
				
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
				"num_muon_tau_Arr": "Num_muons_as_tau_Mass" + mass + "-" + trigger_name 

			}
			
			fourtau_out = iterative_runner(file_dict, treename="4tau_tree", processor_instance=FourTauPlotting(trigger_bit=trigger_pair[0], or_trigger=trigger_pair[1],PUWeights = PUWeight, PU_weight_bool =True, signal_mass = mass))
			for hist_name in four_tau_hist_list: #Loop over all histograms
				#fig,ax = plt.subplots()
				fig0,ax0 = plt.subplots()
				if (hist_name != "Pair_DeltaPhi_Hist" and hist_name != "RadionPTComp_Hist"):
					hist_dict_only_signal = {
						"FourTau_Mass_Arr": hist.Hist.new.Regular(N1,0,3000, label = r"$m_{4\tau}$ [GeV]").Double(),
						"HiggsDeltaPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Higgs $\Delta \phi$").Double(), 
						"Higgs_DeltaR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Higgs $\Delta$R").Double(),
						"leading_dR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
						"subleading_dR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double(),
						"LeadingHiggs_mass" : hist.Hist.new.Regular(N2,0,120, label=r"Leading Higgs Mass (GeV)").Double(),
						"SubLeadingHiggs_mass" : hist.Hist.new.Regular(N2,0,120, label=r"Sub-Leading Higgs Mass (GeV)").Double(),
						"radionPT_Arr" : hist.Hist.new.Regular(N1,0,500, label=r"Radion $p_T$ (GeV)").Double(),
						"tau_pt_Arr": hist.Hist.new.Regular(N1,0,400, label=r"$\tau$ $p_T$ (GeV)").Double(),
						"tau_eta_Arr": hist.Hist.new.Regular(N1,-5,5, label = r"$\tau \ \eta$").Double(),
						"ZMult_Arr": hist.Hist.new.Regular(6,0,6, label = r"Z Boson Multiplicity").Double(),
						"ZMult_ele_Arr": hist.Hist.new.Regular(6,0,6, label = r"Z Boson Multiplicity (electrons only)").Double(),
						"ZMult_mu_Arr": hist.Hist.new.Regular(6,0,6, label = r"Z Boson Multiplicity (muons only)").Double(),
						"ZMult_tau_Arr": hist.Hist.new.Regular(6,0,6, label = r"Z Boson Multiplicity (from taus)").Double(),
						"BJet_Arr": hist.Hist.new.Regular(6,0,6, label = r"BJet Multiplicity").Double(),
						"tau_lead_pt_Arr" : hist.Hist.new.Regular(N1,0,400, label=r"Leading $\tau$ $p_T$ (GeV)").Double(),
						"tau_sublead_pt_Arr" : hist.Hist.new.Regular(N1,0,400, label=r"Subleading $\tau$ $p_T$ (GeV)").Double(),
						"tau_3rdlead_pt_Arr" : hist.Hist.new.Regular(N1,0,400, label=r"Third leading $\tau$ $p_T$ (GeV)").Double(),
						"tau_4thlead_pt_Arr" : hist.Hist.new.Regular(N1,0,400, label=r"Fourth leading $\tau$ $p_T$ (GeV)").Double(),
						"leading_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(), 
						"subleading_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(), 
						"radionMET_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(), 
						"leadingHiggs_Rad_dR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
						"subleadingHiggs_Rad_dR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
						"leadingHiggs_MET_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Leading Higgs MET $\Delta \phi$").Double(), 
						"subleadingHiggs_MET_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta \phi$").Double(),
						"Radion_eta_Arr": hist.Hist.new.Regular(N1,-5,5, label = r"Radion $\eta$").Double(),
						"Radion_Charge_Arr": hist.Hist.new.Regular(10,-5,5,label = r"Radion Electric Charge").Double(),
						"LeadingHiggsSgn_Arr": hist.Hist.new.Regular(8,-4,4,label = r"Leading Higgs Electric Charge").Double(),
						"SubleadingHiggsSgn_Arr": hist.Hist.new.Regular(8,-4,4,label = r"Subleading Higgs Electric Charge").Double(),
						"Num_Electrons_Arr": hist.Hist.new.Regular(8,0,8,label = r"number of electrons").Double(),
						"Num_Muons_Arr" : hist.Hist.new.Regular(8,0,8,label = r"number of muons").Double(),
						"Electron_tau_dR_Arr" : hist.Hist.new.Regular(N1,0,1,label = r"Minimized tau to electron $\Delta$R").Double(),
						"Muon_tau_dR_Arr" : hist.Hist.new.Regular(N1,0,1,label = r"Minimized tau to muon $\Delta$R").Double(),
						"num_electron_tau_Arr": hist.Hist.new.Regular(5,0,5,label=r"Number of electrons identified as taus").Double(),
						"num_muon_tau_Arr": hist.Hist.new.Regular(5,0,5,label=r"Number of muon identified as taus").Double(),
					}
					if (hist_name == "Electron_tau_dR_Arr" or hist_name == "Muon_tau_dR_Arr"):
						#print(fourtau_out["Signal"][hist_name])
						#Drop events with no leptons
						fill_Arr = ak.from_iter(fourtau_out["Signal"][hist_name])
						fill_Arr = fill_Arr[fill_Arr != 999]
						#hist_dict_only_signal[hist_name].fill(ak.ravel(fourtau_out["Signal"][hist_name]))
						hist_dict_only_signal[hist_name].fill(fill_Arr)
						hist_dict_only_signal[hist_name].plot1d(ax=ax0)
						ax0.set_yscale('log')
					else:
						hist_dict_only_signal[hist_name].fill(fourtau_out["Signal"][hist_name],weight = fourtau_out["Signal"]["Weight"])
						hist_dict_only_signal[hist_name].plot1d(ax=ax0)
					plt.title("Signal Mass: " + mass[0] + "TeV")
					plt.savefig("SignalSingle" + four_tau_names[hist_name])
					plt.close()

				if (hist_name != "Pair_DeltaPhi_Hist" and hist_name != "RadionPTComp_Hist"):
					for background_type in background_list:
						hist_dict_single_background = {
							"FourTau_Mass_Arr": hist.Hist.new.Regular(N1,0,3000, label = r"$m_{4\tau}$ [GeV]").Double(),
							"HiggsDeltaPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Higgs $\Delta \phi$").Double(), 
							"Higgs_DeltaR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Higgs $\Delta$R").Double(),
							"leading_dR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
							"subleading_dR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double(),
							"LeadingHiggs_mass" : hist.Hist.new.Regular(N2,0,120, label=r"Leading Higgs Mass (GeV)").Double(),
							"SubLeadingHiggs_mass" : hist.Hist.new.Regular(N2,0,120, label=r"Sub-Leading Higgs Mass (GeV)").Double(),
							"radionPT_Arr" : hist.Hist.new.Regular(N1,0,200, label=r"Radion $p_T$ (GeV)").Double(),
							"tau_pt_Arr": hist.Hist.new.Regular(N1,0,400, label=r"$\tau$ $p_T$ (GeV)").Double(),
							"tau_eta_Arr": hist.Hist.new.Regular(N1,-5,5, label = r"$\tau \ \eta$").Double(),
							"ZMult_Arr": hist.Hist.new.Regular(6,0,6, label = r"Z Boson Multiplicity").Double(),
							"ZMult_ele_Arr": hist.Hist.new.Regular(6,0,6, label = r"Z Boson Multiplicity (electrons only)").Double(),
							"ZMult_mu_Arr": hist.Hist.new.Regular(6,0,6, label = r"Z Boson Multiplicity (muons only)").Double(),
							"ZMult_tau_Arr": hist.Hist.new.Regular(6,0,6, label = r"Z Boson Multiplicity (from taus)").Double(),
							"BJet_Arr": hist.Hist.new.Regular(6,0,6, label = r"BJet Multiplicity").Double(),
							"tau_lead_pt_Arr" : hist.Hist.new.Regular(N1,0,200, label=r"Leading $\tau$ $p_T$ (GeV)").Double(),
							"tau_sublead_pt_Arr" : hist.Hist.new.Regular(N1,0,400, label=r"Subleading $\tau$ $p_T$ (GeV)").Double(),
							"tau_3rdlead_pt_Arr" : hist.Hist.new.Regular(N1,0,400, label=r"Third leading $\tau$ $p_T$ (GeV)").Double(),
							"tau_4thlead_pt_Arr" : hist.Hist.new.Regular(N1,0,400, label=r"Fourth leading $\tau$ $p_T$ (GeV)").Double(),
							"leading_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(), 
							"subleading_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(), 
							"radionMET_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(), 
							"leadingHiggs_Rad_dR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
							"subleadingHiggs_Rad_dR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
							"leadingHiggs_MET_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Leading Higgs MET $\Delta \phi$").Double(), 
							"subleadingHiggs_MET_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta \phi$").Double(),
							"Radion_eta_Arr": hist.Hist.new.Regular(N1,-5,5, label = r"Radion $\eta$").Double(),
							"Radion_Charge_Arr": hist.Hist.new.Regular(10,-5,5,label = r"Radion Electric Charge").Double(),
							"LeadingHiggsSgn_Arr": hist.Hist.new.Regular(8,-4,4,label = r"Leading Higgs Electric Charge").Double(),
							"SubleadingHiggsSgn_Arr": hist.Hist.new.Regular(8,-4,4,label = r"Subleading Higgs Electric Charge").Double(),
							"Num_Electrons_Arr": hist.Hist.new.Regular(8,0,8,label = r"number of electrons").Double(),
							"Num_Muons_Arr" : hist.Hist.new.Regular(8,0,8,label = r"number of muons").Double(),
							"Electron_tau_dR_Arr" : hist.Hist.new.Regular(N1,0,1,label = r"Minimized tau to electron $\Delta$R").Double(),
							"Muon_tau_dR_Arr" : hist.Hist.new.Regular(N1,0,1,label = r"Minimized tau to muon $\Delta$R").Double(),
							"num_electron_tau_Arr": hist.Hist.new.Regular(5,0,5,label=r"Number of electrons identified as taus").Double(),
							"num_muon_tau_Arr": hist.Hist.new.Regular(5,0,5,label=r"Number of muon identified as taus").Double(),
						}
						background_array = []
						backgrounds = background_dict[background_type]
						
						#Loop over all backgrounds
						for background in backgrounds:
							if (mass == "2000"): #Only need to generate single background once
								if (hist_name == "Radion_Charge_Arr"):
									lumi_table_data["MC Sample"].append(background)
									lumi_table_data["Luminosity"].append(fourtau_out[background]["Lumi_Val"])
									lumi_table_data["Cross Section (pb)"].append(fourtau_out[background]["CrossSec_Val"])
									lumi_table_data["Number of Events"].append(fourtau_out[background]["NEvent_Val"])
									lumi_table_data["Calculated Weight"].append(fourtau_out[background]["Weight_Val"])
								
								fig2, ax2 = plt.subplots()
								if (hist_name != "Electron_tau_dR_Arr" and hist_name != "Muon_tau_dR_Arr"):
									hist_dict_single_background[hist_name].fill(fourtau_out[background][hist_name],weight = fourtau_out[background]["Weight"]) #Obtain background distributions 
									hist_dict_single_background[hist_name].plot1d(ax=ax2)
									plt.title(background_type)
									plt.savefig("SingleBackground" + background_plot_names[background_type] + four_tau_names[hist_name])
									plt.close()
								else: #lepton-tau delta R 
									#hist_dict_single_background[hist_name].fill(fourtau_out[background][hist_name]) #Obtain background distributions 
									fill_Arr = ak.from_iter(fourtau_out[background][hist_name])
									fill_Arr = fill_Arr[fill_Arr != 999]
									hist_dict_single_background[hist_name].fill(fill_Arr) #Obtain background distributions 
									hist_dict_single_background[hist_name].plot1d(ax=ax2)
									ax2.set_yscale('log')
									plt.title(background_type)
									plt.savefig("SingleBackground" + background_plot_names[background_type] + four_tau_names[hist_name])
									plt.close()
								#if ((background_type == r"$t\bar{t}$" or background_type == "Drell-Yan+Jets" or background_type == r"$ZZ \rightarrow 4l$") and hist_name == "LeadingHiggs_mass"):
									#print("!!!====================Debugging di-tau masses of %s====================!!!"%background_type)
									#zeroMassCounter = 0
									#firstbin_count = 0
									#for ditau_mass in fourtau_out[background][hist_name]:
									#	if (ditau_mass == 0):
									#		zeroMassCounter += 1
									#	if (ditau_mass >= 0 and ditau_mass <= 10):
									#		firstbin_count += 1
									#		print(ditau_mass)
									#print("# of entries with 0 mass: %d"%zeroMassCounter)
									#print("# of entries in first bin: %d"%firstbin_count)
							
							#Could there be issues here in terms of how the backgrounds are being combined???
							if (hist_name != "Electron_tau_dR_Arr"): # and hist_name != "Muon_tau_dR_Arr"): #Skip the lepton-tau delta R
								#print(fourtau_out[background]["Weight"])
								hist_dict_background[hist_name].fill(background_type,fourtau_out[background][hist_name],weight = fourtau_out[background]["Weight"]) #Obtain background distributions
								print("Background %s added"%background)
							if (hist_name == "num_electron_tau_Arr"): # and np.pi == np.exp(1)): #Count final states
								background_state_array += fin_state_vec(fourtau_out[background]["num_electron_tau_Arr"],fourtau_out[background]["num_muon_tau_Arr"]).tolist()


							
					
					if (hist_name != "Electron_tau_dR_Arr" and hist_name != "Muon_tau_dR_Arr"): #Skip the lepton-tau delta R 
						hist_dict_signal[hist_name].fill("Signal",fourtau_out["Signal"][hist_name],weight = fourtau_out["Signal"]["Weight"]) #Obtain signal distribution
						if (hist_name == "num_electron_tau_Arr"): # and np.pi == np.exp(1)): #Count final states
							final_state_array = fin_state_vec(fourtau_out["Signal"]["num_electron_tau_Arr"],fourtau_out["Signal"]["num_muon_tau_Arr"])
							for state in final_state_array:
								final_state_dict_signal[state] += 1/len(fourtau_out["Signal"]["num_electron_tau_Arr"])
                        

					
						#Obtain data distributions
						print("==================Hist %s================"%hist_name)
						#print("Total amount of data = %d"%(len(fourtau_out["Data_SingleMuon"][hist_name]) + len(fourtau_out["Data_JetHT"][hist_name])))
						#print("Total amount of data = %d"%(len(fourtau_out["Data_SingleMuon"][hist_name])))
						#print("Total amount of data = %d"%(len(fourtau_out["Data_JetHT"][hist_name])))
						if (trigger_name == "Mu50"):
							print("Mu50 Only")
							hist_dict_data[hist_name].fill("Data",fourtau_out["Data_SingleMuon"][hist_name]) 
						if (trigger_name == "PFHT500_PFMET100_PFMHT100_IDTight"):
							print("JetHTMHTMET Only")
							hist_dict_data[hist_name].fill("Data",fourtau_out["Data_JetHT"][hist_name]) 
						if (trigger_name == "EitherOr_Trigger"):
							print("Both Triggers")
							hist_dict_data[hist_name].fill("Data",fourtau_out["Data_SingleMuon"][hist_name]) 
							hist_dict_data[hist_name].fill("Data",fourtau_out["Data_JetHT"][hist_name]) 
							
							if (hist_name == "num_electron_tau_Arr"): # and np.pi == np.exp(1)):
							    final_state_array_Mu = fin_state_vec(fourtau_out["Data_SingleMuon"]["num_electron_tau_Arr"],fourtau_out["Data_SingleMuon"]["num_muon_tau_Arr"])
							    final_state_array_Jet = fin_state_vec(fourtau_out["Data_JetHT"]["num_electron_tau_Arr"],fourtau_out["Data_JetHT"]["num_muon_tau_Arr"])
							    for state in final_state_array_Mu:
								    final_state_dict_data[state] += 1/(len(fourtau_out["Data_SingleMuon"]["num_electron_tau_Arr"]) + len(fourtau_out["Data_JetHT"]["num_electron_tau_Arr"]))
							    for state in final_state_array_Jet:
								    final_state_dict_data[state] += 1/(len(fourtau_out["Data_SingleMuon"]["num_electron_tau_Arr"]) + len(fourtau_out["Data_JetHT"]["num_electron_tau_Arr"]))

						#print("Number of Jet HT entries: %d"%len(fourtau_out["Data_JetHT"][hist_name]))
					
						#Put histograms into stacks and arrays for plotting purposes (is the issue arising here??)
						background_stack = hist_dict_background[hist_name].stack("background")
						signal_stack = hist_dict_signal[hist_name].stack("signal")
						data_stack = hist_dict_data[hist_name].stack("data")
						signal_array = [signal_stack["Signal"]]
						data_array = [data_stack["Data"]]
						for background in background_list:
							background_array.append(background_stack[background])

						if (hist_name == "Radion_Charge_Arr"):
							print("Background Histogram Sum: %f"%hist_dict_background[hist_name].sum())	
							print("Data Histogram Sum: %f"%hist_dict_data[hist_name].sum())
					
						#Stack background distributions and plot signal + data distribution
						fig,ax = plt.subplots()
						hep.histplot(background_array,ax=ax,stack=True,histtype="fill",label=background_list,facecolor=TABLEAU_COLORS[:len(background_list)],edgecolor=TABLEAU_COLORS[:len(background_list)])
						hep.histplot(signal_array,ax=ax,stack=True,histtype="step",label=signal_list,edgecolor=TABLEAU_COLORS[len(background_list)+1],linewidth=2.95)
						hep.histplot(data_array,ax=ax,stack=False,histtype="errorbar", yerr=True,label=["Data"],marker="o",color = "k") #,facecolor='black',edgecolor='black') #,mec='k')
						hep.cms.text("Preliminary",loc=0,fontsize=13)
						#ax.set_title(hist_name_dict[hist_name],loc = "right")
						ax.set_title("2018 Data",loc = "right")
						ax.legend(fontsize=10, loc='upper right')
						plt.savefig(four_tau_names[hist_name])
						plt.close()
	
	#Store final states in tables
	for state in background_state_array:
		final_state_dict_background[state] += 1/len(background_state_array)
	
	#for state in final_state_dict_signal_full:
	#	final_state_dict_signal_full[state].append(final_state_dict_signal[state])
	#for state in final_state_dict_data_full:
	#	final_state_dict_data_full[state].append(final_state_dict_data[state])
	#for state in final_state_dict_background_full:
	#	final_state_dict_background_full[state].append(final_state_dict_background[state])
    
	#print(final_state_dict_signal)	
	#print(final_state_dict_data)	
	#print(final_state_dict_background)
	print("Number of Signal events: %d"%len(fourtau_out["Signal"]["num_electron_tau_Arr"]))
	print("Number of Data events: %d"%(len(fourtau_out["Data_SingleMuon"]["num_electron_tau_Arr"]) + len(fourtau_out["Data_JetHT"]["num_electron_tau_Arr"])))
	print("Number of Background events: %d"%len(background_state_array))


	#Store information in tex table
	store_tau_states = False
	if (store_tau_states):
		file = open("Final_State_Table_Gen.tex","w")
		#file = open("Final_State_Table_Reco.tex","w")

		#Set up the tex document
		file.write("\\documentclass{article} \n")
		file.write("\\usepackage{multirow} \n")
		file.write("\\usepackage{multirow} \n")
		file.write("\\usepackage{lscape}\n")
		file.write("\\begin{document} \n")
		file.write("\\begin{landscape} \n")
		file.write("\\centering \n")

		#Set up the table
		file.write("\\begin{tabular}{|p{4.5cm}|p{3cm}|p{3cm}|p{3cm}|}")
		file.write("\\hline \n")
		file.write("\\multicolumn{4}{|c|}{Final State Table (Reco)} \\\\ \n")
		file.write("\\hline \n")
		file.write("4$\\tau$ Channel & 2 TeV Signal & Drell-Yan + Jets & Data \\\\ \n")
		file.write("\\hline \n")
		for state in final_state_dict_signal:
			file.write(state + " & %.3f"%final_state_dict_signal[state] + " & %.3f"%final_state_dict_background[state] + " & %.3f"%final_state_dict_data[state] + "\\\\")
			file.write("\n")
			file.write("\\hline \n")
		file.write("\\end{tabular} \n")
		file.write("\\end{landscape} \n")
		file.write("\\end{document}")
		file.close()
	
	#final_state_frame_data = pd.DataFrame(final_state_dict_data_full)
	#final_state_frame_signal = pd.DataFrame(final_state_dict_signal_full)
	#final_state_frame_background = pd.DataFrame(final_state_dict_background_full)
	
	#final_state_frame_data.to_csv("Final_State_Table_Data.csv",sep=",")
	#final_state_frame_signal.to_csv("Final_State_Table_Signal.csv",sep=",")
	#final_state_frame_background.to_csv("Final_State_Table_Background.csv",sep=",")

	#Store luminosity Weighting debugging table
	#fig, ax = plt.subplots()
	#fig.patch.set_visible(False)
	#ax.axis('off')
	#ax.axis('tight')
	#lumi_frame = pd.DataFrame(lumi_table_data)
	#lumi_frame.to_csv("Lumi_Weight_Table_Debugging.csv", sep='\t')
	#outTable = ax.table(cellText=lumi_frame.values, colLabels=lumi_frame.columns, loc='center', cellLoc='center')	
	
	
