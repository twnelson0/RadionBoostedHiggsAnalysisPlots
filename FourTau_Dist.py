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

def Z_Mult_Fun_Tau(taus):
	tau_plus = ak.mask(taus,taus.charge > 0)
	tau_minus = ak.mask(taus,taus.charge < 0)

	#Get number of taus and anti-taus
	neg_charge = taus.charge[taus.charge < 0]
	num_minus = ak.num(neg_charge,axis=1) 
	pos_charge = taus.charge[taus.charge > 0]
	num_plus = ak.num(pos_charge,axis=1) 

	#niave_pairing = ak.cartesian([lead_tau,tau_plus], axis=1, nested = False)

	print("Maxiumum theoretical taus: %d"%ak.max(ak.num(tau_plus,axis=1)))
	max_pairs = ak.max(ak.num(tau_plus,axis=1)) #Maximum number of possible pairings (niave)
	max_pairs = 2
	pair_names = pairing_list(max_pairs)
	
	#Drop Nones
	tau_minus.pt = ak.fill_none(tau_minus.pt,-10)
	tau_minus = tau_minus[tau_minus.pt != -10]
	tau_plus.pt = ak.fill_none(tau_plus.pt,-10)
	tau_plus = tau_plus[tau_plus.pt != -10]
	lower_limit = 80
	upper_limit = 100

	
	#Obtain the masses
	lead_tau = ak.firsts(tau_minus)
	leadE = lead_tau.E
	leadPx = lead_tau.Px
	leadPy = lead_tau.Py
	leadPz = lead_tau.Pz
	tau1,tau2 = ak.unzip(ak.cartesian([lead_tau,tau_plus], axis=1, nested = False))
	Pair_Mass = di_mass(tau1,tau2) #Get leading tau other tau invariant masses
	Pair_Mass = ak.fill_none(Pair_Mass,-10)
	tau_plus["first_pair"] = Pair_Mass
	First_Pair = ak.fill_none(Pair_Mass,-10,axis=1)
	First_Pair = First_Pair[First_Pair != -10]
	Z_Lower = Pair_Mass > 80
	Z_Upper = Pair_Mass < 100
	Z_Candidates = ak.mask(Pair_Mass,np.bitwise_and(Z_Lower,Z_Upper))
	Z_Candidates = ak.fill_none(Z_Candidates,-10,axis=1)
	tau_pair = ak.mask(tau_plus,tau_plus.first_pair == ak.firsts(Z_Candidates[Z_Candidates != -10]))
	#tau_pair.E = ak.fill_none(tau_pair)
	firstMass = ak.firsts(Z_Candidates)
	firstPairE = tau_pair.E
	firstPairPx = tau_pair.Px
	firstPairPy = tau_pair.Py
	firstPairPz = tau_pair.Pz
	#tau_plus = tau_plus[tau_plus.pt != ak.all(tau_pair.pt,axis=1)] #This line is causing problems
	tau_plus = tau_plus[tau_plus.E != ak.all(ak.fill_none(tau_pair.E,[-10],axis=0),axis=1)] #This line is an attempted fix 

	#Get second possible pair
	sublead_tau = tau_minus[tau_minus.pt != ak.firsts(tau_minus.pt)] #Remove leading tau from consideration
	sublead_tau = ak.firsts(sublead_tau)
	subleadE = sublead_tau.E
	subleadPx = sublead_tau.Px
	subleadPy = sublead_tau.Py
	subleadPz = sublead_tau.Pz
	tau1,tau2 = ak.unzip(ak.cartesian([sublead_tau,tau_plus], axis=1, nested = False))
	Pair_Mass = di_mass(tau1,tau2) #Get leading tau other tau invariant masses
	Pair_Mass = ak.fill_none(Pair_Mass,-10)
	tau_plus["second_pair"] = Pair_Mass
	Second_Pair = ak.fill_none(Pair_Mass,-10,axis=1)
	Second_Pair = Second_Pair[Second_Pair != -10]
	Z_Lower = Pair_Mass > 80
	Z_Upper = Pair_Mass < 100
	Z_Candidates = ak.mask(Pair_Mass,np.bitwise_and(Z_Lower,Z_Upper))
	Z_Candidates = ak.fill_none(Z_Candidates,-10,axis=1)
	secondMass = ak.firsts(Z_Candidates)
	tau_secondpair = ak.mask(tau_plus,tau_plus.second_pair == ak.firsts(Z_Candidates[Z_Candidates != -10]))
	secondPairE = tau_secondpair.E
	secondPairPx = tau_secondpair.Px
	secondPairPy = tau_secondpair.Py
	secondPairPz = tau_secondpair.Pz


	#Obtian Z-multiplicity
	#Drop none/null values
	tau_pair.first_pair = ak.fill_none(tau_pair.first_pair,-10,axis=1)
	tau_pair.first_pair = ak.fill_none(tau_pair.first_pair,[-10],axis=0)
	tau_secondpair.second_pair = ak.fill_none(tau_secondpair.second_pair,-10,axis=1)
	tau_secondpair.second_pair = ak.fill_none(tau_secondpair.second_pair,[-10],axis=0)
	tau_pair.first_pair = tau_pair.first_pair[tau_pair.first_pair != -10]
	tau_secondpair.second_pair = tau_secondpair.second_pair[tau_secondpair.second_pair != -10]
	
	ZMult = ak.num(tau_pair.first_pair,axis=1)
	#print(ZMult)
	print(ak.num(tau_secondpair.second_pair,axis=1))
	ZMult = ZMult + ak.num(tau_secondpair.second_pair,axis=1)
	
	#Print Stuff out
	#dummyIndex = 10
	for i in range(10):
		print("Event %d"%i)
		print("# of taus = %d"%num_minus[i])
		print("# of anti-taus = %d"%num_plus[i])
		print("Reconstructed Z Multiplicity = %d"%ZMult[i])
		print("All leading tau pair masses")
		print(First_Pair[i])
		print("All subleading tau pair masses")
		print(Second_Pair[i])
		print("=========================================================================================")
		print("Leading 4-momenta:")
		print(leadE[i])
		print(leadPx[i])
		print(leadPy[i])
		print(leadPz[i])
		print("Paired to Leading 4-momenta:")
		print(firstPairE[i])
		print(firstPairPx[i])
		print(firstPairPy[i])
		print(firstPairPz[i])
		print("SubLeading 4-momenta:")
		print(subleadE[i])
		print(subleadPx[i])
		print(subleadPy[i])
		print(subleadPz[i])
		print("Paired to SubLeading 4-momenta:")
		print(secondPairE[i])
		print(secondPairPx[i])
		print(secondPairPy[i])
		print(secondPairPz[i])
		print("=========================================================================================")


	return ZMult
	
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
	def __init__(self, trigger_bit, trigger_cut = True, offline_cut = False, or_trigger = False):
		self.trigger_bit = trigger_bit
		self.offline_cut = offline_cut
		self.trigger_cut = trigger_cut
		self.OrTrigger = or_trigger
		self.isData = False #Default assumption is running on MC
		#pass

	def process(self, events):
		dataset = events.metadata['dataset']
		event_level = ak.zip(
			{
				"jet_trigger": events.HLTJet,
				"mu_trigger": events.HLTEleMuX,
				"pfMET": events.pfMET,
				"pfMETPhi": events.pfMETPhi,
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

		if ("Data_" in dataset): #Check to see if running on data
			self.isData = True	
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
			
				#Single Muon Trigger	
				tau_21 = tau[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				tau_fail = tau[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				AK8Jet_21 = AK8Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				AK8Jet_fail = AK8Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				Jet_21 = Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				Jet_fail = Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				muon_21 = muon[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				muon_fail = muon[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]

				#Apply offline Single Muon Cut
				tau = tau[ak.any(muon.nMu > 0, axis = 1)]
				AK8Jet = AK8Jet[ak.any(muon.nMu > 0, axis = 1)]
				Jet = Jet[ak.any(muon.nMu > 0, axis = 1)]
				muon = muon[ak.any(muon.nMu > 0, axis = 1)]
				
				#pT
				tau_21 = tau_21[ak.any(muon_21.pt > 52, axis = 1)]
				AK8Jet_21 = AK8Jet_21[ak.any(muon_21.pt > 52, axis = 1)]
				Jet_21 = Jet_21[ak.any(muon_21.pt > 52, axis = 1)]
				event_level_21 = event_level_21[ak.any(muon_21.pt > 52, axis = 1)]
				muon_21 = muon_21[ak.any(muon_21.pt > 52, axis = 1)]
				
				#Apply JetHT_MHT_MET Trigger
				tau_39 = tau_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([39])) == bit_mask([39])]
				AK8Jet_39 = AK8Jet_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([39])) == bit_mask([39])]
				Jet_39 = Jet_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([39])) == bit_mask([39])]
				muon_39 = muon_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([39])) == bit_mask([39])]
		
				#HT Cut
				tau_39 = tau_39[event_level_39.HT > 550]	
				AK8Jet_39 = AK8Jet_39[event_level_39.HT > 550]	
				Jet_39 = Jet_39[event_level_39.HT > 550]
				event_level_39 = event_level_39[event_level_39.HT > 550]

				#MHT Cut
				tau_39 = tau_39[event_level_39.MHT > 110]	
				AK8Jet_39 = AK8Jet_39[event_level_39.MHT > 110]	
				Jet_39 = Jet_39[event_level_39.MHT > 110]
				event_level_39 = event_level_39[event_level_39.MHT > 110]

				#MET Cut	
				tau_39 = tau_39[event_level_39.pfMET > 110]	
				AK8Jet_39 = AK8Jet_39[event_level_39.pfMET > 110]	
				Jet_39 = Jet_39[event_level_39.pfMET > 110]
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
				event_level = ak.concatenate((event_level_21, event_level_39))
				
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
						
						#Apply offline Single Muon Cut
						if (np.exp(1) != np.pi):
							tau = tau[ak.any(muon.nMu > 0, axis = 1)]
							AK8Jet = AK8Jet[ak.any(muon.nMu > 0, axis = 1)]
							Jet = Jet[ak.any(muon.nMu > 0, axis = 1)]
							muon = muon[ak.any(muon.nMu > 0, axis = 1)]
								
							#pT
							tau = tau[ak.any(muon.pt > 52, axis = 1)]
							AK8Jet = AK8Jet[ak.any(muon.pt > 52, axis = 1)]
							Jet = Jet[ak.any(muon.pt > 52, axis = 1)]
							event_level = event_level[ak.any(muon.pt > 52, axis = 1)]
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
						event_level = event_level[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
						print("Number of events after Online Trigger(dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
				
						#Offline Cuts
						#HT Cut
						tau = tau[event_level.HT > 550]
						AK8Jet = AK8Jet[event_level.HT > 550]
						Jet = Jet[event_level.HT > 550]
						event_level = event_level[event_level.HT > 550]

						#pfMET	
						tau = tau[event_level.pfMET > 110]	
						AK8Jet = AK8Jet[event_level.pfMET > 110]	
						Jet = Jet[event_level.pfMET > 110]
						event_level = event_level[event_level.pfMET > 110]
			
						#MHT
						tau = tau[event_level.MHT > 110]	
						AK8Jet = AK8Jet[event_level.MHT > 110]	
						Jet = Jet[event_level.MHT > 110]
						event_level = event_level[event_level.MHT > 110]

			print("Number of events after selection + Trigger: %d"%ak.num(tau,axis=0))
			print("Number of events after Trigger + Selection (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		else:
			if ("SingleMuon" in dataset): #Single Mu
				print("Single Muon Trigger")
				tau = tau[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]	
				AK8Jet = AK8Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]	
				Jet = Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]	
				muon = muon[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]	
				event_level = event_level[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]

				#pT
				if (np.exp(1) != np.pi):
					tau = tau[ak.any(muon.nMu > 0, axis = 1)]
					AK8Jet = AK8Jet[ak.any(muon.nMu > 0, axis = 1)]
					Jet = Jet[ak.any(muon.nMu > 0, axis = 1)]
					muon = muon[ak.any(muon.nMu > 0, axis = 1)]
					
					tau = tau[ak.any(muon.pt > 52, axis = 1)]
					AK8Jet = AK8Jet[ak.any(muon.pt > 52, axis = 1)]
					Jet = Jet[ak.any(muon.pt > 52, axis = 1)]
					event_level = event_level[ak.any(muon.pt > 52, axis = 1)]
					muon = muon[ak.any(muon.pt > 52, axis = 1)]
				
			if ("JetHT" in dataset): #HT 
				print("Jet Trigger")
				tau = tau[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) == bit_mask([39])]	
				AK8Jet = AK8Jet[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) == bit_mask([39])]	
				Jet = Jet[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) == bit_mask([39])]	
				muon = muon[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) == bit_mask([39])]	
				event_level = event_level[np.bitwise_and(event_level.jet_trigger,bit_mask([39])) == bit_mask([39])]
				print("Number of events after Online Trigger(dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))

				#Offline Cuts
				#HT Cut
				tau = tau[event_level.HT > 550]
				AK8Jet = AK8Jet[event_level.HT > 550]
				Jet = Jet[event_level.HT > 550]
				event_level = event_level[event_level.HT > 550]

				#pfMET	
				tau = tau[event_level.pfMET > 110]	
				AK8Jet = AK8Jet[event_level.pfMET > 110]	
				Jet = Jet[event_level.pfMET > 110]
				event_level = event_level[event_level.pfMET > 110]
			
				#MHT
				tau = tau[event_level.MHT > 110]	
				AK8Jet = AK8Jet[event_level.MHT > 110]	
				Jet = Jet[event_level.MHT > 110]
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
			
			#Debugging deltaR, phi, eta and delta phi
			NEvents = 10
			#for i in range(NEvents):
			#	print("Leading tau phi %.3f"%tau_lead[i].phi[0])
			#	print("Paired tau phi %.3f"%leadingTau_Pair[i].phi[0])
			#	print("Leading tau eta %.3f"%tau_lead[i].eta[0])
			#	print("Paired tau eta %.3f"%leadingTau_Pair[i].eta[0])
			#	print("Delta phi %.3f"%deltaphi_Arr[i][0])
			#	print("Delta R %.3f"%deltaR_Arr[i])

			#for i in range(len(deltaphi_Arr)):
			#	if (np.abs(deltaphi_Arr[i][0]) > np.pi):
			#		print("!!Delta phi outisde expected range!!")
			#		print(deltaphi_Arr[i][0])
			#		print(delta_phi(tau_lead[i].leadingTau_Pair))
			#		print(tau_lead[i].phi)
			#		print(leadingTau_Pair[i].phi)
			
			#for x in leadingTau_Pair.pt:
			#	if len(x) != 1:
			#		print("!!Multi-tau paired to leading tau!!")
			#		print(x)

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
				id_cond = np.bitwise_and(lepton.IDbit,2) != 0
				d0_cond = np.abs(lepton.D0) < 0.045
				dz_cond = np.abs(lepton.Dz) < 0.2
				good_lepton_cond = np.bitwise_and(id_cond, np.bitwise_and(d0_cond, dz_cond))
				good_lepton = lepton[good_lepton_cond]
			#Make good electron selection
			if (lep_flavor == "ele"):
				cond1 = np.bitwise_and(np.abs(lepton.SCEta) <= 0.8, lepton.IDMVANoIso > 0.837)
				cond2 = np.bitwise_and(np.bitwise_and(np.abs(lepton.SCEta) > 0.8, np.abs(lepton.SCEta) <= 1.5), electron.IDMVANoIso > 0.715)
				cond3 = np.bitwise_and(np.abs(lepton.SCEta) >= 1.5, electron.IDMVANoIso > 0.357)
				good_lepton_cond = np.bitwise_or(cond1,np.bitwise_or(cond2,cond3))
				good_lepton = lepton[good_lepton_cond]
			
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


		
		#Construct all possible valid ditau pairs
		tau = tau[ak.num(tau,axis=1) > 0] #Handle empty arrays left by the trigger
		#for t in tau:
		#	print(t.eta)
		#	if len(t) != 4:
		#		print("Event does not have 4 events")

		#tau_plus = tau[tau.charge > 0]		
		#tau_minus = tau[tau.charge < 0]
		#tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau_plus,2))
		#tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau_minus,2))

		#if (self.isData):
		#print(tau_plus1.eta)
		#print(tau_plus2.eta)
		#print(tau_minus1.eta)
		#print(tau_minus2.eta)

		#Get leading, subleading and fourth leading taus
		##leading_tau = tau[:,0]
		#subleading_tau = tau[:,1]
		#thirdleading_tau = tau[:,2]
		#fourthleading_tau = tau[:,3]
		
		#deltaR11 = deltaR(tau_plus1, tau_minus1)
		#deltaR12 = deltaR(tau_plus1, tau_minus2)
		#deltaR22 = deltaR(tau_plus2, tau_minus2)
		#deltaR21 = deltaR(tau_plus2, tau_minus1)

		#radionPT = np.sqrt((np.array(leading_tau.Px) + np.array(subleading_tau.Px) + np.array(thirdleading_tau.Px) + np.array(fourthleading_tau.Px))**2 + (np.array(leading_tau.Py) + np.array(subleading_tau.Py) + np.array(thirdleading_tau.Py) + np.array(fourthleading_tau.Py))**2) 
		
		#Construct each Higgs' transverse momenta
		#lead_cond1 = np.bitwise_and(tau_plus1.pt > tau_minus1.pt, deltaR11 < deltaR12)
		#lead_cond2 = np.bitwise_and(tau_plus1.pt > tau_minus1.pt, deltaR12 < deltaR11)
		#lead_cond3 = np.bitwise_and(tau_plus1.pt < tau_minus1.pt, deltaR11 < deltaR21)
		#lead_cond4 = np.bitwise_and(tau_plus1.pt < tau_minus1.pt, deltaR21 < deltaR11)
		
		#PxLeading = np.append(ak.ravel(tau_plus1[lead_cond1].Px + tau_minus1[lead_cond1].Px),ak.ravel(tau_plus1[lead_cond2].Px + tau_minus2[lead_cond2].Px))
		#PxLeading = np.append(PxLeading,ak.ravel(tau_plus1[lead_cond3].Px + tau_minus1[lead_cond3].Px))
		#PxLeading = np.append(PxLeading,ak.ravel(tau_plus2[lead_cond4].Px + tau_minus1[lead_cond4].Px))
		
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

		#Look at individual pair delta Phi
		#TauDeltaPhi_all_hist.fill("Leading pair", ak.ravel((tau_plus1[lead_cond1].phi - tau_minus1[lead_cond1].phi + np.pi) % (2 * np.pi) - np.pi))	
		#TauDeltaPhi_all_hist.fill("Subleading pair", ak.ravel((tau_plus2[lead_cond1].phi - tau_minus2[lead_cond1].phi + np.pi) % (2 * np.pi) - np.pi))	
		#TauDeltaPhi_all_hist.fill("Leading pair", ak.ravel((tau_plus1[lead_cond2].phi - tau_minus2[lead_cond2].phi + np.pi) % (2 * np.pi) - np.pi))	
		#TauDeltaPhi_all_hist.fill("Subleading pair", ak.ravel((tau_plus2[lead_cond2].phi - tau_minus1[lead_cond2].phi + np.pi) % (2 * np.pi) - np.pi))	
		#TauDeltaPhi_all_hist.fill("Leading pair", ak.ravel((tau_plus1[lead_cond3].phi - tau_minus1[lead_cond3].phi + np.pi) % (2 * np.pi) - np.pi))	
		#TauDeltaPhi_all_hist.fill("Subleading pair", ak.ravel((tau_plus2[lead_cond3].phi - tau_minus2[lead_cond3].phi + np.pi) % (2 * np.pi) - np.pi))	
		#TauDeltaPhi_all_hist.fill("Leading pair", ak.ravel((tau_plus2[lead_cond4].phi - tau_minus1[lead_cond4].phi + np.pi) % (2 * np.pi) - np.pi))	
		#TauDeltaPhi_all_hist.fill("Subleading pair", ak.ravel((tau_plus1[lead_cond4].phi - tau_minus2[lead_cond4].phi + np.pi) % (2 * np.pi) - np.pi))	

		
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
		
		return{
			dataset: {
				"Weight": weight_Val,
				"FourTau_Mass_Arr": ak.to_list(FourTau_Mass_Arr), 
				"HiggsDeltaPhi_Arr": ak.to_list(Higgs_DeltaPhi_Arr),
				"LeadingHiggs_mass": ak.to_list(LeadingHiggs_mass_Arr),
				"SubLeadingHiggs_mass": ak.to_list(SubLeadingHiggs_mass_Arr),
				"Higgs_DeltaR_Arr": ak.to_list(diHiggs_dR_Arr),
				"leading_dR_Arr": ak.to_list(leading_dR_Arr),
				"subleading_dR_Arr": ak.to_list(subleading_dR_Arr),
				
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
				"tau_lead_pt_Arr": ak.to_list(ak.ravel(tau[ak.argsort(tau.pt,axis=-1)][:,0].pt)),
				"tau_sublead_pt_Arr": ak.to_list(ak.ravel(tau[ak.argsort(tau.pt,axis=-1)][:,1].pt)),
				"tau_3rdlead_pt_Arr": ak.to_list(ak.ravel(tau[ak.argsort(tau.pt,axis=-1)][:,2].pt)),
				"tau_4thlead_pt_Arr": ak.to_list(ak.ravel(tau[ak.argsort(tau.pt,axis=-1)][:,3].pt)),
				"tau_eta_Arr": ak.to_list(ak.ravel(tau.eta)),
				"ZMult_Arr": ak.to_list(ak.ravel(event_level.ZMult)),
				"ZMult_ele_Arr": ak.to_list(ak.ravel(event_level.ZMult_e)),
				"ZMult_mu_Arr": ak.to_list(ak.ravel(event_level.ZMult_mu)),
				"ZMult_tau_Arr": ak.to_list(ak.ravel(event_level.ZMult_tau)),
                "BJet_Arr": ak.to_list(ak.ravel(event_level.nBJets)),
				"Lumi_Val": Lumi_2018,
				"CrossSec_Val": crossSecVal,
				"NEvent_Val": numEvents_Dict[dataset],
				"Weight_Val": weight_Val,
			}
		}
	
	def postprocess(self, accumulator):
		pass	

if __name__ == "__main__":
	#mass_str_arr = ["1000","2000","3000"]
	mass_str_arr = ["2000"]
	
	#Functions and variables for Luminosity weights
	lumi_table_data = {"MC Sample":[], "Luminosity":[], "Cross Section (pb)":[], "Number of Events":[], "Calculated Weight":[]}
	
	#Trigger dictionaries
	#trigger_dict = {"Mu50": (21,False), "PFMET120_PFMHT120_IDTight": (27,False), "EitherOr_Trigger": (41,True)}
	#trigger_dict = {"Mu50": (21,False), "PFMET120_PFMHT120_IDTight": (27,False), "EitherOr_Trigger": (41,True)}
	#trigger_dict = {"EitherOr_Trigger": (41,True)}
	trigger_dict = {"Mu50": (21,False)}
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
	four_tau_hist_list = ["FourTau_Mass_Arr","HiggsDeltaPhi_Arr", "Higgs_DeltaR_Arr","leading_dR_Arr","subleading_dR_Arr","LeadingHiggs_mass","SubLeadingHiggs_mass", "radionPT_Arr", "tau_pt_Arr", 
			"tau_eta_Arr","ZMult_Arr", "BJet_Arr", "tau_lead_pt_Arr", "tau_sublead_pt_Arr", "tau_3rdlead_pt_Arr", "tau_4thlead_pt_Arr", "leading_dPhi_Arr", "subleading_dPhi_Arr", 
			"radionMET_dPhi_Arr","leadingHiggs_Rad_dR_Arr","subleadingHiggs_Rad_dR_Arr","leadingHiggs_MET_dPhi_Arr","subleadingHiggs_MET_dPhi_Arr","Radion_eta_Arr", "Radion_Charge_Arr"]
	#four_tau_hist_list = ["ZMult_Arr","ZMult_ele_Arr","ZMult_mu_Arr", "ZMult_tau_Arr"]
	#four_tau_hist_list = ["LeadingHiggs_mass"] #Only make 1 histogram for brevity/debugging purposes
	hist_name_dict = {"FourTau_Mass_Arr": r"Reconstructed 4-$\tau$ invariant mass", "HiggsDeltaPhi_Arr": r"Reconstructed Higgs $\Delta \phi$", "Higgs_DeltaR_Arr": r"Reconstructed Higgs $\Delta R$",
					"leading_dR_Arr": r"$\Delta R$ of leading di-$\tau$ pair", "subleading_dR_Arr": r"$\Delta R$ of subleading di-$\tau$ pair", 
					"LeadingHiggs_mass": r"Leading di-$\tau$ pair invariant mass", "SubLeadingHiggs_mass": r"Subleading di-$\tau$ pair invariant mass", "radionPT_Arr": r"Reconstructed Radion $p_T$",
					"tau_pt_Arr": r"$\tau$ $p_T$", "tau_eta_Arr": r"$\tau \ \eta$", "ZMult_Arr": r"Z Boson Multiplicity", "ZMult_mu_Arr": r"Z Boson Multiplicity (muons only)", 
                    "ZMult_ele_Arr": r"Z Boson Multiplicity (electrons only)", "ZMult_tau_Arr" : r"Z Boson Multiplicity (from taus)", "BJet_Arr": r"B-Jet Multiplicity", 
                    "tau_lead_pt_Arr": r"Leadng $\tau$ $p_T$", "tau_sublead_pt_Arr": r"Subleading $\tau$ $p_T$", "tau_3rdlead_pt_Arr": r"Third leading $\tau$ $p_T$", 
					"tau_4thlead_pt_Arr": r"Fourth leading $\tau$ $p_T$", "leading_dPhi_Arr": r"Leading di-$\tau$ $\Delta \phi$", "subleading_dPhi_Arr": r"Subleading di-$\tau$ $\Delta \phi$",
					"radionMET_dPhi_Arr": r"Radion MET $\Delta \phi$", "leadingHiggs_Rad_dR_Arr": r"Leading Higgs Radion $\Delta$R", 
					"subleadingHiggs_Rad_dR_Arr": r"Subleading Higgs Radion $\Delta$R", "leadingHiggs_MET_dPhi_Arr": r"Leading Higgs MET $\Delta \phi$", 
					"subleadingHiggs_MET_dPhi_Arr": r"Subleading Higgs MET $\Delta \phi$", "Radion_eta_Arr": r"Radion $\eta$", "Radion_Charge_Arr": r"Radion Charge"}
	#four_tau_hist_list = ["HiggsDeltaPhi_Arr","Pair_DeltaPhi_Hist"]


	#Loop over all mass points
	for mass in mass_str_arr:
		print("====================Radion Mass = " + mass[0] + "." + mass[1] + " TeV====================")
		file_dict_test = { #Reduced files to run over
			"ZZ4l": [background_base + "ZZ4l.root"],
			"Signal": [signal_base + mass + ".root"],
			#"Data_SingleMuon": [data_loc + "SingleMu_Run2018A.root"] #, data_loc + "SingleMu_Run2018B.root", data_loc + "SingleMu_Run2018C.root", data_loc + "SingleMu_Run2018D.root"],
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
				numEvents_Dict[key_name] = tempFile['hEvents'].member('fEntries')/2
			else: #Ignore data files
				continue
		
		#Histogram bins
		N1 = 20 #20 for Single Mu and 6 for Jet METMHTHT Trigger
		N2 = 12 #12 for Single Mu and 6 for Jet MEHTMHT Trigger
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
			"tau_lead_pt_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,200, label=r"Subleading $\tau$ $p_T$ (GeV)").Double(),
			"tau_sublead_pt_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,200, label=r"Leading $\tau$ $p_T$ (GeV)").Double(),
			"tau_3rdlead_pt_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,200, label=r"Third Leading $\tau$ $p_T$ (GeV)").Double(),
			"tau_4thlead_pt_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,200, label=r"Fourth Leading $\tau$ $p_T$ (GeV)").Double(),
			
			"leading_dPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(), 
			"subleading_dPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(), 
			"radionMET_dPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(), 
			"leadingHiggs_Rad_dR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
			"subleadingHiggs_Rad_dR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
			"leadingHiggs_MET_dPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Leading Higgs MET $\Delta R$").Double(), 
			"subleadingHiggs_MET_dPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta R$").Double(),
			"Radion_eta_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-5,5, label = r"Radion $\eta$").Double(),
			"Radion_Charge_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name = "background").Reg(10,-5,5,label = r"Radion Electric Charge").Double(),

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
			"tau_lead_pt_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,200, label=r"Leading $\tau$ $p_T$ (GeV)").Double(),
			"tau_sublead_pt_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,200, label=r"Subleading $\tau$ $p_T$ (GeV)").Double(),
			"tau_3rdlead_pt_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,200, label=r"Third leading $\tau$ $p_T$ (GeV)").Double(),
			"tau_4thlead_pt_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,200, label=r"Fourth leading $\tau$ $p_T$ (GeV)").Double(),
			"leading_dPhi_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(), 
			"subleading_dPhi_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(), 
			"radionMET_dPhi_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(), 
			"leadingHiggs_Rad_dR_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
			"subleadingHiggs_Rad_dR_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
			"leadingHiggs_MET_dPhi_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Leading Higgs MET $\Delta R$").Double(), 
			"subleadingHiggs_MET_dPhi_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta R$").Double(),
			"Radion_eta_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-5,5, label = r"Radion $\eta$").Double(),
			"Radion_Charge_Arr": hist.Hist.new.StrCat(["Signal"],name = "signal").Reg(10,-5,5,label = r"Radion Electric Charge").Double()


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
			"tau_lead_pt_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,200, label=r"Leading $\tau$ $p_T$ (GeV)").Double(),
			"tau_sublead_pt_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,200, label=r"Subleading $\tau$ $p_T$ (GeV)").Double(),
			"tau_3rdlead_pt_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,200, label=r"Third leading $\tau$ $p_T$ (GeV)").Double(),
			"tau_4thlead_pt_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,200, label=r"Fourth leading $\tau$ $p_T$ (GeV)").Double(),
			"leading_dPhi_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(), 
			"subleading_dPhi_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(), 
			"radionMET_dPhi_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(), 
			"leadingHiggs_Rad_dR_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
			"subleadingHiggs_Rad_dR_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
			"leadingHiggs_MET_dPhi_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Leading Higgs MET $\Delta R$").Double(), 
			"subleadingHiggs_MET_dPhi_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta R$").Double(),
			"Radion_eta_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-5,5, label = r"Radion $\eta$").Double(),
			"Radion_Charge_Arr": hist.Hist.new.StrCat(["Data"],name = "data").Reg(10,-5,5,label = r"Radion Electric Charge").Double()
		}
		
		background_list = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"]
		#background_list = [r"$ZZ \rightarrow 4l$"]
		signal_list = [r"MC Sample $m_\phi$ = %s TeV"%mass[0]]
		background_plot_names = {r"$t\bar{t}$" : "_ttbar_", r"Drell-Yan+Jets": "_DYJets_", "Di-Bosons" : "_DiBosons_", "Single Top": "_SingleTop+", "QCD" : "_QCD_", "W+Jets" : "_WJets_", r"$ZZ \rightarrow 4l$" : "_ZZ4l_"} #For file names
		background_dict = {r"$t\bar{t}$" : ["TTToSemiLeptonic","TTTo2L2Nu","TTToHadronic"], 
				r"Drell-Yan+Jets": ["DYJetsToLL_Pt-50To100","DYJetsToLL_Pt-100To250","DYJetsToLL_Pt-250To400","DYJetsToLL_Pt-400To650","DYJetsToLL_Pt-650ToInf"], 
				"Di-Bosons": ["WZ3l1nu","WZ2l2q","WZ1l1nu2q","ZZ2l2q", "WZ1l3nu", "VV2l2nu"], "Single Top": ["Tbar-tchan","T-tchan","Tbar-tW","T-tW"], 
				"W+Jets": ["WJetsToLNu_HT-100To200","WJetsToLNu_HT-200To400","WJetsToLNu_HT-400To600","WJetsToLNu_HT-600To800","WJetsToLNu_HT-800To1200","WJetsToLNu_HT-1200To2500","WJetsToLNu_HT-2500ToInf"],
				r"$ZZ \rightarrow 4l$" : ["ZZ4l"]
		}
		#background_dict = {r"$ZZ \rightarrow 4l$" : ["ZZ4l"]}

		for trigger_name, trigger_pair in trigger_dict.items(): #Run over all triggers/combinations of interest
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
				"leading_dPhi_Arr": "leadingdiTau_DeltaPhi_Mass_" + mass + "-" + trigger_name, "subleading_dPhi_Arr": "leadingdiTau_DeltaPhi_Mass_" + mass + "-" + trigger_name,
				"radionMET_dPhi_Arr": "Radion_MET_DeltaPhi_Mass_" + mass + "-" + trigger_name, "leadingHiggs_Rad_dR_Arr": "LeadingHiggs_Radion_DeltaR_Mass_" + mass + "-" + trigger_name, 
				"subleadingHiggs_Rad_dR_Arr": "SubLeadingHiggs_Radion_DeltaR_Mass_" + mass + "-" + trigger_name, 
				"leadingHiggs_MET_dPhi_Arr": "LeadingHiggs_MET_DeltaPhi_Mass_" + mass + "-" + trigger_name, 
				"subleadingHiggs_MET_dPhi_Arr": "SubLeadingHiggs_MET_DeltaPhi_Mass_" + mass + "-" + trigger_name, "Radion_eta_Arr": "Radion_eta_Mass_" + mass + "-" + trigger_name,
				"Radion_Charge_Arr": "Radion_Charge_Mass" + mass + "-" + trigger_name
			}
			
			fourtau_out = iterative_runner(file_dict, treename="4tau_tree", processor_instance=FourTauPlotting(trigger_bit=trigger_pair[0], or_trigger=trigger_pair[1]))
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
						"tau_lead_pt_Arr" : hist.Hist.new.Regular(N1,0,200, label=r"Leading $\tau$ $p_T$ (GeV)").Double(),
						"tau_sublead_pt_Arr" : hist.Hist.new.Regular(N1,0,200, label=r"Subleading $\tau$ $p_T$ (GeV)").Double(),
						"tau_3rdlead_pt_Arr" : hist.Hist.new.Regular(N1,0,200, label=r"Third leading $\tau$ $p_T$ (GeV)").Double(),
						"tau_4thlead_pt_Arr" : hist.Hist.new.Regular(N1,0,200, label=r"Fourth leading $\tau$ $p_T$ (GeV)").Double(),
						"leading_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(), 
						"subleading_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(), 
						"radionMET_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(), 
						"leadingHiggs_Rad_dR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
						"subleadingHiggs_Rad_dR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
						"leadingHiggs_MET_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Leading Higgs MET $\Delta R$").Double(), 
						"subleadingHiggs_MET_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta R$").Double(),
						"Radion_eta_Arr": hist.Hist.new.Regular(N1,-5,5, label = r"Radion $\eta$").Double(),
						"Radion_Charge_Arr": hist.Hist.new.Regular(10,-5,5,label = r"Radion Electric Charge").Double(),
					}
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
							"tau_sublead_pt_Arr" : hist.Hist.new.Regular(N1,0,200, label=r"Subleading $\tau$ $p_T$ (GeV)").Double(),
							"tau_3rdlead_pt_Arr" : hist.Hist.new.Regular(N1,0,200, label=r"Third leading $\tau$ $p_T$ (GeV)").Double(),
							"tau_4thlead_pt_Arr" : hist.Hist.new.Regular(N1,0,200, label=r"Fourth leading $\tau$ $p_T$ (GeV)").Double(),
							"leading_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(), 
							"subleading_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(), 
							"radionMET_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(), 
							"leadingHiggs_Rad_dR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
							"subleadingHiggs_Rad_dR_Arr": hist.Hist.new.Regular(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
							"leadingHiggs_MET_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Leading Higgs MET $\Delta R$").Double(), 
							"subleadingHiggs_MET_dPhi_Arr": hist.Hist.new.Regular(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta R$").Double(),
							"Radion_eta_Arr": hist.Hist.new.Regular(N1,-5,5, label = r"Radion $\eta$").Double(),
							"Radion_Charge_Arr": hist.Hist.new.Regular(10,-5,5,label = r"Radion Electric Charge").Double()
						}
						background_array = []
						backgrounds = background_dict[background_type]
						
						#Loop over all backgrounds
						for background in backgrounds:
							if (mass == "2000"): #Only need to generate single background once
								if (hist_name == "Radion_CHarge_Arr"):
									lumi_table_data["MC Sample"].append(background)
									lumi_table_data["Luminosity"].append(fourtau_out[background]["Lumi_Val"])
									lumi_table_data["Cross Section (pb)"].append(fourtau_out[background]["CrossSec_Val"])
									lumi_table_data["Number of Events"].append(fourtau_out[background]["NEvent_Val"])
									lumi_table_data["Calculated Weight"].append(fourtau_out[background]["Weight_Val"])
								fig2, ax2 = plt.subplots()
								hist_dict_single_background[hist_name].fill(fourtau_out[background][hist_name],weight = fourtau_out[background]["Weight"]) #Obtain background distributions 
								hist_dict_single_background[hist_name].plot1d(ax=ax2)
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
							hist_dict_background[hist_name].fill(background_type,fourtau_out[background][hist_name],weight = fourtau_out[background]["Weight"]) #Obtain background distributions
							print("Background %s added"%background)
							
					
					hist_dict_signal[hist_name].fill("Signal",fourtau_out["Signal"][hist_name],weight = fourtau_out["Signal"]["Weight"]) #Obtain signal distribution
					
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

					#print("Number of Jet HT entries: %d"%len(fourtau_out["Data_JetHT"][hist_name]))
					
					#Put histograms into stacks and arrays for plotting purposes (is the issue arising here??)
					background_stack = hist_dict_background[hist_name].stack("background")
					signal_stack = hist_dict_signal[hist_name].stack("signal")
					data_stack = hist_dict_data[hist_name].stack("data")
					signal_array = [signal_stack["Signal"]]
					data_array = [data_stack["Data"]]
					for background in background_list:
						background_array.append(background_stack[background])
					
					#Stack background distributions and plot signal + data distribution
					fig,ax = plt.subplots()
					hep.histplot(background_array,ax=ax,stack=True,histtype="fill",label=background_list)
					hep.histplot(signal_array,ax=ax,stack=True,histtype="step",label=signal_list)
					hep.histplot(data_array,ax=ax,stack=False,histtype="errorbar", yerr=False,label=["Data"],marker="o",color = "k") #,facecolor='black',edgecolor='black') #,mec='k')
					hep.cms.text("Preliminary",loc=0,fontsize=13)
					ax.set_title(hist_name_dict[hist_name],loc = "right")
					ax.legend(fontsize=10, loc='upper right')
					plt.savefig(four_tau_names[hist_name])
					plt.close()
	
	#Store luminosity Weighting debugging table
	#fig, ax = plt.subplots()
	#fig.patch.set_visible(False)
	#ax.axis('off')
	#ax.axis('tight')
	#lumi_frame = pd.DataFrame(lumi_table_data)
	#lumi_frame.to_csv("Lumi_Weight_Table_Debugging.csv", sep='\t')
	#outTable = ax.table(cellText=lumi_frame.values, colLabels=lumi_frame.columns, loc='center', cellLoc='center')	
	
	
