import awkward as ak
import uproot
import hist
from hist import intervals
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from coffea import processor, nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import candidate
from math import pi

hep.style.use(hep.style.CMS)
TABLEAU_COLORS = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

#Global Variables
WScaleFactor = 1.21
DYScaleFactor = 1.23
TT_FullLep_BR = 0.1061
TT_SemiLep_BR = 0.4392
TT_Had_BR = 0.4544

def delta_phi(vec1,vec2):
	return (vec1.phi - vec2.phi + pi) % (2*pi) - pi	

def deltaR(part1, part2):
	return np.sqrt((part2.eta - part1.eta)**2 + (delta_phi(part1,part2))**2)

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
xSection_Dictionary = {"Signal": 0.001, #Chosen to make plots readable
						#TTBar Background
						"TTTo2L2Nu": 831.76*TT_FullLep_BR, "TTToSemiLeptonic": 831.76*TT_SemiLep_BR, "TTToHadronic": 831.76*TT_Had_BR,
						#DiBoson Background
						"ZZ2l2q": 3.22, "WZ3l1nu": 4.708, "WZ2l2q": 5.595, "WZ1l1nu2q": 10.71, "VV2l2nu": 11.95, "WZ1l3nu": 3.05, #"WZ3l1nu.root" : 27.57,
						#ZZ->4l
						"ZZ4l": 1.121, "ZZTo4L_powheg": 1.121,
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

def weight_calc(sample,events,numEvents=1):
	return Lumi_2018*xSection_Dictionary[sample]/numEvents

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
				"MET": events.pfMET
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
				"pfMET": events.pfMET,
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
					
			},
			with_name="ElectronArray",
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
				"pfMET": events.pfMET,
				"PFLooseId": events.jetPFLooseId,
				"HT": ak.sum(events.jetPt, axis=1),
				"eta": events.jetEta,
				"phi": events.jetPhi,
			},
			with_name="PFJetArray",
			behavior=candidate.behavior,
		)
		
		#Histograms 4 tau
		FourTau_Mass_hist = hist.Hist.new.Reg(40,0,3000, label = r"$m_{4\tau} [GeV]$").Double()
		Higgs_DeltaPhi = hist.Hist.new.Reg(40,-pi,pi, label = r"Higgs $\Delta \phi$").Double()
		TauDeltaPhi_all_hist = (
			hist.Hist.new
			.StrCat(["Leading pair","Subleading pair"], name = "delta_phi")
			.Reg(50, -pi, pi, name="delta_phi_all", label=r"$\Delta \phi$") 
            .Double()
		)
		DiTauMass_all_hist = (
			hist.Hist.new.StrCat(["Leading Pair","Subleading Pair"], name = "diTauMass")
			.Reg(50,0,125, name = "diTau_Mass", label=r"Di-$\tau$ Mass (GeV)")
			.Double()
		)
		RadionPTComp = (
				hist.Hist.new
				.StrCat([r"$\tau$ Reconstructed","Higgs Reconstructed"], name = "pTComp")
				.Reg(40,0,2000,name="pTComp_all",label = r"p_T [GeV]")
				.Double()
		)
		TauPt_Hist = (
				hist.Hist.new
				.StrCat([r"Leading $\tau$",r"Sub-leading $\tau$",r"Third leading $\tau$",r"Fourth leading $\tau$"], name = "taupT")
				.Reg(50,0,1500., name="p_T_all", label=r"$p_T$ [GeV]")
				.Double()
		)

		print("!!!=====Dataset=====!!!!")	
		print(type(dataset))
		print(dataset)

		if ("Data_" in dataset): #Check to see if running on data
			self.isData = True	
		print("Number of events before selection + Trigger: %d"%ak.num(tau,axis=0))
		#MHT
		Jet_MHT = Jet[Jet.Pt > 30]
		Jet_MHT = Jet_MHT[np.abs(Jet_MHT.eta) < 5]
		Jet_MHT = Jet_MHT[Jet_MHT.PFLooseId > 0.5]
		Jet_MHT["MHT_x"] = ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False)# + ak.sum(JetUp_MHT.PtTotUncUp*np.cos(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.cos(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet_MHT["MHT_y"] = ak.sum(Jet_MHT.Pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False)# + ak.sum(JetUp_MHT.PtTotUncUp*np.sin(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.sin(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet_MHT["MHT"] = np.sqrt(Jet_MHT.MHT_x**2 + Jet_MHT.MHT_y**2)
		
		#HT Seleciton (new)
		#tau_temp1,HT = ak.unzip(ak.cartesian([tau,Jet_MHT], axis = 1, nested = True))
		Jet_HT = Jet[Jet.Pt > 30]
		Jet_HT = Jet_HT[np.abs(Jet_HT.eta) < 3]
		Jet_HT = Jet_HT[Jet_HT.PFLooseId > 0.5]
		Jet_temp,tau_temp1 = ak.unzip(ak.cartesian([Jet_HT,tau], axis = 1, nested = True))
		
		#Get Cross clean free histograms
		HT_num = 0
		for x in ak.sum(Jet.Pt,axis = 1,keepdims=False):
			HT_num += 1
	
		mval_temp = deltaR(tau_temp1,Jet_temp) >= 0.5
		if (len(Jet.Pt) != len(mval_temp)):
			print("Things aren't good")
			if (len(Jet.Pt) > len(mval_temp)):
				print("More Jets than entries in mval_temp")
			if (len(Jet.Pt) < len(mval_temp)):
				print("Fewer entries in Jets than mval_temp")

		Jet_HT["dR"] = mval_temp

		HT_Val_NoCuts = ak.sum(Jet_HT.Pt,axis = 1,keepdims=True)# + ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
		test_HT = ak.sum(Jet.Pt,axis = 1,keepdims=True)
		HT_num = 0
		for x in ak.ravel(Jet.HT):
			HT_num += 1
				
		#Apply selections
		trigger_mask = bit_mask([self.trigger_bit])		
		tau = tau[tau.pt > 20] #pT selection
		if (self.isData or not(self.isData)):
			print("# of events after pT cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		tau = tau[np.abs(tau.eta) < 2.3] #eta selection
		if (self.isData or not(self.isData)):
			print("# of events after eta cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		
		#Isolation and decay selections
		tau = tau[tau.decay >= 0.5]	
		tau = tau[tau.iso >= 0.0] #Make loose to ensure high number of statistics
		if (self.isData or not(self.isData)):
			print("# of events after isolation and decay cuts (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))


		#Delta R Cut on taus (identifiy and remove jets incorrectly reconstructed as taus)
		a,b = ak.unzip(ak.cartesian([tau,tau], axis = 1, nested = True)) #Create all di-tau pairs
		select_arr = np.bitwise_and(deltaR(a,b) < 0.8, deltaR(a,b) != 0) 
		tau["dRCut"] = select_arr
		#print(ak.num(tau[ak.any(tau.dRCut, axis = 2) == True],axis=1))
		#for i in range(10):
		#	print(select_arr[i])
		#	print(deltaR(a,b)[i])	
		tau = tau[ak.any(tau.dRCut, axis = 2) == True]
		if (self.isData or not(self.isData)):
			print("# of events after delta R cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		
		AK8Jet = AK8Jet[(ak.sum(tau.charge,axis=1) == 0)] #Apply charge conservation cut to AK8Jets
		event_level = event_level[(ak.sum(tau.charge,axis=1) == 0)]
		Jet = Jet[(ak.sum(tau.charge,axis=1) == 0)]
		Jet_HT = Jet_HT[(ak.sum(tau.charge,axis=1) == 0)]
		Jet_MHT = Jet_MHT[(ak.sum(tau.charge,axis=1) == 0)]
		electron = electron[(ak.sum(tau.charge,axis=1) == 0)]
		muon = muon[(ak.sum(tau.charge,axis=1) == 0)]
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		if (self.isData or not(self.isData)):
			print("# of events after lepton number cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))

		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		event_level = event_level[ak.num(tau) >= 4]
		Jet_MHT = Jet_MHT[ak.num(tau) >= 4]
		Jet_HT = Jet_HT[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		electron = electron[ak.num(tau) >= 4] 
		muon = muon[ak.num(tau) >= 4] 
		tau = tau[ak.num(tau) >= 4] #4 tau events
		if (self.isData or not(self.isData)):
			print("# of events after 4-tau cut (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
	
		print("tau length = %d\nevent_level length = %d"%(ak.num(tau,axis=0),ak.num(event_level,axis=0)))	

		
		if (not(self.isData)):	#MC trigger logic
			if (self.OrTrigger): #Select for both triggers
				event_level_21 = event_level[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				event_level_fail = event_level[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				event_level_27 = event_level_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([27])) == bit_mask([27])]
				
				tau_21 = tau[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				tau_fail = tau[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				tau_27 = tau_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([27])) == bit_mask([27])]
				AK8Jet_21 = AK8Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				AK8Jet_fail = AK8Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				AK8Jet_27 = AK8Jet_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([27])) == bit_mask([27])]
				Jet_21 = Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]
				Jet_fail = Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) != bit_mask([21])]
				Jet_27 = Jet_fail[np.bitwise_and(event_level_fail.jet_trigger,bit_mask([27])) == bit_mask([27])]

				#Recombine
				tau = ak.concatenate((tau_21,tau_27))
				AK8Jet = ak.concatenate((AK8Jet_21, AK8Jet_27))
				Jet = ak.concatenate((Jet_21,Jet_27))
				
			else: #Single Trigger
				if (self.trigger_bit != None and self.OrTrigger != False):
					if (self.trigger_bit == 21):
						tau = tau[np.bitwise_and(event_level.mu_trigger,trigger_mask) == trigger_mask]
						AK8Jet = AK8Jet[np.bitwise_and(event_level.mu_trigger,trigger_mask) == trigger_mask]
						Jet = Jet[np.bitwise_and(event_level.mu_trigger,trigger_mask) == trigger_mask]
					if (self.trigger_bit == 27):
						tau = tau[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
						AK8Jet = AK8Jet[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
						Jet = Jet[np.bitwise_and(event_level.jet_trigger,trigger_mask) == trigger_mask]
			
			print("Number of events after selection + Trigger: %d"%ak.num(tau,axis=0))
			print("Number of events after Trigger + Selection (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		else:
			if ("SingleMuon" in dataset): #Single Mu
				print("Single Muon Trigger")
				tau = tau[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]	
				AK8Jet = AK8Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]	
				Jet = Jet[np.bitwise_and(event_level.mu_trigger,bit_mask([21])) == bit_mask([21])]	
				
			if ("JetHT" in dataset): #HT 
				print("Jet HT Trigger")
				tau = tau[np.bitwise_and(event_level.jet_trigger,bit_mask([27])) == bit_mask([27])]	
				AK8Jet = AK8Jet[np.bitwise_and(event_level.jet_trigger,bit_mask([27])) == bit_mask([27])]	
				Jet = Jet[np.bitwise_and(event_level.jet_trigger,bit_mask([27])) == bit_mask([27])]	
			
			print("# of events after Trigger + Selection: %d"%ak.num(tau,axis=0))
			print("# of events after Trigger + Selection (dropping empty arrays): %d"%ak.num(tau[ak.num(tau,axis=1) > 0],axis=0))
		
		#Construct all possible valid ditau pairs
		tau = tau[ak.num(tau) > 0] #Handle empty arrays left by the trigger

		tau_plus = tau[tau.charge > 0]		
		tau_minus = tau[tau.charge < 0]
		tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau_plus,2))
		tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau_minus,2))

		#if (self.isData):
		print(tau_plus1.eta)
		print(tau_plus2.eta)
		print(tau_minus1.eta)
		print(tau_minus2.eta)

		#Get leading, subleading and fourth leading taus
		leading_tau = tau[:,0]
		subleading_tau = tau[:,1]
		thirdleading_tau = tau[:,2]
		fourthleading_tau = tau[:,3]
		
		deltaR11 = deltaR(tau_plus1, tau_minus1)
		deltaR12 = deltaR(tau_plus1, tau_minus2)
		deltaR22 = deltaR(tau_plus2, tau_minus2)
		deltaR21 = deltaR(tau_plus2, tau_minus1)

		#print(leading_tau)
		#print(leading_tau.Px)
		#print(np.array(leading_tau.Px))
		radionPT = np.sqrt((np.array(leading_tau.Px) + np.array(subleading_tau.Px) + np.array(thirdleading_tau.Px) + np.array(fourthleading_tau.Px))**2 + (np.array(leading_tau.Py) + np.array(subleading_tau.Py) + np.array(thirdleading_tau.Py) + np.array(fourthleading_tau.Py))**2) 
		plusNum = 0
		minusNum = 0
		
		print(plusNum)
		print(minusNum)	
		#Test debugging
		#print("Tau plus 1 PT:")
		#print(tau_plus1.pt)
		#print(len(tau_plus1.pt))
		#print("Tau minus 1 PT:")
		#print(tau_plus1.pt)
		#print(len(tau_plus1.pt))
		
		#for i in range(len(tau_plus1.pt)):
		#	print("=======================================")
		#	print(tau_plus1.pt[i])
		#	print(tau_minus1.pt[i])
		#	print("=======================================")
			#if ( (ak.ravel(tau_plus1.pt[i]) != []) and (ak.ravel(tau_plus2.pt[i]) == []) ) or ( (ak.ravel(tau_plus2.pt[i]) != []) and (ak.ravel(tau_plus1.pt[i]) == [])):
			#	print("!!Issue with comparing pt!!")
		
		#Construct each Higgs' transverse momenta
		lead_cond1 = np.bitwise_and(tau_plus1.pt > tau_minus1.pt, deltaR11 < deltaR12)
		lead_cond2 = np.bitwise_and(tau_plus1.pt > tau_minus1.pt, deltaR12 < deltaR11)
		lead_cond3 = np.bitwise_and(tau_plus1.pt < tau_minus1.pt, deltaR11 < deltaR21)
		lead_cond4 = np.bitwise_and(tau_plus1.pt < tau_minus1.pt, deltaR21 < deltaR11)
		
		PxLeading = np.append(ak.ravel(tau_plus1[lead_cond1].Px + tau_minus1[lead_cond1].Px),ak.ravel(tau_plus1[lead_cond2].Px + tau_minus2[lead_cond2].Px))
		PxLeading = np.append(PxLeading,ak.ravel(tau_plus1[lead_cond3].Px + tau_minus1[lead_cond3].Px))
		PxLeading = np.append(PxLeading,ak.ravel(tau_plus2[lead_cond4].Px + tau_minus1[lead_cond4].Px))
			
		PyLeading = np.append(ak.ravel(tau_plus1[lead_cond1].Py + tau_minus1[lead_cond1].Py),ak.ravel(tau_plus1[lead_cond2].Py + tau_minus2[lead_cond2].Py))
		PyLeading = np.append(PyLeading,ak.ravel(tau_plus1[lead_cond3].Py + tau_minus1[lead_cond3].Py))
		PyLeading = np.append(PyLeading,ak.ravel(tau_plus2[lead_cond4].Py + tau_minus1[lead_cond4].Py))
		
		PzLeading = np.append(ak.ravel(tau_plus1[lead_cond1].Pz + tau_minus1[lead_cond1].Pz),ak.ravel(tau_plus1[lead_cond2].Pz + tau_minus2[lead_cond2].Pz))
		PzLeading = np.append(PzLeading,ak.ravel(tau_plus1[lead_cond3].Pz + tau_minus1[lead_cond3].Pz))
		PzLeading = np.append(PzLeading,ak.ravel(tau_plus2[lead_cond4].Pz + tau_minus1[lead_cond4].Pz))
		
		ELeading = np.append(ak.ravel(tau_plus1[lead_cond1].E + tau_minus1[lead_cond1].E),ak.ravel(tau_plus1[lead_cond2].E + tau_minus2[lead_cond2].E))
		ELeading = np.append(ELeading,ak.ravel(tau_plus1[lead_cond3].E + tau_minus1[lead_cond3].E))
		ELeading = np.append(ELeading,ak.ravel(tau_plus2[lead_cond4].E + tau_minus1[lead_cond4].E))

		PxSubLeading = np.append(ak.ravel(tau_plus2[lead_cond1].Px + tau_minus2[lead_cond1].Px),ak.ravel(tau_plus2[lead_cond2].Px + tau_minus1[lead_cond2].Px))
		PxSubLeading = np.append(PxSubLeading,ak.ravel(tau_plus2[lead_cond3].Px + tau_minus2[lead_cond3].Px))
		PxSubLeading = np.append(PxSubLeading,ak.ravel(tau_plus1[lead_cond4].Px + tau_minus2[lead_cond4].Px))
			
		PySubLeading = np.append(ak.ravel(tau_plus2[lead_cond1].Py + tau_minus2[lead_cond1].Py),ak.ravel(tau_plus2[lead_cond2].Py + tau_minus1[lead_cond2].Py))
		PySubLeading = np.append(PySubLeading,ak.ravel(tau_plus2[lead_cond3].Py + tau_minus2[lead_cond3].Py))
		PySubLeading = np.append(PySubLeading,ak.ravel(tau_plus1[lead_cond4].Py + tau_minus2[lead_cond4].Py))
		
		PzSubLeading = np.append(ak.ravel(tau_plus2[lead_cond1].Pz + tau_minus2[lead_cond1].Pz),ak.ravel(tau_plus2[lead_cond2].Pz + tau_minus1[lead_cond2].Pz))
		PzSubLeading = np.append(PzSubLeading,ak.ravel(tau_plus2[lead_cond3].Pz + tau_minus2[lead_cond3].Pz))
		PzSubLeading = np.append(PzSubLeading,ak.ravel(tau_plus1[lead_cond4].Pz + tau_minus2[lead_cond4].Pz))
		
		ESubLeading = np.append(ak.ravel(tau_plus2[lead_cond1].E + tau_minus2[lead_cond1].E),ak.ravel(tau_plus2[lead_cond2].E + tau_minus1[lead_cond2].E))
		ESubLeading = np.append(ESubLeading,ak.ravel(tau_plus2[lead_cond3].E + tau_minus2[lead_cond3].E))
		ESubLeading = np.append(ESubLeading,ak.ravel(tau_plus1[lead_cond4].E + tau_minus2[lead_cond4].E))
		
		#Get pair delta R distributions
		leading_dR_Arr = np.append(ak.ravel(deltaR(tau_plus1[lead_cond1], tau_minus1[lead_cond1])), ak.ravel(deltaR(tau_plus1[lead_cond2],tau_minus2[lead_cond2])))	
		leading_dR_Arr = np.append(leading_dR_Arr,ak.ravel(deltaR(tau_plus1[lead_cond3], tau_minus1[lead_cond3])))
		leading_dR_Arr = np.append(leading_dR_Arr,ak.ravel(deltaR(tau_plus2[lead_cond4], tau_minus1[lead_cond4])))

		subleading_dR_Arr = np.append(ak.ravel(deltaR(tau_plus2[lead_cond1], tau_minus2[lead_cond1])),ak.ravel(deltaR(tau_plus2[lead_cond2], tau_minus1[lead_cond2])))	
		subleading_dR_Arr = np.append(subleading_dR_Arr,ak.ravel(deltaR(tau_plus2[lead_cond3], tau_minus2[lead_cond3])))		
		subleading_dR_Arr = np.append(subleading_dR_Arr,ak.ravel(deltaR(tau_plus1[lead_cond4], tau_minus2[lead_cond4])))		
	
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
		
		if (len(Higgs_Leading.eta) != 0):
			#if (self.isData):
			print("Mass Reconstructed")
			diHiggs_dR_Arr = ak.ravel(deltaR(Higgs_Leading,Higgs_SubLeading))
			print(len(diHiggs_dR_Arr))
			LeadingHiggs_mass_Arr = ak.ravel(single_mass(Higgs_Leading))	
			SubLeadingHiggs_mass_Arr = ak.ravel(single_mass(Higgs_SubLeading))
		else:
			#if (self.isData):
			print("Mass Not Reconstructed")
			diHiggs_dR_Arr = np.array([])
			LeadingHiggs_mass_Arr = np.array([])
			SubLeadingHiggs_mass_Arr = np.array([])

		#Look at individual pair delta Phi
		TauDeltaPhi_all_hist.fill("Leading pair", ak.ravel((tau_plus1[lead_cond1].phi - tau_minus1[lead_cond1].phi + np.pi) % (2 * np.pi) - np.pi))	
		TauDeltaPhi_all_hist.fill("Subleading pair", ak.ravel((tau_plus2[lead_cond1].phi - tau_minus2[lead_cond1].phi + np.pi) % (2 * np.pi) - np.pi))	
		TauDeltaPhi_all_hist.fill("Leading pair", ak.ravel((tau_plus1[lead_cond2].phi - tau_minus2[lead_cond2].phi + np.pi) % (2 * np.pi) - np.pi))	
		TauDeltaPhi_all_hist.fill("Subleading pair", ak.ravel((tau_plus2[lead_cond2].phi - tau_minus1[lead_cond2].phi + np.pi) % (2 * np.pi) - np.pi))	
		TauDeltaPhi_all_hist.fill("Leading pair", ak.ravel((tau_plus1[lead_cond3].phi - tau_minus1[lead_cond3].phi + np.pi) % (2 * np.pi) - np.pi))	
		TauDeltaPhi_all_hist.fill("Subleading pair", ak.ravel((tau_plus2[lead_cond3].phi - tau_minus2[lead_cond3].phi + np.pi) % (2 * np.pi) - np.pi))	
		TauDeltaPhi_all_hist.fill("Leading pair", ak.ravel((tau_plus2[lead_cond4].phi - tau_minus1[lead_cond4].phi + np.pi) % (2 * np.pi) - np.pi))	
		TauDeltaPhi_all_hist.fill("Subleading pair", ak.ravel((tau_plus1[lead_cond4].phi - tau_minus2[lead_cond4].phi + np.pi) % (2 * np.pi) - np.pi))	

		
		#Fill Higgs Delta Phi
		phi_leading = np.arctan2(PyLeading,PxLeading)
		phi_subleading = np.arctan2(PySubLeading,PxSubLeading)
		Higgs_DeltaPhi_Arr = ak.ravel((phi_leading - phi_subleading + np.pi) % (2 * np.pi) - np.pi)
		radionPT_HiggsReco = np.sqrt((PxLeading + PxSubLeading)**2 + (PyLeading + PySubLeading)**2)
		radionPT_Arr = ak.ravel(radionPT_HiggsReco)
		
		
		#RadionPTComp.fill(ak.ravel(radionPT),ak.ravel(radionPT_HiggsReco)) r"$\tau Reconstructed"
		RadionPTComp.fill(r"$\tau Reconstructed",ak.ravel(radionPT))		
		RadionPTComp.fill("Higgs Reconstructed",ak.ravel(radionPT_HiggsReco))		
	
		#print(len(tau))
		FourTau_Mass_Arr =four_mass([tau[:,0],tau[:,1],tau[:,2],tau[:,3]]) #ak.ravel(tau.FourMass)
	
		#Obtain the weight	
		if (self.isData):
			weight_Val = 1 
		else:
			weight_Val = weight_calc(dataset,tau,numEvents_Dict[dataset]) 
		
		#Efficiency Histograms
		
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
				"radionPT_Arr" : ak.to_list(radionPT_Arr)
			}
		}
	
	def postprocess(self, accumulator):
		pass	

if __name__ == "__main__":
	#mass_str_arr = ["1000","2000","3000"]
	mass_str_arr = ["2000"]

	#Structure to assemble all the input files of interest
	mass_number_dict = {"1000": "164152", "2000": "164237", "3000": "164320"}
	
	#Trigger dictionaries
	#trigger_dict = {"Mu50": (21,False), "PFMET120_PFMHT120_IDTight": (27,False), "EitherOr_Trigger": (41,True)}
	#trigger_dict = {"Mu50": (21,False), "PFMET120_PFMHT120_IDTight": (27,False), "EitherOr_Trigger": (41,True)}
	trigger_dict = {"EitherOr_Trigger": (41,True)}
	#trigger_dict = {"No_Trigger": (42,False)}
	#trigger_dict = {"PFHT500_PFMET100_PFMHT100_IDTight": (39,False), "AK8PFJet400_TrimMass30": (40,False), "EitherOr_Trigger": (41,True)}
	
	#Locations of files
	signal_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/GluGluToRadionToHHTo4T_M-"
	background_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"	
	data_loc = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"
	
	#signal_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2/GluGluToRadionToHHTo4T_M-"
	#background_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2/"	
	#data_loc = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2/"


	iterative_runner = processor.Runner(
		executor = processor.IterativeExecutor(compression=None),
		schema=BaseSchema
	)
	four_tau_hist_list = ["FourTau_Mass_Arr","HiggsDeltaPhi_Arr", "Higgs_DeltaR_Arr","leading_dR_Arr","subleading_dR_Arr","LeadingHiggs_mass","SubLeadingHiggs_mass", "radionPT_Arr"]
	#four_tau_hist_list = ["LeadingHiggs_mass"] #Only make 1 histogram for brevity/debugging purposes
	hist_name_dict = {"FourTau_Mass_Arr": r"Reconstructed 4-$\tau$ invariant mass", "HiggsDeltaPhi_Arr": r"Reconstructed Higgs $\Delta$ $\phi$", "Higgs_DeltaR_Arr": r"Reconstructed Higgs $\Delta$ R",
					"leading_dR_Arr": r"$\Delta$R of leading di-$\tau$ pair", "subleading_dR_Arr": r"$\Delta$R of subleading di-$\tau$ pair", 
					"LeadingHiggs_mass": r"Leading di-$\tau$ pair invariant mass", "SubLeadingHiggs_mass": r"Subleading di-$\tau$ pair invariant mass", "radionPT_Arr": r"Reconstructed Radion p_T"}
	#four_tau_hist_list = ["HiggsDeltaPhi_Arr","Pair_DeltaPhi_Hist"]



	for mass in mass_str_arr:
		print("====================Radion Mass = " + mass[0] + "." + mass[1] + " TeV====================")
		file_dict_test = { #Reduced files to run over
			"ZZ4l": [background_base + "ZZ4l.root"],
			"Signal": [signal_base + mass + ".root"],
			"Data_SingleMuon": [data_loc + "SingleMu_Run2018A.root", data_loc + "SingleMu_Run2018B.root", data_loc + "SingleMu_Run2018C.root", data_loc + "SingleMu_Run2018D.root"],
			"Data_JetHT": [data_loc + "JetHT_Run2018A-17Sep2018-v1.root", data_loc + "JetHT_Run2018B-17Sep2018-v1.root", data_loc + "JetHT_Run2018C-17Sep2018-v1.root",data_loc + "JetHT_Run2018D-PromptReco-v2.root"]
  
        }
		
		#Grand Unified Background + Signal Dictionary
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
			#"Data_SingleMuon": [data_loc + "SingleMuon_Run2018A-17Sep2018-v2_220117_194427_000" + str(np.floor(i/10))[0] + "_" + str(i % 10) + ".root" for i in range(20)] + 
			#					[data_loc + "SingleMuon_Run2018B-17Sep2018-v1_220120_131752_0000_" + str(i) + ".root" for i in range(10)] + 
			#					[data_loc + "SingleMuon_Run2018C-17Sep2018-v1_220117_194517_0000_" + str(i) + ".root" for i in range(10)] + 
			#					[data_loc + "SingleMuon_Run2018D-22Jan2019-v2_220117_173256_000" + str(np.floor(i/10))[0] + "_" + str(i % 10) + ".root" for i in range(50) if (i != 4 and i != 10 and i != 16)],
			#"Data_JetHT": [data_loc + "JetHT_Run2018A-17Sep2018-v1_220120_131700_000" + str(np.floor(i/10))[0] + "_" + str(i % 10) + ".root" for i in range(40)] + 
			#				[data_loc + "JetHT_Run2018B-17Sep2018-v1_220120_131726_000" + str(np.floor(i / 10))[0] + "_" + str(i % 10) + ".root" for i in range(40)] +
			#				[data_loc + "JetHT_Run2018C-17Sep2018-v1_220117_194402_0000_" + str(i % 10) + ".root" for i in range(10)] + 
			#				[data_loc + "JetHT_Run2018D-PromptReco-v2_220123_233624_000" + str(np.floor(i/10))[0] + "_" + str(i % 10) + ".root" for i in range(70)]
			"Data_SingleMuon": [data_loc + "SingleMu_Run2018A.root", data_loc + "SingleMu_Run2018B.root", data_loc + "SingleMu_Run2018C.root", data_loc + "SingleMu_Run2018D.root"],
			"Data_JetHT": [data_loc + "JetHT_Run2018A-17Sep2018-v1.root", data_loc + "JetHT_Run2018B-17Sep2018-v1.root", data_loc + "JetHT_Run2018C-17Sep2018-v1.root",data_loc + "JetHT_Run2018D-PromptReco-v2.root"]
		}
		
		#event_num_dict = {}
		#Generate dictionary of number of processed events
		for key_name, file_name in file_dict.items(): 
			if (file_name != "Data_JetHT" or file_name != "Data_SingleMuon"):
				tempFile = uproot.open(file_name[0]) #Get file
				#print("Input Type:")
				#print(key_name)
				#isData = False
				numEvents_Dict[key_name] = tempFile['hEvents'].member('fEntries')/2
				#print(tempFile['hEvents'].member('fEntries')/2)
			else: #Ignore data files
				continue


		hist_dict_background = {
			"FourTau_Mass_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Regular(20,0,3000, label = r"$m_{4\tau}$ [GeV]").Double(),
			"HiggsDeltaPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(20,-pi,pi, label = r"Higgs $\Delta \phi$").Double(), 
			"Higgs_DeltaR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(20,0,5, label = r"Higgs $\Delta$R").Double(),
			"leading_dR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(20,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
			"subleading_dR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(20,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double(),
			"LeadingHiggs_mass" : hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(12,0,120, label=r"Leading di-$\tau$ Mass (GeV)").Double(),
			"SubLeadingHiggs_mass" : hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(12,0,120, label=r"Sub-Leading di-$\tau$ Mass (GeV)").Double(),
			"radionPT_Arr" : hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(20,0,500, label=r"Radion $p_T$ (GeV)").Double()
		}
		
		hist_dict_signal = {
			"FourTau_Mass_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Regular(20,0,3000, label = r"$m_{4\tau}$ [GeV]").Double(),
			"HiggsDeltaPhi_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(20,-pi,pi, label = r"Higgs $\Delta \phi$").Double(), 
			"Higgs_DeltaR_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(20,0,5, label = r"Higgs $\Delta$R").Double(),
			"leading_dR_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(20,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
			"subleading_dR_Arr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(20,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double(),
			"LeadingHiggs_mass" : hist.Hist.new.StrCat(["Signal"],name="signal").Reg(12,0,120, label=r"Leading di-$\tau$ Mass (GeV)").Double(),
			"SubLeadingHiggs_mass" : hist.Hist.new.StrCat(["Signal"],name="signal").Reg(12,0,120, label=r"Sub-Leading di-$\tau$ Mass (GeV)").Double(),
			"radionPT_Arr" : hist.Hist.new.StrCat(["Signal"],name="signal").Reg(20,0,500, label=r"Radion $p_T$ (GeV)").Double()
		}
		hist_dict_data = {
			"FourTau_Mass_Arr": hist.Hist.new.StrCat(["Data"],name="data").Regular(20,0,3000, label = r"$m_{4\tau}$ [GeV]").Double(),
			"HiggsDeltaPhi_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(20,-pi,pi, label = r"Higgs $\Delta \phi$").Double(), 
			"Higgs_DeltaR_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(20,0,5, label = r"Higgs $\Delta$R").Double(),
			"leading_dR_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(20,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
			"subleading_dR_Arr": hist.Hist.new.StrCat(["Data"],name="data").Reg(20,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double(),
			"LeadingHiggs_mass" : hist.Hist.new.StrCat(["Data"],name="data").Reg(12,0,120, label=r"Leading di-$\tau$ Mass (GeV)").Double(),
			"SubLeadingHiggs_mass" : hist.Hist.new.StrCat(["Data"],name="data").Reg(12,0,120, label=r"Sub-Leading di-$\tau$ Mass (GeV)").Double(),
			"radionPT_Arr" : hist.Hist.new.StrCat(["Data"],name="data").Reg(20,0,500, label=r"Radion $p_T$ (GeV)").Double()
		}
		
		background_list = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"]
		#background_list = [r"$ZZ \rightarrow 4l$"]
		signal_list = [r"MC Sample $m_\phi$ = %s TeV"%mass[0]]
		background_plot_names = {r"$t\bar{t}$" : "_ttbar_", r"Drell-Yan+Jets": "_DYJets_", "Di-Bosons" : "_DiBosons_", "Single Top": "_SingleTop+", "QCD" : "_QCD_", "W+Jets" : "_WJets_", r"$ZZ \rightarrow 4l$" : "_ZZ4l_"}
		background_dict = {r"$t\bar{t}$" : ["TTToSemiLeptonic","TTTo2L2Nu","TTToHadronic"], 
				r"Drell-Yan+Jets": ["DYJetsToLL_Pt-50To100","DYJetsToLL_Pt-100To250","DYJetsToLL_Pt-250To400","DYJetsToLL_Pt-400To650","DYJetsToLL_Pt-650ToInf"], 
				"Di-Bosons": ["WZ3l1nu","WZ2l2q","WZ1l1nu2q","ZZ2l2q","WZ3l1nu", "WZ1l3nu", "VV2l2nu"], "Single Top": ["Tbar-tchan","T-tchan","Tbar-tW","T-tW"], 
				"W+Jets": ["WJetsToLNu_HT-100To200","WJetsToLNu_HT-200To400","WJetsToLNu_HT-400To600","WJetsToLNu_HT-600To800","WJetsToLNu_HT-800To1200","WJetsToLNu_HT-1200To2500","WJetsToLNu_HT-2500ToInf"],
				r"$ZZ \rightarrow 4l$" : ["ZZ4l"]
		}
		#background_dict = {r"$ZZ \rightarrow 4l$" : ["ZZ4l"]}

		for trigger_name, trigger_pair in trigger_dict.items():
			four_tau_names = {"FourTau_Mass_Arr": "FourTauMass_" + mass + "-" + trigger_name, "HiggsDeltaPhi_Arr": "HiggsDeltaPhi_" + mass + "-" + trigger_name, 
				"Pair_DeltaPhi_Hist": "TauPair_DeltaPhi_" + mass + "-" + trigger_name, "RadionPTComp_Hist": "pTReco_Comp_"+mass+ "-"+ trigger_name,
				"Higgs_DeltaR_Arr": "Higgs_DeltaR_" + mass + "-" + trigger_name, "leading_dR_Arr": "leading_diTau_DeltaR_" + mass + "-" + trigger_name, 
				"subleading_dR_Arr": "subleading_diTau_DeltaR_" + mass + "-" + trigger_name, "LeadingHiggs_mass" : "LeadingHiggs_Mass_"+ mass + "-" + trigger_name, 
				"SubLeadingHiggs_mass" : "SubLeadingHiggs_Mass_" + mass + "-" + trigger_name, "radionPT_Arr": "radion_pT_Mass_" + mass + "-" + trigger_name
			}
			#if ()
			
			fourtau_out = iterative_runner(file_dict, treename="4tau_tree", processor_instance=FourTauPlotting(trigger_bit=trigger_pair[0], or_trigger=trigger_pair[1]))
			for hist_name in four_tau_hist_list: #Loop over all histograms
				#fig,ax = plt.subplots()
				fig0,ax0 = plt.subplots()
				if (hist_name != "Pair_DeltaPhi_Hist" and hist_name != "RadionPTComp_Hist"):
					hist_dict_only_signal = {
						"FourTau_Mass_Arr": hist.Hist.new.Regular(20,0,3000, label = r"$m_{4\tau}$ [GeV]").Double(),
						"HiggsDeltaPhi_Arr": hist.Hist.new.Regular(20,-pi,pi, label = r"Higgs $\Delta \phi$").Double(), 
						"Higgs_DeltaR_Arr": hist.Hist.new.Regular(20,0,5, label = r"Higgs $\Delta$R").Double(),
						"leading_dR_Arr": hist.Hist.new.Regular(20,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
						"subleading_dR_Arr": hist.Hist.new.Regular(20,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double(),
						"LeadingHiggs_mass" : hist.Hist.new.Regular(20,0,120, label=r"Leading Higgs Mass (GeV)").Double(),
						"SubLeadingHiggs_mass" : hist.Hist.new.Regular(20,0,120, label=r"Sub-Leading Higgs Mass (GeV)").Double(),
						"radionPT_Arr" : hist.Hist.new.Regular(20,0,500, label=r"Radion $p_T$ (GeV)").Double()
					}
					#hist_dict_signal[hist_name].fill(fourtau_out["Signal"][hist_name],weight=fourtau_out["Signal"]["Weight"]*1/len(fourtau_out["Signal"][hist_name]))
					hist_dict_only_signal[hist_name].fill(fourtau_out["Signal"][hist_name],weight=fourtau_out["Signal"]["Weight"])
					hist_dict_only_signal[hist_name].plot1d(ax=ax0)
					plt.title("Signal Mass: " + mass[0] + "TeV")
					plt.savefig("SignalSingle" + four_tau_names[hist_name])
					plt.close()

				if (hist_name != "Pair_DeltaPhi_Hist" and hist_name != "RadionPTComp_Hist"):
					for background_type in background_list:
						hist_dict_single_background = {
							"FourTau_Mass_Arr": hist.Hist.new.Regular(20,0,3000, label = r"$m_{4\tau}$ [GeV]").Double(),
							"HiggsDeltaPhi_Arr": hist.Hist.new.Regular(20,-pi,pi, label = r"Higgs $\Delta \phi$").Double(), 
							"Higgs_DeltaR_Arr": hist.Hist.new.Regular(20,0,5, label = r"Higgs $\Delta$R").Double(),
							"leading_dR_Arr": hist.Hist.new.Regular(20,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
							"subleading_dR_Arr": hist.Hist.new.Regular(20,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double(),
							"LeadingHiggs_mass" : hist.Hist.new.Regular(20,0,120, label=r"Leading Higgs Mass (GeV)").Double(),
							"SubLeadingHiggs_mass" : hist.Hist.new.Regular(20,0,120, label=r"Sub-Leading Higgs Mass (GeV)").Double(),
							"radionPT_Arr" : hist.Hist.new.Regular(20,0,500, label=r"Radion $p_T$ (GeV)").Double()
						}
						background_array = []
						backgrounds = background_dict[background_type]
						
						#Loop over all backgrounds
						for background in backgrounds:
							#print("Current background = %s"%background)
							#print("Number of entries = %d"%len(fourtau_out[background][hist_name]))
							#print("Cross Section = %.3f"%xSection_Dictionary[background])
							#print("Background Weight = %.3f"%fourtau_out[background]["Weight"])
							if (mass == "1000"): #Only need to generate single background once
								fig2, ax2 = plt.subplots()
								hist_dict_single_background[hist_name].fill(fourtau_out[background][hist_name],weight=fourtau_out[background]["Weight"]) #Obtain background distributions 
								hist_dict_single_background[hist_name].plot1d(ax=ax2)
								plt.title(background_type)
								plt.savefig("SingleBackground" + background_plot_names[background_type] + four_tau_names[hist_name])
								plt.close()
							if ((background_type == r"$t\bar{t}$" or background_type == "Drell-Yan+Jets" or background_type == r"$ZZ \rightarrow 4l$") and hist_name == "LeadingHiggs_mass"):
								print("!!!====================Debugging di-tau masses of %s====================!!!"%background_type)
								zeroMassCounter = 0
								firstbin_count = 0
								for ditau_mass in fourtau_out[background][hist_name]:
									if (ditau_mass == 0):
										zeroMassCounter += 1
									if (ditau_mass >= 0 and ditau_mass <= 10):
										firstbin_count += 1
										print(ditau_mass)
								print("# of entries with 0 mass: %d"%zeroMassCounter)
								print("# of entries in first bin: %d"%firstbin_count)
								
						
							hist_dict_background[hist_name].fill(background_type,fourtau_out[background][hist_name],weight=fourtau_out[background]["Weight"]) #Obtain background distributions
							#print("Number of background entries for %s: %d"%(background,len(fourtau_out[background][hist_name])))
							#hist_dict[hist_name].fill(background_type,fourtau_out[background][hist_name],weight=fourtau_out[background]["Weight"]) #Obtain background distributions
							
					
					#background_array = np.append(background_array,fourtau_out[background][hist_name])
					#print("Number of Signal Entries: %d"%len(fourtau_out["Signal"][hist_name]))
					hist_dict_signal[hist_name].fill("Signal",fourtau_out["Signal"][hist_name],weight=fourtau_out["Signal"]["Weight"]) #Obtain signal distribution
					#print("Number of Single Muon entries: %d"%len(fourtau_out["Data_SingleMuon"][hist_name]))
					
					#Obtain data distributions
					print("==================Hist %s================"%hist_name)
					print("Total amount of data = %d"%(len(fourtau_out["Data_SingleMuon"][hist_name]) + len(fourtau_out["Data_JetHT"][hist_name])))
					hist_dict_data[hist_name].fill("Data",fourtau_out["Data_SingleMuon"][hist_name]) #,weight=fourtau_out["Data_SingleMuon"]["Weight"])
					hist_dict_data[hist_name].fill("Data",fourtau_out["Data_JetHT"][hist_name]) #,weight=fourtau_out["Data_JetHT"]["Weight"])
					#print("Number of Jet HT entries: %d"%len(fourtau_out["Data_JetHT"][hist_name]))
					
					#Put histograms into stacks and arrays for polotting purposes
					background_stack = hist_dict_background[hist_name].stack("background")
					signal_stack = hist_dict_signal[hist_name].stack("signal")
					data_stack = hist_dict_data[hist_name].stack("data")
					signal_array = [signal_stack["Signal"]]
					data_array = [data_stack["Data"]]
					for background in background_list:
						background_array.append(background_stack[background])
					
					#Stack background distributions and signal distribution
					fig,ax = plt.subplots()
					hep.histplot(background_array,ax=ax,stack=True,histtype="fill",label=background_list)
					hep.histplot(signal_array,ax=ax,stack=True,histtype="step",label=signal_list)
					hep.histplot(data_array,ax=ax,stack=True,histtype="errorbar", yerr=False,label=["Data"],marker="o")
					hep.cms.text("Preliminary",loc=0,fontsize=13)
					ax.set_title(hist_name_dict[hist_name],loc = "right")
					ax.legend(fontsize=10, loc='upper right')
					#ax.set_yscale("log")
					#ax.legend(title="Legend")
					plt.savefig(four_tau_names[hist_name])
					plt.close()

