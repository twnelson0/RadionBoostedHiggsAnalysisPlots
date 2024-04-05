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
xSection_Dictionary = {"Signal": 0, #Chosen to make plots readable 
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
	def __init__(self, trigger_bit, trigger_cut = True, offline_cut = False, or_trigger = False, is_data = False):
		self.trigger_bit = trigger_bit
		self.offline_cut = offline_cut
		self.trigger_cut = trigger_cut
		self.OrTrigger = or_trigger
		self.isData = is_data
		#pass

	def process(self, events):
		dataset = events.metadata['dataset']
		tau_reco = ak.zip(
			{
				"trigger": events.HLTJet,
			}
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
				"trigger": events.HLTJet,
				"pfMET": events.pfMET,
				#"hcount": events.hcount,
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
				"trigger": events.HLTJet,
					
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
				"trigger": events.HLTJet,
					
			},
			with_name="ElectronArray",
			behavior=candidate.behavior,
			
		)

		AK8Jet = ak.zip(
			{
				"AK8JetDropMass": events.AK8JetSoftDropMass,
				"AK8JetPt": events.AK8JetPt,
				"nMu": events.nMu,
				"eta": events.AK8JetEta,
				"phi": events.AK8JetPhi,
				"nEle": events.nEle,
				"trigger": events.HLTJet,
			},
			with_name="AK8JetArray",
			behavior=candidate.behavior,
		)
		
		Jet = ak.zip(
			{
				"Pt": events.jetPt,
				"pfMET": events.pfMET,
				"PtTotUncDown": events.jetPtTotUncDown,
				"PtTotUncUp": events.jetPtTotUncUp,
				"PFLooseId": events.jetPFLooseId,
				"HT": ak.sum(events.jetPt, axis=1),
				"eta": events.jetEta,
				"phi": events.jetPhi,
				"trigger": events.HLTJet,
			},
			with_name="PFJetArray",
			behavior=candidate.behavior,
		)
		
		#Histograms (AK8Jet) (Trigger bit = 40)
		AK8Pt_PreTrigg = hist.Hist.new.Reg(40, 0, 1100, name = "JetPt_Trigg", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8Pt_all = hist.Hist.new.StrCat(["No Trigger","Trigger"], name = "AK8Pt_hist").Reg(40,0,1100, name="AK8Pt", label = "AK8 Jet r$p_T$ [GeV]").Double()	
		AK8SoftMass_all = hist.Hist.new.StrCat(["No Trigger","Trigger"], name = "AK8SoftMass_hist").Reg(40,0,400, name="AK8SoftMass", label = "AK8 Jet Soft Drop Mass [GeV]").Double()

		AK8Pt_NoCut = hist.Hist.new.Reg(40, 0, 1800, name = "JetPt_NoCut", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8Pt_Trigg = hist.Hist.new.Reg(40, 0, 1100, name = "JetPt_Trigg", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8SoftMass_PreTrigg = hist.Hist.new.Reg(40, 0, 300, name = "SoftMass_Trigg", label = "AK8Jet Soft Mass [GeV]").Double()
		AK8SoftMass_NoCut = hist.Hist.new.Reg(40, 0, 300, name = "SoftMass_NoCut", label = "AK8Jet Soft Mass [GeV]").Double()
		AK8SoftMass_Trigg = hist.Hist.new.Reg(40, 0, 300, name = "SoftMass_Trigg", label = "AK8Jet Soft Mass [GeV]").Double()		
		
		#2D Histograms
		AK8Jet_PreTrigger = hist.Hist(
						hist.axis.Regular(20, 0, 1100, name="JetPt", label=r"AK8Jet $p_T$ [GeV]"),
						hist.axis.Regular(10, 0, 300, name="SoftMass", label=r"AK8Jet Soft Mass [GeV]")
					)		
		AK8Jet_Trigger = hist.Hist(
						hist.axis.Regular(20, 0, 1100, name="JetPt", label=r"AK8Jet $p_T$ [GeV]"),
						hist.axis.Regular(10, 0, 300, name="SoftMass", label="AK8Jet Soft Mass [GeV]")
					)		
		eff_AK8Jet = hist.Hist(
						hist.axis.Regular(20, 0, 1100, name="JetPt", label=r"AK8Jet $p_T$ [GeV]"),
						hist.axis.Regular(10, 0, 300, name="SoftMass", label="AK8Jet Soft Mass [GeV]")
					)		

		#Histograms (MET and HT) (Trigger bit = 39)
		HT_PreTrigg = hist.Hist.new.Reg(40, 0, 3500., label = "HT [GeV]").Double()
		HT_NoCut = hist.Hist.new.Reg(40, 0, 3500., label = "HT [GeV]").Double()
		HT_NoCrossCleaning = hist.Hist.new.Reg(40, 0, 3500., label = "HT [GeV]").Double()
		HT_Trigg = hist.Hist.new.Reg(40, 0, 3500., label = "HT [GeV]").Double()
		MET_PreTrigg = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_NoCut = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_NoCrossCleaning = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_Trigg = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()

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
		#Higgs_DeltaR = hist.Hist.new.Reg().Double()
		#LeadingPair_dR = hist.Hist.new.Reg().Double()
		#SubleadingPair_dR = hist.Hist.new.Reg().Double()
		

		print("!!!=====Dataset=====!!!!")	
		print(type(dataset))
		print(dataset)
		print("!!!=====Dataset=====!!!!")	
		print("Number of events before selection + Trigger: %d"%ak.num(tau,axis=0))
		#MHT
		Jet_MHT = Jet[Jet.Pt > 30]
		Jet_MHT = Jet[np.abs(Jet.eta) < 5]
		Jet_MHT = Jet_MHT[Jet_MHT.PFLooseId > 0.5]
		JetUp_MHT = Jet[Jet.PtTotUncUp > 30]
		JetUp_MHT = JetUp_MHT[np.abs(JetUp_MHT.eta) < 5]
		JetUp_MHT = JetUp_MHT[JetUp_MHT.PFLooseId > 0.5]
		JetDown_MHT = Jet[Jet.PtTotUncDown > 30]
		JetDown_MHT = JetDown_MHT[np.abs(JetDown_MHT.eta) < 5]
		JetDown_MHT = JetDown_MHT[JetDown_MHT.PFLooseId > 0.5]
		Jet_MHT["MHT_x"] = ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False) + ak.sum(JetUp_MHT.PtTotUncUp*np.cos(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.cos(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet_MHT["MHT_y"] = ak.sum(Jet_MHT.Pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False) + ak.sum(JetUp_MHT.PtTotUncUp*np.sin(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.sin(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet_MHT["MHT"] = np.sqrt(Jet_MHT.MHT_x**2 + Jet_MHT.MHT_y**2)
		#print("Jet MHT Defined:")
		
		#HT Seleciton (new)
		#tau_temp1,HT = ak.unzip(ak.cartesian([tau,Jet_MHT], axis = 1, nested = True))
		Jet_HT = Jet[Jet.Pt > 30]
		Jet_HT = Jet_HT[np.abs(Jet_HT.eta) < 3]
		Jet_HT = Jet_HT[Jet_HT.PFLooseId > 0.5]
		JetUp_HT = Jet[Jet.PtTotUncUp > 30]
		JetUp_HT = JetUp_HT[np.abs(JetUp_HT.eta) < 3]
		JetUp_HT = JetUp_HT[JetUp_HT.PFLooseId > 0.5]
		JetDown_HT = Jet[Jet.PtTotUncDown > 30]
		JetUp_HT = JetDown_HT[np.abs(JetDown_HT.eta) < 3]
		JetDown_HT = JetDown_HT[JetDown_HT.PFLooseId > 0.5]
		HT,tau_temp1 = ak.unzip(ak.cartesian([Jet_HT,tau], axis = 1, nested = True))
		HT_up,tau_temp2 = ak.unzip(ak.cartesian([JetUp_HT,tau], axis = 1, nested = True))
		HT_down,tau_temp3 = ak.unzip(ak.cartesian([JetDown_HT,tau], axis = 1, nested = True))
		
		#Get Cross clean free histograms
		HT_Var_NoCrossClean = ak.sum(Jet_HT.Pt,axis = 1,keepdims=True) + ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
		MET_NoCrossCleaning.fill(ak.ravel(Jet.pfMET))
		MET_Acc_NoCrossClean = hist.accumulators.Mean().fill(ak.ravel(HT_Var_NoCrossClean[HT_Var_NoCrossClean > 0]))
		HT_NoCrossCleaning.fill(ak.ravel(HT_Var_NoCrossClean[HT_Var_NoCrossClean > 0]))
		HT_num = 0
		for x in ak.sum(Jet.Pt,axis = 1,keepdims=False):
			HT_num += 1
			
		#print(ak.sum(Jet.Pt,axis = 1,keepdims=True))
		#print("HT Num (No Cross Cleaning): %d"%HT_num)
		HT_Acc_NoCrossClean = hist.accumulators.Mean().fill(ak.ravel(HT_Var_NoCrossClean[HT_Var_NoCrossClean > 0]))
	
		mval_temp = deltaR(tau_temp1,HT) >= 0.5
		print(mval_temp)
		if (len(Jet.Pt) != len(mval_temp)):
			print("Things aren't good")
			if (len(Jet.Pt) > len(mval_temp)):
				print("More Jets than entries in mval_temp")
			if (len(Jet.Pt) < len(mval_temp)):
				print("Fewer entries in Jets than mval_temp")

		Jet_HT["dR"] = mval_temp
		mval_temp = deltaR(tau_temp2,HT_up) >= 0.5
		JetUp_HT["dR"] = mval_temp 
		mval_temp = deltaR(tau_temp3,HT_down) >= 0.5
		JetDown_HT["dR"] = mval_temp

		HT_Val_NoCuts = ak.sum(Jet_HT.Pt,axis = 1,keepdims=True) + ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
		test_HT = ak.sum(Jet.Pt,axis = 1,keepdims=True)
		HT_num = 0
		for x in ak.ravel(Jet.HT):
			HT_num += 1

		#Histograms of variables relavent to trigger 
		if (self.trigger_bit == 40):
			AK8Jet_Temp = AK8Jet[ak.num(AK8Jet, axis = 1) > 0]
			AK8Pt_NoCut.fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetPt)) #Change to leading pt
			AK8Pt_Acc = hist.accumulators.Mean().fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetPt))
			AK8SoftMass_NoCut.fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetDropMass))
			AK8SoftMass_Acc = hist.accumulators.Mean().fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetDropMass))
		if (self.trigger_bit == 39):
			#print("No Cut Accumulators updated")
			HT_NoCut.fill(ak.ravel(HT_Val_NoCuts[HT_Val_NoCuts > 0]))
			HT_Acc_NoCut = hist.accumulators.Mean().fill(ak.ravel(HT_Val_NoCuts[HT_Val_NoCuts > 0]))
			#print("HT Mean: %f"%HT_Acc_NoCut.value)
			MET_NoCut.fill(ak.ravel(Jet.pfMET))
			MET_Acc_NoCut = hist.accumulators.Mean().fill(ak.ravel(Jet.pfMET))
			#print("MET Mean: %f"%MET_Acc_NoCut.value)
				

		trigger_mask = bit_mask([self.trigger_bit])		
		tau = tau[tau.pt > 20] #pT
		tau = tau[np.abs(tau.eta) < 2.3] #eta
		
		#Loose isolation
		tau = tau[tau.decay >= 0.5]	
		tau = tau[tau.iso >= 0.5]


		#Delta R Cut on taus
		a,b = ak.unzip(ak.cartesian([tau,tau], axis = 1, nested = True))
		mval = deltaR(a,b) < 0.8 
		tau["dRCut"] = mval
		tau = tau[ak.any(tau.dRCut, axis = 2) == True]	
		
		AK8Jet = AK8Jet[(ak.sum(tau.charge,axis=1) == 0)] #Apply charge conservation cut to AK8Jets
		Jet = Jet[(ak.sum(tau.charge,axis=1) == 0)]
		Jet_HT = Jet_HT[(ak.sum(tau.charge,axis=1) == 0)]
		JetUp_HT = JetUp_HT[(ak.sum(tau.charge,axis=1) == 0)]
		JetDown_HT = JetDown_HT[(ak.sum(tau.charge,axis=1) == 0)]
		Jet_MHT = Jet_MHT[(ak.sum(tau.charge,axis=1) == 0)]
		electron = electron[(ak.sum(tau.charge,axis=1) == 0)]
		muon = muon[(ak.sum(tau.charge,axis=1) == 0)]
		tau_reco = tau_reco[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		#Higgs_Leading = Higgs_Leading[(ak.sum(tau.charge,axis=1) == 0)]
		#Higgs_SubLeading = Higgs_SubLeading[(ak.sum(tau.charge,axis=1) == 0)]
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation

		
		#Investegate entries with more than 4 taus
		# n_more = len(tau[ak.num(tau) > 4])
		# print("Events with more than 4 taus: %d"%n_more)
		
		# if (n_more > 0):
		# 	print("========!!Important information about events with more than 4 tau!!========")
		# 	diff_num = n_more
		# 	test_obj = tau[ak.num(tau) > 4] 
		# 	n = 5s
		# 	while(n_more > 0):
		# 		N_events = len(test_obj[ak.num(test_obj) == 5])
		# 		print("Number of events with %d taus: %d"%(n,N_events))
		# 		n +=1
		# 		diff_num -= N_events
		

		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		Jet_MHT = Jet_MHT[ak.num(tau) >= 4]
		Jet_HT = Jet_HT[ak.num(tau) >= 4]
		JetUp_HT = JetUp_HT[ak.num(tau) >= 4]
		JetDown_HT = JetDown_HT[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		electron = electron[ak.num(tau) >= 4] 
		muon = muon[ak.num(tau) >= 4] 
		tau_reco = tau_reco[ak.num(tau) >= 4] 
		#Higgs_Leading = Higgs_Leading[ak.num(tau) >= 4] 
		#Higgs_SubLeading = Higgs_SubLeading[ak.num(tau) >= 4] 
		tau = tau[ak.num(tau) >= 4] #4 tau events

		#Add momenta to tau_reco
		#electron_dR = deltaR(tau,electron) 
		#muon_dR = deltaR(tau,muon)
			
		#Reco variables
		reco_pt = []
		reco_E = []
		reco_eta = []
		reco_phi = []
		
		#Loop over all events
		#for (ele_event,mu_event,tau_event) in zip(electron,muon,tau):
			#Reco variables
		#	event_pt = []
		#	event_E = []
		#	event_eta = []
		#	event_phi = []
			
			#Find where a tau should be a lepton of a different flavor
		#	for tau_temp in tau_event:
		#		ele_dR = deltaR(tau_temp,ele_event)
		#		mu_dR = deltaR(tau_temp,mu_event)
		#		ele_event = ele_event[ele_dR < 0.05]
		#		ele_dR = ele_dR[ele_dR < 0.05]
		#		mu_event = mu_event[mu_dR < 0.05]
		#		mu_dR = mu_dR[mu_dR < 0.05]
		#		if (ak.num(ele_event.pt,axis=0) >= 1): #Use Electron
					#print("Electron Used")
		#			ele_event = ele_event[ele_dR <= min(ele_dR)]
		#			event_pt.append(ele_event.pt[0])
		#			event_E.append(ele_event.E[0])
		#			event_eta.append(ele_event.eta[0])
		#			event_phi.append(ele_event.phi[0])
		#			continue
		#		if (ak.num(mu_event.pt,axis=0) >= 1): #Use Muon
					#print("Muon Used")
					#print(mu_dR)
		#			mu_event = mu_event[mu_dR <= min(mu_dR)]
		#			event_pt.append(mu_event.pt[0])
		#			event_E.append(mu_event.E[0])
		#			event_eta.append(mu_event.eta[0])
		#			event_phi.append(mu_event.phi[0])
		#			continue
		#		if (ak.num(mu_event.pt,axis=0) == 0 and ak.num(ele_event.pt,axis=0) == 0): #Use tau
		#			event_pt.append(tau_temp.pt)
		#			event_E.append(tau_temp.E)
		#			event_eta.append(tau_temp.eta)
		#			event_phi.append(tau_temp.phi)
			
			#Add to reco arrays
			#reco_pt.append(event_pt)
			#reco_E.append(event_E)
			#reco_eta.append(event_eta)
			#reco_phi.append(event_phi)
				
			
		#Populate tau reco object
		#tau_reco["pt"] = ak.from_iter(reco_pt)
		#tau_reco["E"] = ak.from_iter(reco_E)
		#tau_reco["eta"] = ak.from_iter(reco_eta)
		#tau_reco["phi"] = ak.from_iter(reco_phi)
		#tau_reco["Px"] = ak.from_iter(reco_pt*np.cos(reco_phi))		
		#tau_reco["Py"] = ak.from_iter(reco_pt*np.cos(reco_phi))
		#tau_reco["Pz"] = ak.from_iter(reco_pt*np.tan(2*np.arctan(np.exp(reco_eta)))**-1)
		#tau_reco["charge"] = ak.from_iter(tau.charge)
		
		#print(tau[:,0].pt)
		#tau = tau_reco #Cheap way to use my tau reco variable	
		#print(tau.pt)
		#print(tau[:])
		#Debugging stuff
		#for evnt in tau.pt:
		#	if (len(evnt) == 0):
		#		print("!!0 Length Issue!!")
		
		#print(tau[:,0])	
		#print(tau[:,0].pt)	
		tau["FourMass"] = four_mass([tau[:,0],tau[:,1],tau[:,2],tau[:,3]])
		
		if (self.OrTrigger): #Select for both triggers
			tau = tau[np.bitwise_and(tau.trigger,bit_mask([21,27])) != 0]
			AK8Jet = AK8Jet[np.bitwise_and(AK8Jet.trigger,bit_mask([21,27])) != 0]
			Jet = Jet[np.bitwise_and(Jet.trigger,bit_mask([21,27])) != 0]
			Jet_HT = Jet_HT[np.bitwise_and(Jet_HT.trigger,bit_mask([21,27])) != 0]
			Jet_MHT = Jet_MHT[np.bitwise_and(Jet_MHT.trigger,bit_mask([21,27])) != 0]
			
			#Jet["HT"] = ak.sum(Jet_HT.Pt, axis = 1, keepdims=False) + ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=False)
			#tau = tau[np.bitwise_and(tau.trigger,bit_mask([39,40])) != 0]
			#AK8Jet = AK8Jet[np.bitwise_and(AK8Jet.trigger,bit_mask([39,40])) != 0]
			#Jet = Jet[np.bitwise_and(Jet.trigger,bit_mask([39,40])) != 0]
			#Jet_HT = Jet_HT[np.bitwise_and(Jet_HT.trigger,bit_mask([39,40])) != 0]
			#Jet_MHT = Jet_MHT[np.bitwise_and(Jet_MHT.trigger,bit_mask([39,40])) != 0]
		else: #Single Trigger
			if (self.trigger_bit == 40):
				AK8Pt_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetPt))
				AK8Pt_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
				AK8SoftMass_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
				AK8SoftMass_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
				AK8Pt_all.fill("No Trigger",ak.ravel(AK8Jet.AK8JetPt))
			
			if (self.trigger_bit == 39):
				#Fill Histograms
				HT_Val_PreTrigger = ak.sum(Jet_HT.Pt, axis = 1, keepdims=True) + ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
				Jet["HT"] = ak.sum(Jet_HT.Pt, axis = 1, keepdims=False) + ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=False)
				HT_PreTrigg.fill(ak.ravel(HT_Val_PreTrigger[HT_Val_PreTrigger > 0]))
				#HT_NoTrigg_Arr = ak.ravel(HT_Val_PreTrigger[HT_Val_PreTrigger > 0])
				HT_NoTrigg_Arr = ak.ravel(Jet.HT)
				MET_PreTrigg.fill(ak.ravel(Jet.pfMET))
				MET_NoTrigg_Arr = ak.ravel(Jet.pfMET)
				print("MET Len: %d"%len(MET_NoTrigg_Arr))	
				print("HT Len: %d"%len(HT_NoTrigg_Arr))	

			if (self.trigger_bit != None and self.OrTrigger != False):
				tau = tau[np.bitwise_and(tau.trigger,trigger_mask) == trigger_mask]
				AK8Jet = AK8Jet[np.bitwise_and(AK8Jet.trigger,trigger_mask) == trigger_mask]
				Jet = Jet[np.bitwise_and(Jet.trigger,trigger_mask) == trigger_mask]
				Jet_HT = Jet_HT[np.bitwise_and(Jet_HT.trigger,trigger_mask) == trigger_mask]
				Jet_MHT = Jet_MHT[np.bitwise_and(Jet_MHT.trigger,trigger_mask) == trigger_mask]
			#if (self.OrTrigger == True):		
			#	tau = tau[np.bitwise_and(tau.trigger,bit_mask([21,27])) != 0]
			#	AK8Jet = AK8Jet[np.bitwise_and(AK8Jet.trigger,bit_mask([21,27])) != 0]
			#	Jet = Jet[np.bitwise_and(Jet.trigger,bit_mask([21,27])) != 0]
			#	Jet_HT = Jet_HT[np.bitwise_and(Jet_HT.trigger,bit_mask([21,27])) != 0]
			#	Jet_MHT = Jet_MHT[np.bitwise_and(Jet_MHT.trigger,bit_mask([21,27])) != 0]
		
		print("Number of events after selection + Trigger: %d"%ak.num(tau,axis=0))
		
		#Construct all possible valid ditau pairs
		tau_plus = tau[tau.charge > 0]		
		tau_minus = tau[tau.charge < 0]
		tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau_plus,2))
		tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau_minus,2))

		#Check assumption about tau pairings
		#print("Positively charged tau")
		#for (tau1,tau2) in zip(ak.ravel(tau_plus1.pt),ak.ravel(tau_plus2.pt)):
		#	if (tau1 < tau2):
		#		print("!!!Assumption about pT ordering wrong!!!")
		#		print("tau1.pt = %.4f"%tau1)
		#		print("tau2.pt = %.4f"%tau2)
		
		#print("Negetively charged tau")
		#for (tau1,tau2) in zip(ak.ravel(tau_minus1.pt),ak.ravel(tau_minus2.pt)):
		#	if (tau1 < tau2):
		#		print("!!!Assumption about pT ordering wrong!!!")
		#		print("tau1.pt = %.4f"%tau1)
		#		print("tau2.pt = %.4f"%tau2)

		#Get leading, subleading and fourth leading taus
		tau = tau[ak.num(tau) > 0] #Handle trigger
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
		#sublead_cond1 = tau_plus1.pT > tau_minus1.pT and deltaR11 < deltaR21
		
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
	
		print("================================================")
		print(len(tau_plus1))
		print(len(PxLeading))
		print(len(PxSubLeading))

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
		#print(Higgs_Leading.Px)
		#print(Higgs_Leading.Pz)
		#print("Eta and phi:")
		#print(Higgs_Leading.eta)	
		#print(Higgs_Leading.phi)
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
			diHiggs_dR_Arr = ak.ravel(deltaR(Higgs_Leading,Higgs_SubLeading))
			LeadingHiggs_mass_Arr = ak.ravel(single_mass(Higgs_Leading))	
			SubLeadingHiggs_mass_Arr = ak.ravel(single_mass(Higgs_SubLeading))
		else:
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

		#Get the di-tau masses
		#lead_cond1 = np.bitwise_and(tau_plus1.pt > tau_minus1.pt, deltaR11 < deltaR12)
		#lead_cond2 = np.bitwise_and(tau_plus1.pt > tau_minus1.pt, deltaR12 < deltaR11)
		#lead_cond3 = np.bitwise_and(tau_plus1.pt < tau_minus1.pt, deltaR22 < deltaR21)
		#lead_cond4 = np.bitwise_and(tau_plus1.pt < tau_minus1.pt, deltaR21 < deltaR22)
		
		#DiTauMass_all_hist.fill("Leading pair", ak.ravel(di_mass(tau_plus1[lead_cond1], tau_minus1[lead_cond1])))
		#DiTauMass_all_hist.fill("Leading pair", ak.ravel(di_mass(tau_plus2[lead_cond2], tau_minus1[lead_cond2])))
		#DiTauMass_lead_Arr = ak.ravel()
		#DiTauMass_lead_Arr = ak.ravel()
		#dimass_leading_acc = hist.accumulators.Mean().fill(ak.concatenate([ak.ravel(di_mass(tau_plus1[lead_cond1], tau_minus1[lead_cond1])), ak.ravel(di_mass(tau_plus2[lead_cond2], tau_minus1[lead_cond2]))]))
		#DiTauMass_all_hist.fill("Subleading pair", ak.ravel(di_mass(tau_plus1[lead_cond4], tau_minus2[lead_cond4])))
		#DiTauMass_all_hist.fill("Subleading pair", ak.ravel(di_mass(tau_plus2[lead_cond3], tau_minus2[lead_cond3])))
		#DiTauMass_sublead_Arr = ak.ravel()
		#DiTauMass_sublead_Arr = ak.ravel()
		#TauDeltaPhi_all_hist *= 1/(TauDeltaPhi_all_hist.sum()) #Causes divide by 0 errors
		
		#Fill Higgs Delta Phi
		#Higgs_DeltaPhi.fill(np.abs(np.arctan(PyLeading/PxLeading) - np.arctan(PySubLeading/PxSubLeading)))
		#Higgs_DeltaPhi_Arr = ak.ravel(np.abs(np.arctan(PyLeading/PxLeading) - np.arctan(PySubLeading/PxSubLeading)))
		phi_leading = np.arctan2(PyLeading,PxLeading)
		phi_subleading = np.arctan2(PySubLeading,PxSubLeading)
		Higgs_DeltaPhi_Arr = ak.ravel((phi_leading - phi_subleading + np.pi) % (2 * np.pi) - np.pi)
		radionPT_HiggsReco = np.sqrt((PxLeading + PxSubLeading)**2 + (PyLeading + PySubLeading)**2)
		radionPT_Arr = ak.ravel(radionPT_HiggsReco)
		#Higgs_dR_Arr = ak.ravel()
		
		
		#RadionPTComp.fill(ak.ravel(radionPT),ak.ravel(radionPT_HiggsReco)) r"$\tau Reconstructed"
		RadionPTComp.fill(r"$\tau Reconstructed",ak.ravel(radionPT))		
		RadionPTComp.fill("Higgs Reconstructed",ak.ravel(radionPT_HiggsReco))		
	
		#Print Statements
		#print(PxLeading)
		#if (len(PxLeading) >= 10):
		#	for i in range(10):
		#		print("")
		#		print("Leading Px[%d] = %.4f"%(i,ak.ravel(PxLeading)[i]))
		#		print("Leading Py[%d] = %.4f"%(i,ak.ravel(PyLeading)[i]))
		#		print("Leading Higgs p_T = %.4f"%np.sqrt(ak.ravel(PxLeading)[i]**2 + ak.ravel(PyLeading)[i]**2))
		#		print("Subleading Px[%d] = %.4f"%(i,ak.ravel(PxSubLeading)[i]))
		#		print("Subleading Py[%d] = %.4f"%(i,ak.ravel(PySubLeading)[i]))
		#		print("Subleading Higgs p_T = %.4f"%np.sqrt(ak.ravel(PxSubLeading)[i]**2 + ak.ravel(PySubLeading)[i]**2))
		#		print("Radion pT = %.4f"%radionPT[i])
		#		print("Higgs reconstructed Radion pT = %.4f"%np.sqrt((ak.ravel(PxLeading)[i] + ak.ravel(PxSubLeading)[i])**2 + (ak.ravel(PyLeading)[i] + ak.ravel(PySubLeading)[i])**2))
		#		print("Higgs delta phi = %.4f"%Higgs_DeltaPhi_Arr[i])
		#		print("Higgs 1 phi = %.4f"%phi_leading[i])
		#		print("Higgs 2 phi = %.4f"%phi_subleading[i])
		#else:
		#	print("Empty momenta")
		

		#Fill 4tau Mass Histogram
		#print(len(tau))
		#FourTau_Mass_hist.fill(ak.ravel(four_mass(tau)))
		FourTau_Mass_hist.fill(ak.ravel(tau.FourMass))
		FourTau_Mass_Arr = ak.ravel(tau.FourMass)
		#tau["FourTau_Mass"] = 
		#FourTau_Mass_hist.fill(four_mass(tau))
		#if (FourTau_Mass_hist.sum() > 0):
			#FourTau_Mass_hist *= (1/FourTau_Mass_hist.sum()) #Normalize the histogram
		FourTau_Mass_Acc = hist.accumulators.Mean().fill(ak.ravel(tau.FourMass))
		weight_Val = weight_calc(dataset,tau,numEvents_Dict[dataset]) 
		
		#Efficiency Histograms
		if (self.trigger_bit == 40):	
			AK8Pt_Trigg.fill(ak.ravel(AK8Jet.AK8JetPt))
			AK8SoftMass_Trigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
			AK8Pt_all.fill("Trigger",ak.ravel(AK8Jet.AK8JetPt))	
			AK8Pt_Trigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
			AK8SoftMass_all.fill("Trigger",ak.ravel(AK8Jet.AK8JetDropMass))	
			AK8SoftMass_Trigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
			pre_triggernum = ak.num(AK8Pt_NoTrigg_Arr,axis=0)
			post_triggernum = ak.num(AK8Pt_Trigg_Arr,axis=0)
            
			AK8Jet_PreTrigger.fill(AK8Pt_NoTrigg_Arr, AK8SoftMass_NoTrigg_Arr)
			AK8Jet_Trigger.fill(AK8Pt_Trigg_Arr, AK8SoftMass_Trigg_Arr)
			eff_AK8Jet = AK8Jet_Trigger/AK8Jet_PreTrigger
		
		if (self.trigger_bit == 39):
			HT_Val_PreTrigger = ak.sum(Jet_HT.Pt, axis = 1, keepdims=True)
			HT_Trigg.fill(ak.ravel(HT_Val_PreTrigger[HT_Val_PreTrigger > 0]))
			#HT_Trigg_Arr = ak.ravel(HT_Val_PreTrigger[HT_Val_PreTrigger > 0])
			HT_Trigg_Arr = ak.ravel(Jet.HT)
			MET_Trigg.fill(ak.ravel(Jet.pfMET))
			MET_Trigg_Arr = ak.ravel(Jet.pfMET)
			pre_triggernum = ak.num(MET_NoTrigg_Arr,axis=0)
			post_triggernum = ak.num(MET_Trigg_Arr,axis=0)	
			
			#print("Efficiency (HT+MET Trigger): %f"%(ak.num(MET_Trigg_Arr,axis=0)/ak.num(MET_NoTrigg_Arr,axis=0)))
		
		return{
			dataset: {
				"Weight": weight_Val,
				"FourTau_Mass_Arr": FourTau_Mass_Arr, 
				"FourTau_Mass_hist": FourTau_Mass_hist,
				"FourTau_Mass_Acc": FourTau_Mass_Acc,
				"HiggsDeltaPhi_Arr": Higgs_DeltaPhi_Arr,
				"Pair_DeltaPhi_Hist": TauDeltaPhi_all_hist,
				"RadionPTComp_Hist": RadionPTComp,
				"DiTauMass_Hist": DiTauMass_all_hist,
				"LeadingHiggs_mass": LeadingHiggs_mass_Arr,
				"SubLeadingHiggs_mass": SubLeadingHiggs_mass_Arr,
				"Higgs_DeltaR_Arr": diHiggs_dR_Arr,
				"leading_dR_Arr": leading_dR_Arr,
				"subleading_dR_Arr": subleading_dR_Arr,
				"radionPT_Arr" : radionPT_Arr
			}
		}
	
	def postprocess(self, accumulator):
		pass	

if __name__ == "__main__":
	#mass_str_arr = ["1000","2000","3000"]
	mass_str_arr = ["1000"]

	#trigger_dict = {"Mu50": (21,False), "PFMET120_PFMHT120_IDTight": (27,False), "EitherOr_Trigger": (41,True)}
	#trigger_dict = {"Mu50": (21,False), "PFMET120_PFMHT120_IDTight": (27,False), "EitherOr_Trigger": (41,True)}
	trigger_dict = {"EitherOr_Trigger": (41,True)}
	#trigger_dict = {"No_Trigger": (42,False)}
	#trigger_dict = {"PFHT500_PFMET100_PFMHT100_IDTight": (39,False), "AK8PFJet400_TrimMass30": (40,False), "EitherOr_Trigger": (41,True)}

	signal_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v1_Hadd/GluGluToRadionToHHTo4T_M-"
	background_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"	


	iterative_runner = processor.Runner(
		executor = processor.IterativeExecutor(compression=None),
		schema=BaseSchema
	)
	four_tau_hist_list = ["FourTau_Mass_Arr","HiggsDeltaPhi_Arr","Pair_DeltaPhi_Hist","RadionPTComp_Hist", "Higgs_DeltaR_Arr","leading_dR_Arr","subleading_dR_Arr","LeadingHiggs_mass","SubLeadingHiggs_mass", "radionPT_Arr"]
	#four_tau_hist_list = ["HiggsDeltaPhi_Arr","Pair_DeltaPhi_Hist"]



	for mass in mass_str_arr:
		print("====================Radion Mass = " + mass[0] + "." + mass[1] + " TeV====================")
		#Grand Unified Background + Signal Dictionary
		file_dict = {
			"TTToSemiLeptonic": [background_base + "TTToSemiLeptonic.root"], "TTTo2L2Nu": [background_base + "TTTo2L2Nu.root"], "TTToHadronic": [background_base + "TTToHadronic.root"],
			"ZZ4l": [background_base + "ZZ4l.root"], #"ZZTo4L_powheg": [background_base + "ZZTo4L_powheg"] , "WZ3l1nu" : [background_base + "WZ3l1nu.root"],  
			"VV2l2nu" : [background_base + "VV2l2nu.root"], "WZ1l3nu" : [background_base + "WZ1l3nu.root"], #"ZZTo2L2Nu_powheg": [backgroung_base + "ZZTo2L2Nu_powheg.root"], "" : [background_base + ".root"], : [background_base + ".root"]
			"WZ3l1nu" : [background_base + "WZ3l1nu.root"], "ZZ2l2q" : [background_base + "ZZ2l2q.root"], "WZ2l2q" : [background_base + "WZ2l2q.root"], "WZ1l1nu2q" : [background_base + "WZ1l1nu2q.root"],
			"DYJetsToLL_Pt-50To100": [background_base + "DYJetsToLL_Pt-50To100.root"] ,"DYJetsToLL_Pt-100To250": [ background_base + "DYJetsToLL_Pt-100To250.root"], 
			"DYJetsToLL_Pt-250To400": [ background_base + "DYJetsToLL_Pt-250To400.root"], "DYJetsToLL_Pt-400To650": [ background_base + "DYJetsToLL_Pt-400To650.root"], 
			"DYJetsToLL_Pt-650ToInf": [background_base + "DYJetsToLL_Pt-650ToInf.root"],
			"Tbar-tchan" : [background_base + "Tbar-tchan.root"], "T-tchan" : [background_base + "T-tchan.root"], "Tbar-tW" : [background_base + "Tbar-tW.root"], 
			"T-tW" : [background_base + "T-tW.root"],
			"WJetsToLNu_HT-100To200" : [background_base + "WJetsToLNu_HT-100To200.root"],"WJetsToLNu_HT-200To400" : [background_base + "WJetsToLNu_HT-200To400.root"], 
			"WJetsToLNu_HT-400To600" : [background_base + "WJetsToLNu_HT-400To600.root"], "WJetsToLNu_HT-600To800" : [background_base + "WJetsToLNu_HT-600To800.root"],
			"WJetsToLNu_HT-800To1200" : [background_base + "WJetsToLNu_HT-800To1200.root"],"WJetsToLNu_HT-1200To2500" : [background_base + "WJetsToLNu_HT-1200To2500.root"],"WJetsToLNu_HT-2500ToInf" : [background_base + "WJetsToLNu_HT-2500ToInf.root"],
			"Signal": [signal_base + mass + ".root"]	
		}
		#event_num_dict = {}
		#Generate dictionary of number of processed events
		for key_name, file_name in file_dict.items():
			tempFile = uproot.open(file_name[0]) #Get file
			print("Input Type:")
			print(key_name)
			numEvents_Dict[key_name] = tempFile['hEvents'].member('fEntries')/2
			print(tempFile['hEvents'].member('fEntries')/2)

		hist_dict = {
			"FourTau_Mass_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", "ZZ4l", "Signal"]).Regular(20,0,3000, label = r"$m_{4\tau}$ [GeV]").Double(),
			"HiggsDeltaPhi_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", "ZZ4l", "Signal"]).Reg(20,-pi,pi, label = r"Higgs $\Delta \phi$").Double(), 
			"Higgs_DeltaR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", "ZZ4l", "Signal"]).Reg(20,0,5, label = r"Higgs $\Delta$R").Double(),
			"leading_dR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", "ZZ4l", "Signal"]).Reg(20,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
			"subleading_dR_Arr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", "ZZ4l", "Signal"]).Reg(20,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double(),
			"LeadingHiggs_mass" : hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", "ZZ4l", "Signal"]).Reg(12,0,120, label=r"Leading Higgs Mass (GeV)").Double(),
			"SubLeadingHiggs_mass" : hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", "ZZ4l", "Signal"]).Reg(12,0,120, label=r"Sub-Leading Higgs Mass (GeV)").Double(),
			"radionPT_Arr" : hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", "ZZ4l", "Signal"]).Reg(20,0,500, label=r"Radion $p_T$ (GeV)").Double()
		}
		

		background_list = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"]
		background_plot_names = {r"$t\bar{t}$" : "_ttbar_", r"Drell-Yan+Jets": "_DYJets_", "Di-Bosons" : "_DiBosons_", "Single Top": "_SingleTop+", "QCD" : "_QCD_", "W+Jets" : "_WJets_", r"$ZZ \rightarrow 4l$" : "_ZZ4l_"}
		background_dict = {r"$t\bar{t}$" : ["TTToSemiLeptonic","TTTo2L2Nu","TTToHadronic"], 
					r"Drell-Yan+Jets": ["DYJetsToLL_Pt-50To100","DYJetsToLL_Pt-100To250","DYJetsToLL_Pt-250To400","DYJetsToLL_Pt-400To650","DYJetsToLL_Pt-650ToInf"], 
					"Di-Bosons": ["WZ3l1nu","WZ2l2q","WZ1l1nu2q","ZZ2l2q","WZ3l1nu", "WZ1l3nu", "VV2l2nu"], "Single Top": ["Tbar-tchan","T-tchan","Tbar-tW","T-tW"], 
					#"QCD": ["QCD_HT300to500","QCD_HT500to700","QCD_HT700to1000","QCD_HT1000to1500","QCD_HT1500to2000","QCD_HT2000toInf"], 
					"W+Jets": ["WJetsToLNu_HT-100To200","WJetsToLNu_HT-200To400","WJetsToLNu_HT-400To600","WJetsToLNu_HT-600To800","WJetsToLNu_HT-800To1200","WJetsToLNu_HT-1200To2500","WJetsToLNu_HT-2500ToInf"],
					r"$ZZ \rightarrow 4l$" : ["ZZ4l"],
				}
		#file_dict = {"Background" : [background_base + "ZZ4l.root"], "Signal": [signal_base + mass + ".root"]}
		for trigger_name, trigger_pair in trigger_dict.items():
			four_tau_names = {"FourTau_Mass_Arr": "FourTauMass_" + mass + "-" + trigger_name, "HiggsDeltaPhi_Arr": "HiggsDeltaPhi_" + mass + "-" + trigger_name, 
				"Pair_DeltaPhi_Hist": "TauPair_DeltaPhi_" + mass + "-" + trigger_name, "RadionPTComp_Hist": "pTReco_Comp_"+mass+ "-"+ trigger_name,
				"Higgs_DeltaR_Arr": "Higgs_DeltaR_" + mass + "-" + trigger_name, "leading_dR_Arr": "leading_diTau_DeltaR_" + mass + "-" + trigger_name, 
				"subleading_dR_Arr": "subleading_diTau_DeltaR_" + mass + "-" + trigger_name, "LeadingHiggs_mass" : "LeadingHiggs_Mass_"+ mass + "-" + trigger_name, 
				"SubLeadingHiggs_mass" : "SubLeadingHiggs_Mass_" + mass + "-" + trigger_name, "radionPT_Arr": "radion_pT_Mass_" + mass + "-" + trigger_name
			}
			
			fourtau_out = iterative_runner(file_dict, treename="4tau_tree", processor_instance=FourTauPlotting(trigger_bit=trigger_pair[0], or_trigger=trigger_pair[1], is_data =False))
			for hist_name in four_tau_hist_list: #Loop over all histograms
				fig,ax = plt.subplots()
				fig0,ax0 = plt.subplots()
				if (hist_name != "Pair_DeltaPhi_Hist" and hist_name != "RadionPTComp_Hist"):
					hist_dict_signal = {
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
					hist_dict_signal[hist_name].fill(fourtau_out["Signal"][hist_name],weight=fourtau_out["Signal"]["Weight"])
					hist_dict_signal[hist_name].plot1d(ax=ax0)
					plt.title("Signal Mass: " + mass[0] + "TeV")
					plt.savefig("SignalSingle" + four_tau_names[hist_name])
					plt.close()

				if (hist_name != "Pair_DeltaPhi_Hist" and hist_name != "RadionPTComp_Hist"):
					for background_type in background_list:
						hist_dict_background = {
							"FourTau_Mass_Arr": hist.Hist.new.Regular(20,0,3000, label = r"$m_{4\tau}$ [GeV]").Double(),
							"HiggsDeltaPhi_Arr": hist.Hist.new.Regular(20,-pi,pi, label = r"Higgs $\Delta \phi$").Double(), 
							"Higgs_DeltaR_Arr": hist.Hist.new.Regular(20,0,5, label = r"Higgs $\Delta$R").Double(),
							"leading_dR_Arr": hist.Hist.new.Regular(20,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
							"subleading_dR_Arr": hist.Hist.new.Regular(20,0,5, label = r"Sub-leading di-$\tau$ $\Delta$R").Double(),
							"LeadingHiggs_mass" : hist.Hist.new.Regular(20,0,120, label=r"Leading Higgs Mass (GeV)").Double(),
							"SubLeadingHiggs_mass" : hist.Hist.new.Regular(20,0,120, label=r"Sub-Leading Higgs Mass (GeV)").Double(),
							"radionPT_Arr" : hist.Hist.new.Regular(20,0,500, label=r"Radion $p_T$ (GeV)").Double()
						}
						background_array = np.array([])
						backgrounds = background_dict[background_type]
						#tempArray = np.array([])
						#for background in backgrounds:
						#Loop over all backgrounds
						for background in backgrounds:
							print("Current background = %s"%background)
							print("Number of entries = %d"%len(fourtau_out[background][hist_name]))
							print("Cross Section = %.3f"%xSection_Dictionary[background])
							print("Background Weight = %.3f"%fourtau_out[background]["Weight"])
							if (mass == "1000"): #Only need to generate background once
								fig2, ax2 = plt.subplots()
								#if (len(fourtau_out[background][hist_name]) != 0):
								hist_dict_background[hist_name].fill(fourtau_out[background][hist_name],weight=fourtau_out[background]["Weight"]) 								
								#else:
								#	hist_dict_background[hist_name].fill(fourtau_out[background][hist_name],weight=fourtau_out[background]["Weight"]) 								
								hist_dict_background[hist_name].plot1d(ax=ax2)
								plt.title(background_type)
								plt.savefig("SingleBackground" + background_plot_names[background_type] + four_tau_names[hist_name])
								plt.close()
							#hist_dict_background[hist_name]
							if (len(fourtau_out[background][hist_name]) != 0):
								#hist_dict[hist_name].fill(background_type,fourtau_out[background][hist_name],weight=fourtau_out[background]["Weight"]*1/len(fourtau_out[background][hist_name]))
								hist_dict[hist_name].fill(background_type,fourtau_out[background][hist_name],weight=fourtau_out[background]["Weight"])
							else:
								hist_dict[hist_name].fill(background_type,fourtau_out[background][hist_name],weight=fourtau_out[background]["Weight"])
					
					background_array = np.append(background_array,fourtau_out[background][hist_name])
					#hist_dict[hist_name].fill("Signal",fourtau_out["Signal"][hist_name],weight=fourtau_out["Signal"]["Weight"]*1/len(fourtau_out["Signal"][hist_name]))
					hist_dict[hist_name].fill("Signal",fourtau_out["Signal"][hist_name],weight=fourtau_out["Signal"]["Weight"])
					print("# of Signal events: %d"%len(fourtau_out["Signal"][hist_name]))
					
					#hist_dict[hist_name]["Signal"] *= (1/hist_dict[hist_name]["Signal"].sum())
						#background_acc = hist.accumulators.Mean().fill(background_array)
					signal_acc = hist.accumulators.Mean().fill(fourtau_out["Signal"][hist_name])
					#hist_dict[hist_name] *= (1/hist_dict[hist_name].sum()) #Normalize histogram??
					hist_dict[hist_name].plot1d(ax=ax,histtype="fill") #Draw Histogram
					#if (hist_name == "FourTau_Mass_Arr"):
					#	plt.text(x = 0.6,y = 0.7,s = "Background mean: %.2f GeV"%background_acc.value, transform = ax.transAxes, fontsize="small")
					#	plt.text(x = 0.6,y = 0.6,s = "Signal mean: %.2f GeV"%signal_acc.value, transform = ax.transAxes, fontsize="small")
					#	plt.title(r"4$\tau$ Mass " + mass[0] + "." + mass[1] + " TeV Signal")
					#if (hist_name == "HiggsDeltaPhi_Arr"):
					#	plt.text(x = 0.6,y = 0.7,s = "Background mean: %.2f"%background_acc.value, transform = ax.transAxes, fontsize="small")
					#	plt.text(x = 0.6,y = 0.6,s = "Signal mean: %.2f"%signal_acc.value, transform = ax.transAxes, fontsize="small")
					#	plt.title(r"Higgs $\Delta \phi$ " + mass[0] + "." + mass[1] + " TeV Signal")

					ax.legend(title="Legend")
					plt.savefig(four_tau_names[hist_name])
					plt.close()
				if (hist_name == "Pair_DeltaPhi_Hist"):
					print(hist_name)
					fourtau_out["Signal"][hist_name].plot1d(ax=ax,histtype="fill")
					ax.legend(title=r"Di-$\tau$ Pair")
					plt.savefig(four_tau_names[hist_name])
					plt.close()
				if (hist_name == "RadionPTComp_Hist"):
					fourtau_out["Signal"][hist_name].plot1d(ax=ax,histtype="fill")
					ax.legend(title=r"Radion p_T method")
					plt.savefig(four_tau_names[hist_name])
					plt.close()
						
			
			#fourtau_out["Background"]["FourTau_Mass_hist"].plot1d(ax=ax, label = "Background") #Draw Histogram


