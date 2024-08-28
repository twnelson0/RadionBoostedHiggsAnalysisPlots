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
import vector
import os

class TauLeptonPlotting(processor.ProcessorABC):
    def __init__(self):
        #self.dummy_var = True
        pass
    
    def process(self, events):
        dataset = events.metadata['dataset']
        tau = ak.zip(
            {
                "Pt": events.boostedTauPt,
                "E": events.boostedTauEnergy,
                "Px": events.boostedTauPx,
                "Py": events.boostedTauPy,
                "Pz": events.boostedTauPz,
                "eta": events.boostedTauEta,
                "phi": events.boostedTauPhi,

            },
            with_name="TauArray",
            behavior=candidate.behavior,
        )
        electron = ak.zip(
            {
                "Pt": events.elePt,
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
                "Pt": events.muPt,
                "E": events.muEn,
                "eta": events.muEta,
                "phi": events.muPhi,
                "charge": events.muCharge,
                "Px": events.muPt*np.cos(events.muPhi),
                "Py": events.muPt*np.sin(events.muPhi),
                "Pz": events.muPt*np.tan(2*np.arctan(np.exp(-events.muEta)))**-1,
            },
            with_name="MuonArray",
            behavior=candidate.behavior,
        )
        Gen_Level = ak.zip(
            {
                "MCId": events.mcPID,
                "Pt": events.mcPt,
                "E": events.mcE,
                "eta": events.mcEta,
                "phi": events.mcPhi,
                "Px": events.mcPt*np.cos(events.mcPhi),
                "Py": events.mcPt*np.sin(events.mcPhi),
                "Pz": events.mcPt*np.tan(2*np.arctan(np.exp(-events.mcEta)))**-1,
            },
            with_name="Gen_Array",
            behavior=candidate.behavior,
        )

        #Get gen level Muons, and electrons
        gen_electron = Gen_Level[np.abs(Gen_Level.MCId) == 11]
        gen_muon = Gen_Level[np.abs(Gen_Level.MCId) == 13]

        #Apply momentum selections
        muon = muon[muon.Pt > 20]
        electron = electron[electron.Pt > 20]

        print("Number of reco electrons: %d"%ak.sum(ak.num(electron,axis=1)))
        print("Number of reco muons: %d"%ak.sum(ak.num(muon,axis=1)))

        #Construct four vectors for reco electron, muon and tau
        electron_fourVec = ak.zip({"x": electron.Px, "y": electron.Py, "z": electron.Pz, "t": electron.E},with_name = "LorentzVector")
        muon_fourVec = ak.zip({"x": muon.Px, "y": muon.Py, "z": muon.Pz, "t": muon.E},with_name = "LorentzVector")
        tau_fourVec = ak.zip({"x": tau.Px, "y": tau.Py, "z": tau.Pz, "t": tau.E},with_name = "LorentzVector")
        #electron_fourVec = ak.zip({"pt": electron.Pt, "phi": electron.phi, "eta": electron.eta, "t": electron.E},with_name = "LorentzVector")
        #muon_fourVec = ak.zip({"pt": muon.Pt, "phi": muon.phi, "eta": muon.eta, "t": muon.E},with_name = "LorentzVector")
        #tau_fourVec = ak.zip({"pt": tau.Pt, "phi": tau.phi, "eta": tau.eta, "t": tau.E},with_name = "LorentzVector")

        #electron_fourVec.delta_r(tau_fourVec)
        min_tau_ele = electron_fourVec.nearest(tau_fourVec)
        min_tau_mu = muon_fourVec.nearest(tau_fourVec)

        ele_dR_collection = electron_fourVec.delta_r(min_tau_ele)
        mu_dR_collection = muon_fourVec.delta_r(min_tau_mu)
        electron["tau_min_dR"] = ele_dR_collection
        muon["tau_min_dR"] = mu_dR_collection

        #Obtain the dR distribution of reco leptons to nearest gen level leptons
        gen_ele_fourVec = ak.zip({"x": gen_electron.Px, "y": gen_electron.Py, "z": gen_electron.Pz, "t": gen_electron.E},with_name="LorentzVector")
        gen_mu_fourVec = ak.zip({"x": gen_muon.Px, "y": gen_muon.Py, "z": gen_muon.Pz, "t": gen_muon.E},with_name="LorentzVector")

        min_ele_gen = electron_fourVec.nearest(gen_ele_fourVec) 
        min_mu_gen = muon_fourVec.nearest(gen_mu_fourVec)
        ele_gen_dR = electron_fourVec.delta_r(min_ele_gen) 
        mu_gen_dR = muon_fourVec.delta_r(min_mu_gen)

        electron["gen_min_dR"] = ele_gen_dR
        muon["gen_min_dR"] = mu_gen_dR
        
        return{
            dataset: {
                "Electron_tau_dR_Arr" : ak.to_list(ak.ravel(ak.where(ak.num(electron.tau_min_dR,axis=1)!= 0, electron.tau_min_dR, ak.singletons(np.ones(ak.num(electron.tau_min_dR,axis=0))*999)))),
                "Muon_tau_dR_Arr" : ak.to_list(ak.ravel(ak.where(ak.num(muon.tau_min_dR,axis=1) != 0, muon.tau_min_dR, ak.singletons(np.ones(ak.num(muon.tau_min_dR,axis=0))*999)))),
                "Electron_gen_dR_Arr": ak.to_list(ak.ravel(ak.where(ak.num(electron.gen_min_dR,axis=1)!= 0, electron.gen_min_dR, ak.singletons(np.ones(ak.num(electron.gen_min_dR,axis=0))*999)))),
                "Muon_gen_dR_Arr": ak.to_list(ak.ravel(ak.where(ak.num(muon.gen_min_dR,axis=1)!= 0, muon.gen_min_dR, ak.singletons(np.ones(ak.num(muon.gen_min_dR,axis=0))*999)))),
             }
        }

    def postprocess(self, accumulator):
        pass

if __name__ == "__main__":
    signal_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/GluGluToRadionToHHTo4T_M-"
	
    iterative_runner = processor.Runner(executor = processor.IterativeExecutor(compression=None),schema=BaseSchema)
    mass = "2000"
    file_dict = {"Signal": [signal_base + mass + ".root"]}
    taulepton_out = iterative_runner(file_dict, treename="4tau_tree", processor_instance=TauLeptonPlotting())

    #Set up histograms
    histogram_dict = {
            "Electron_tau_dR_Arr": hist.Hist.new.Regular(10,0,1,label = r"Minimized tau to electron $\Delta$R").Double(),
            "Muon_tau_dR_Arr": hist.Hist.new.Regular(10,0,1,label = r"Minimized tau to muon $\Delta$R").Double(),
            "Electron_gen_dR_Arr": hist.Hist.new.Regular(10,0,1,label = r"Minimized gen electron to reco electron $\Delta$R").Double(),
            "Muon_gen_dR_Arr": hist.Hist.new.Regular(10,0,1,label = r"Minimized gen muon to reco muon $\Delta$R").Double(),
    }
    hist_name_dict = {
            "Electron_tau_dR_Arr": "Signal_Electron_Tau_Minimized_dR_LinScale",
            "Muon_tau_dR_Arr": "Signal_Muon_Tau_Minimized_dR_LinScale",
            "Electron_gen_dR_Arr": "Signal_Electron_Gen_Minimized_dR_LinScale",
            "Muon_gen_dR_Arr": "Signal_Muon_Gen_Minimized_dR_LinScale",
    }

    
    #Obtain reco lepton reco tau dR distributions
    for hist_name in histogram_dict:
        fig0,ax0 = plt.subplots()
        fill_Arr = ak.from_iter(taulepton_out["Signal"][hist_name])
        fill_Arr = fill_Arr[fill_Arr != 999] #Drop empty values 
        histogram_dict[hist_name].fill(fill_Arr)
        histogram_dict[hist_name].plot1d(ax = ax0)
        #ax0.set_yscale('log')
        plt.savefig(hist_name_dict[hist_name])
    
    #ele_tau_mindR = ak.from_iter(fourtau_out["Signal"]["Electron_tau_dR_Arr"])
    #ele_tau_mindR = ele_tau_mindR[ele_tau_mindR != 999]



