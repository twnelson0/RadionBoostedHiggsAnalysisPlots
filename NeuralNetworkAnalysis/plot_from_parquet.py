import pandas as pd
import numpy as np
from math import pi
import awkward as ak
import mplhep as hep
import matplotlib.pyplot as plt
import hist

hep.style.use(hep.style.CMS)
TABLEAU_COLORS = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

def StackPlot(signal_array,signal_list,background_array,background_list,data_array,plot_name):
    fig,ax = plt.subplots()
    hep.histplot(background_array,ax=ax,stack=True,histtype="fill",label=background_list,facecolor=TABLEAU_COLORS[:len(background_list)],edgecolor=TABLEAU_COLORS[:len(background_list)])
    hep.histplot(signal_array,ax=ax,stack=True,histtype="step",label=signal_list,edgecolor=TABLEAU_COLORS[len(background_list)+1],linewidth=2.95)
    hep.histplot(data_array,ax=ax,stack=False,histtype="errorbar", yerr=True,label=["Data"],marker="o",color = "k") 
    hep.cms.text("Preliminary",loc=0,fontsize=13)
    ax.set_title("2018 Data",loc = "right")
    ax.legend(fontsize=10, loc='upper right')
    plt.savefig(plot_name)
    plt.close()


if __name__ == "__main__":
    print("Test")

    mass = "2000"
    mass_array = ["1000","2000","3000"]

    #Neural Network Varaibles
    var_array = ["radion_pt","vis_mass","vis_mass2","radion_eta","higgs1_dr","higgs2_dr","dphi_H1","dphi_H2",
            "dphi_H1_MET","dphi_H2_MET","dr_HH","dphi_HH","dr_H1_Rad","dr_H2_Rad","dphi_rad_MET","H1OS","H2OS","numBJet","predictions"] #,"weight","predictions"]
    
    #Variables for plotting
    dict_plot_names = {"radion_pt": "radion_pT_Mass_" + mass + "-EitherOr_Trigger","vis_mass": "LeadingHiggs_Mass_"+ mass + "-EitherOr_Trigger",
            "vis_mass2": "LeadingHiggs_Mass_"+ mass + "-EitherOr_Trigger", "radion_eta": "Radion_eta_Mass_" + mass + "-EitherOr_Trigger", 
            "higgs1_dr": "leading_diTau_DeltaR_" + mass + "-EitherOr_Trigger", "higgs2_dr": "subleading_diTau_DeltaR_" + mass + "-EitherOr_Trigger",
            "dphi_H1": "LeadingdiTau_DeltaPhi_Mass_" + mass + "-EitherOr_Trigger", "dphi_H2": "SubleadingdiTau_DeltaPhi_Mass_" + mass + "-EitherOr_Trigger",
            "dphi_H1_MET": "LeadingHiggs_MET_DeltaPhi_Mass_" + mass + "-EitherOr_Trigger", "dphi_H2_MET": "SubleadingHiggs_MET_DeltaPhi_Mass_" + mass + "-EitherOr_Trigger",
            "dr_HH": "Higgs_DeltaR_" + mass + "-EitherOr_Trigger", "dphi_HH": "HiggsDeltaPhi_" + mass + "-EitherOr_Trigger" ,"dr_H1_Rad": "LeadingHiggs_Radion_DeltaR_Mass_" + mass + "-EitherOr_Trigger",
            "dr_H2_Rad": "SubLeadingHiggs_Radion_DeltaR_Mass_" + mass + "-EitherOr_Trigger", "dphi_rad_MET": "Radion_MET_DeltaPhi_Mass_" + mass + "-EitherOr_Trigger",
            "H1OS":"LeadingHiggs_Charge_Mass" + mass + "-EitherOr_Trigger","H2OS": "SubleadingHiggs_Charge_Mass" + mass + "-EitherOr_Trigger", "numBJet": "BJetMult_Mass_" + mass + "-EitherOr_Trigger", 
            "predictions": "NeuralNetwork_Prediction_Mass" + mass + "-EitherOr_Trigger"}
    data_files = ["Data_JetHT","Data_SingleMuon"]
    signal_file = ["Signal"]
    background_list = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"]
    signal_list = [r"MC Sample $m_\phi$ = %s TeV"%mass[0]]
    background_plot_names = {r"$t\bar{t}$" : "_ttbar_", r"Drell-Yan+Jets": "_DYJets_", "Di-Bosons" : "_DiBosons_", "Single Top": "_SingleTop+", "QCD" : "_QCD_", "W+Jets" : "_WJets_", r"$ZZ \rightarrow 4l$" : "_ZZ4l_"}
    background_dict = {r"$t\bar{t}$" : ["TTToSemiLeptonic","TTTo2L2Nu","TTToHadronic"], 
				r"Drell-Yan+Jets": ["DYJetsToLL_Pt-50To100","DYJetsToLL_Pt-100To250","DYJetsToLL_Pt-250To400","DYJetsToLL_Pt-400To650","DYJetsToLL_Pt-650ToInf"], 
				"Di-Bosons": ["WZ3l1nu","WZ2l2q","WZ1l1nu2q","ZZ2l2q", "WZ1l3nu", "VV2l2nu"], "Single Top": ["Tbar-tchan","T-tchan","Tbar-tW","T-tW"], 
				"W+Jets": ["WJetsToLNu_HT-100To200","WJetsToLNu_HT-200To400","WJetsToLNu_HT-400To600","WJetsToLNu_HT-600To800","WJetsToLNu_HT-800To1200","WJetsToLNu_HT-1200To2500","WJetsToLNu_HT-2500ToInf"],
				r"$ZZ \rightarrow 4l$" : ["ZZ4l"]
            }
    
    #Import data, background and signal
    background_input_dict = {}
    for background_type in background_dict:
        for background in background_dict[background_type]:
            background_input_dict[background] = pd.read_parquet(background + "with_predictions.parquet",engine="pyarrow")

    signal_input = pd.read_parquet("Signalwith_predictions.parquet",engine="pyarrow")

    data_input_dict = {}
    for data_file in data_files:
        data_input_dict[data_file] = pd.read_parquet(data_file + "with_predictions.parquet",engine="pyarrow")

    
    #Histogram binning
    N1 = 10
    N2 = 8

    #Histogram dictionaries
    hist_dict_background = {
            "radion_pt": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,500, label=r"Radion $p_T$ (GeV)").Double(), 
            "vis_mass": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N2,0,120, label=r"Leading di-$\tau$ Mass (GeV)").Double(),
            "vis_mass2": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N2,0,120, label=r"Subleading di-$\tau$ Mass (GeV)").Double(),
            "radion_eta": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-5,5, label = r"Radion $\eta$").Double(),
            "higgs1_dr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
            "higgs2_dr": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,5, label = r"Subleading di-$\tau$ $\Delta$R").Double(),
            "dphi_H1": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(),
            "dphi_H2": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(),
            "dphi_H1_MET": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Leading Higgs MET $\Delta \phi$").Double(),
            "dphi_H2_MET": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta \phi$").Double() ,
            "dr_HH": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,5, label = r"Higgs $\Delta$R").Double(),
            "dphi_HH": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Higgs $\Delta \phi$").Double(),
            "dr_H1_Rad": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
            "dr_H2_Rad": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
            "dphi_rad_MET": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(),
            "H1OS": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name = "background").Reg(8,-4,4,label = r"Leading Higgs Electric Charge").Double(),
            "H2OS": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name = "background").Reg(8,-4,4,label = r"Subleading Higgs Electric Charge").Double(),
            "numBJet": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(6,0,6, label = r"B Jet Multiplicity").Double(),
            "predictions": hist.Hist.new.StrCat([r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"],name="background").Reg(N1,0,1, label = r"Neural Network Model Output").Double() 
        }

    hist_dict_signal = {
            "radion_pt": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,500, label=r"Radion $p_T$ (GeV)").Double(), 
            "vis_mass": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N2,0,120, label=r"Leading di-$\tau$ Mass (GeV)").Double(),
            "vis_mass2": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N2,0,120, label=r"Subleading di-$\tau$ Mass (GeV)").Double(),
            "radion_eta": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-5,5, label = r"Radion $\eta$").Double(),
            "higgs1_dr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
            "higgs2_dr": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,5, label = r"Subleading di-$\tau$ $\Delta$R").Double(),
            "dphi_H1": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(),
            "dphi_H2": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(),
            "dphi_H1_MET": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Leading Higgs MET $\Delta \phi$").Double(),
            "dphi_H2_MET": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta \phi$").Double() ,
            "dr_HH": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,5, label = r"Higgs $\Delta$R").Double(),
            "dphi_HH": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Higgs $\Delta \phi$").Double(),
            "dr_H1_Rad": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
            "dr_H2_Rad": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
            "dphi_rad_MET": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(),
            "H1OS": hist.Hist.new.StrCat(["Signal"],name = "signal").Reg(8,-4,4,label = r"Leading Higgs Electric Charge").Double(),
            "H2OS": hist.Hist.new.StrCat(["Signal"],name = "signal").Reg(8,-4,4,label = r"Subleading Higgs Electric Charge").Double(),
            "numBJet": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(6,0,6, label = r"B Jet Multiplicity").Double(),
            "predictions": hist.Hist.new.StrCat(["Signal"],name="signal").Reg(N1,0,1, label = r"Neural Network Model Output").Double() 
        }

    hist_dict_data = {
            "radion_pt": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,500, label=r"Radion $p_T$ (GeV)").Double(), 
            "vis_mass": hist.Hist.new.StrCat(["Data"],name="data").Reg(N2,0,120, label=r"Leading di-$\tau$ Mass (GeV)").Double(),
            "vis_mass2": hist.Hist.new.StrCat(["Data"],name="data").Reg(N2,0,120, label=r"Subleading di-$\tau$ Mass (GeV)").Double(),
            "radion_eta": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-5,5, label = r"Radion $\eta$").Double(),
            "higgs1_dr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,5, label = r"Leading di-$\tau$ $\Delta$R").Double(),
            "higgs2_dr": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,5, label = r"Subleading di-$\tau$ $\Delta$R").Double(),
            "dphi_H1": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Subleading di-$\tau$ $\Delta \phi$").Double(),
            "dphi_H2": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Leading di-$\tau$ $\Delta \phi$").Double(),
            "dphi_H1_MET": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Leading Higgs MET $\Delta \phi$").Double(),
            "dphi_H2_MET": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Subleading Higgs MET $\Delta \phi$").Double() ,
            "dr_HH": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,5, label = r"Higgs $\Delta$R").Double(),
            "dphi_HH": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Higgs $\Delta \phi$").Double(),
            "dr_H1_Rad": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,5, label = r"Leading Higgs Radion $\Delta$R").Double(),
            "dr_H2_Rad": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,5, label = r"Subleading Higgs Radion $\Delta$R").Double(),
            "dphi_rad_MET": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,-pi,pi, label = r"Radion MET $\Delta \phi$").Double(),
            "H1OS": hist.Hist.new.StrCat(["Data"],name = "data").Reg(8,-4,4,label = r"Leading Higgs Electric Charge").Double(),
            "H2OS": hist.Hist.new.StrCat(["Data"],name = "data").Reg(8,-4,4,label = r"Subleading Higgs Electric Charge").Double(),
            "numBJet": hist.Hist.new.StrCat(["Data"],name="data").Reg(6,0,6, label = r"B Jet Multiplicity").Double(),
            "predictions": hist.Hist.new.StrCat(["Data"],name="data").Reg(N1,0,1, label = r"Neural Network Model Output").Double() 
        }

    #Generate all histograms
    for hist_name in var_array:
        #Fill background histograms
        for background_type in background_list:
            for background in background_dict[background_type]:
                hist_dict_background[hist_name].fill(background_type,background_input_dict[background][hist_name].to_numpy(),weight=background_input_dict[background]["weight"].to_numpy())
                print("Hist name: " + hist_name)
                print("Weight for " + background +":")
                print(background_input_dict[background]["weight"].to_numpy())
                if (background == "ZZ4l" and hist_name == "radion_pt"):
                    print("Z4l radion pT being read in:")
                    print(background_input_dict[background][hist_name])
        
        #Fill in signal
        hist_dict_signal[hist_name].fill("Signal",signal_input[hist_name].to_numpy(),weight=signal_input["weight"].to_numpy())

        #Fill in the data
        for data_file in data_files:
            hist_dict_data[hist_name].fill("Data",data_input_dict[data_file][hist_name].to_numpy())

        #Put dictionaries into stacks and arrays for plotting
        background_array = []
        background_stack = hist_dict_background[hist_name].stack("background")
        signal_stack = hist_dict_signal[hist_name].stack("signal")
        data_stack = hist_dict_data[hist_name].stack("data")

        signal_array = [signal_stack["Signal"]]
        data_array = [data_stack["Data"]]
        for background in background_list:
            background_array.append(background_stack[background])

        #Generate plots
        StackPlot(signal_array,signal_list,background_array,background_list,data_array,dict_plot_names[hist_name]) 
        
         

