import uproot
import ROOT
import numpy as np
import pandas as pd
import mplhep as hep
import matplotlib.pyplot as plt
import awkward as ak
import hist
import os

if __name__ == "__main__":
    
    mass_array = ["1000","2000","3000"]
    #mass = "2000"
    
    for mass in mass_array:
        data_files = ["Data_JetHT","Data_SingleMuon"]
        file_dict = {"TTT" : ["TTToSemiLeptonic","TTTo2L2Nu","TTToHadronic","Tbar-tchan","T-tchan","Tbar-tW","T-tW"], 
                    "DY": ["DYJetsToLL_Pt-50To100","DYJetsToLL_Pt-100To250","DYJetsToLL_Pt-250To400","DYJetsToLL_Pt-400To650","DYJetsToLL_Pt-650ToInf"], 
                    "VV": ["WZ3l1nu","WZ2l2q","WZ1l1nu2q","ZZ2l2q", "WZ1l3nu", "VV2l2nu"], "ZZ" : ["ZZ4l"], "Radion" + str(mass): ["Signal_mass_" + str(mass) + "GeV"], 
                    "data_obs": ["Data_JetHT","Data_SingleMuon"],
                    "other_bkg": ["WJetsToLNu_HT-100To200","WJetsToLNu_HT-200To400","WJetsToLNu_HT-400To600","WJetsToLNu_HT-600To800","WJetsToLNu_HT-800To1200","WJetsToLNu_HT-1200To2500","WJetsToLNu_HT-2500ToInf"]
                }

        dir_array = ["TTCR_0jet", "ZCR_0jet", "highPurity_0jet", "lowPurity_0jet", "fakeCR_0jet"] #ROOT file directories
        #repeat_test_dict = {"TTCR_0jet": [], "ZCR_0jet": [], "highPurity_0jet": [], "lowPurity_0jet": [], "fakeCR_0jet": []}

        #file = uproot.create("RadionHH4t.root")
        num_signal_events = 0
        num_signal_regions = 0
        #hist_names = ["Signal" : "Radion" + str(mass), "ZZ4l": "ZZ", ]
        for curnt_dir in dir_array:
            print(curnt_dir)
            for file_type in file_dict:
                RecoRadion_Hist = hist.Hist.new.Regular(30,0,3000).Double()
                for file in file_dict[file_type]: #Loop over all files
                    input_var_struct = ak.from_parquet(file + "with_predictions.parquet")
                    
                    if (file_type == "Radion" + str(mass) and curnt_dir == "TTCR_0jet"):
                        num_signal_events += ak.num(input_var_struct["RecoRadion_Mass"],axis=0)
                    if (file_type == "Radion" + str(mass)):
                        input_var_struct["evnt_indx"] = ak.Array([i for i in range(num_signal_events)]) #Index events to ensure no event ends up in the same control region

                    #Convert variables of interest into awkward arrays for selection purposes
                    if (curnt_dir == "TTCR_0jet"):
                        #reco_rad_mass = input_var_struct[input_var_struct["numBJet"] >= 1]["RecoRadion_Mass"] 
                        reco_rad_mass = input_var_struct[np.bitwise_and(input_var_struct["numBJet"] >= 1, input_var_struct["ZMult"] < 1)]["RecoRadion_Mass"] 
                        if (file_type != "data_obs"):
                            weight_input = input_var_struct[np.bitwise_and(input_var_struct["numBJet"] >= 1, input_var_struct["ZMult"] < 1)]["weight"]
                        if (file_type == "Radion" + str(mass)):
                            s_ttcr = set(ak.to_list(input_var_struct[np.bitwise_and(input_var_struct["numBJet"] >= 1, input_var_struct["ZMult"] < 1)]["evnt_indx"]))
                            #s_ttcr = set(ak.to_list(input_var_struct[input_var_struct["numBJet"] >= 1]["evnt_indx"]))
                    if (curnt_dir == "ZCR_0jet"):
                        #reco_rad_mass = input_var_struct[input_var_struct["ZMult"] >= 1]["RecoRadion_Mass"]
                        reco_rad_mass = input_var_struct[np.bitwise_and(input_var_struct["ZMult"] >= 1, input_var_struct["numBJet"] < 1)]["RecoRadion_Mass"]
                        if (file_type != "data_obs"):
                            weight_input = input_var_struct[np.bitwise_and(input_var_struct["ZMult"] >= 1, input_var_struct["numBJet"] < 1)]["weight"]
                        if (file_type == "Radion" + str(mass)):
                            s_zcr = set(ak.to_list(input_var_struct[np.bitwise_and(input_var_struct["ZMult"] >= 1, input_var_struct["numBJet"] < 1)]["evnt_indx"]))
                            #s_zcr = set(ak.to_list(input_var_struct[input_var_struct["ZMult"] >= 1]["evnt_indx"]))
                    if (curnt_dir == "fakeCR_0jet"):
                        input_var_struct = input_var_struct[np.bitwise_and(input_var_struct["ZMult"] == 0,input_var_struct["numBJet"] == 0)]
                        reco_rad_mass = input_var_struct[np.bitwise_or(input_var_struct["H1OS"] != 0, input_var_struct["H2OS"] != 0)]["RecoRadion_Mass"] #Select for charged Higgs
                        if (file_type != "data_obs"):
                            weight_input = input_var_struct[np.bitwise_or(input_var_struct["H1OS"] != 0, input_var_struct["H2OS"] != 0)]["weight"]
                        if (file_type == "Radion" + str(mass)):
                            s_fakecr = set(ak.to_list(input_var_struct[np.bitwise_or(input_var_struct["H1OS"] != 0, input_var_struct["H2OS"] != 0)]["evnt_indx"]))
                    if (curnt_dir == "highPurity_0jet"):
                        input_var_struct = input_var_struct[np.bitwise_and(input_var_struct["ZMult"] == 0,input_var_struct["numBJet"] == 0)]
                        input_var_struct = input_var_struct[np.bitwise_and(input_var_struct["H1OS"] == 0, input_var_struct["H2OS"] == 0)]
                        reco_rad_mass = input_var_struct[input_var_struct["predictions"] >= 0.7]["RecoRadion_Mass"]
                        if (file_type != "data_obs"):
                            weight_input = input_var_struct[input_var_struct["predictions"] >= 0.7]["weight"]
                        if (file_type == "Radion" + str(mass)):
                            s_highpurity = set(ak.to_list(input_var_struct[input_var_struct["predictions"] >= 0.7]["evnt_indx"]))
                    if (curnt_dir == "lowPurity_0jet"):
                        input_var_struct = input_var_struct[np.bitwise_and(input_var_struct["ZMult"] == 0,input_var_struct["numBJet"] == 0)]
                        input_var_struct = input_var_struct[np.bitwise_and(input_var_struct["H1OS"] == 0, input_var_struct["H2OS"] == 0)]
                        if (file_type != "data_obs"):
                            weight_input = input_var_struct[input_var_struct["predictions"] < 0.7]["weight"]
                        reco_rad_mass = input_var_struct[input_var_struct["predictions"] < 0.7]["RecoRadion_Mass"]
                        if (file_type == "Radion" + str(mass)):
                            s_lowpurity = set(ak.to_list(input_var_struct[input_var_struct["predictions"] < 0.7]["evnt_indx"]))

                    #Append data to histogram
                    if (file_type == "data_obs"):
                        RecoRadion_Hist.fill(reco_rad_mass)
                    else:
                        RecoRadion_Hist.fill(reco_rad_mass,weight=weight_input)
                    if (file_type == "Radion" + str(mass)):
                        num_signal_regions += ak.num(reco_rad_mass,axis=0)


                #Append histogram to output file
                if not(os.path.isfile("HH4t_new.root")):
                    uproot.create("HH4t_new.root")
                with uproot.update("HH4t_new.root") as file:
                    #input_dir = uproot.writing.writable.WritableDirectory(curnt_dir,file)
                    #input_dir[file_type] = RecoRadion_Hist
                    if (file_type != "Radion" + str(mass) and mass == "1000"):
                        file[curnt_dir + "/" + file_type] = RecoRadion_Hist 
                    if (file_type == "Radion" + str(mass)):
                        file[curnt_dir + "/" + file_type] = RecoRadion_Hist 

if (num_signal_regions == num_signal_events):
    print("Number of events is consistent")
else:
    print("!!Number of events mismatch!!")
    print("Number of events: %d"%num_signal_events)
    print("Number of events in all control regions: %d"%num_signal_regions)

print("Intersection of all sets is: " + str(s_ttcr & s_zcr & s_fakecr & s_highpurity & s_lowpurity))
print("ttcr and zcr: " + str(s_ttcr & s_zcr))
print("ttcr and fakecr: " + str(s_ttcr & s_fakecr))
print("ttcr and highpurity: " + str(s_ttcr & s_highpurity))
print("ttcr and lowpurity: " + str(s_ttcr & s_lowpurity))
print("zcr and fakecr: " + str(s_zcr & s_fakecr))
print("zcr and highpurity: " + str(s_zcr & s_highpurity))
print("zcr and lowpurity: " + str(s_zcr & s_lowpurity))
print("fakecr and highpurity:" + str(s_fakecr & s_highpurity))
print("fakecr and lowpurity:" + str(s_fakecr & s_lowpurity))
print("highpurity and lowpurity: " + str(s_highpurity & s_lowpurity))



