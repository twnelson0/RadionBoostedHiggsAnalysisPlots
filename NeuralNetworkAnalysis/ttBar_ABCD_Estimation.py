import pandas as pd
import numpy as np
from math import pi
import awkward as ak
import mplhep as hep
import matplotlib.pyplot as plt
import hist


hep.style.use(hep.style.CMS)

if __name__ == "__main__":
    #mass_array = ["2000"]
    mass_array = ["1000","2000","3000"]
    Region_Array = ["B","C","D","A"]
    Est_Dict = {} #Estimated background for each masspoint
    #model_dir = "Old_Pred_Parquet/"
    #model_dir = "New_Pred_Parquet/"
    model_dir = ""

    for mass in mass_array:
        #Set up number of events in B,C and D Regions
        NB = 0
        NB_error = 0
        NC = 0
        NC_error = 0
        ND = 0
        ND_error = 0
        for region_name in Region_Array:
            ttbar_dict = {"ttBar": ["TTToSemiLeptonic","TTTo2L2Nu","TTToHadronic"]}
            background_input_dict = {}
            
            #Get ttbar background
            for sample_name in ttbar_dict["ttBar"]:
                background_input_dict[sample_name] = pd.read_parquet(model_dir + sample_name + "with_predictions.parquet",engine="pyarrow")

            #Apply "universal selections"
            for sample in ttbar_dict["ttBar"]:
                background_input_dict[sample] = background_input_dict[sample][background_input_dict[sample].ZMult < 1] #No Z-Bosons
                background_input_dict[sample] = background_input_dict[sample][np.bitwise_and(background_input_dict[sample].H1OS == 0,background_input_dict[sample].H2OS == 0)] #OSOS Only
            
            if (region_name == "B"):
                fig,ax = plt.subplots()
                four_tau_hist = hist.Hist.new.Regular(10,0,3000,label="$m_{4\\tau}$ (GeV)").Double()
                four_tau_hist_noweight = hist.Hist.new.Regular(10,0,3000).Double()
                weight_sum = 0 
                for sample in ttbar_dict["ttBar"]:
                    background_input_dict[sample] = background_input_dict[sample][background_input_dict[sample].numBJet >= 1] #BJet Multiplicity
                    background_input_dict[sample] = background_input_dict[sample][background_input_dict[sample].predictions >= 0.7] #High Purity
                    four_tau_hist.fill(background_input_dict[sample]["RecoRadion_Mass"].to_numpy(), weight = background_input_dict[sample]["weight"].to_numpy())
                    weight_sum += np.sum(background_input_dict[sample]["weight"].to_numpy())
                    four_tau_hist_noweight.fill(background_input_dict[sample]["RecoRadion_Mass"].to_numpy())
                NB = four_tau_hist.sum()
                NB_real = four_tau_hist_noweight.sum()
                NB_real_error = np.sqrt(NB_real)
                #NB_error = NB_real_error*NB
                NB_error = NB_real_error*(weight_sum/NB_real)
                hep.histplot(four_tau_hist)
                ax.set_title(r"$N_B$ = %.3f $\pm$ %.3f"%(NB,NB_error),loc="center")
                ax.set_yscale('log')
                print("There are %.3f events in region B +/- %.3f"%(NB,NB_error))
                plt.savefig("ttBar_region_" + region_name + "_Mass_" +mass)
                plt.close()

            if (region_name == "C"):
                fig,ax = plt.subplots()
                four_tau_hist = hist.Hist.new.Regular(10,0,3000,label="$m_{4\\tau}$ (GeV)").Double()
                four_tau_hist_noweight = hist.Hist.new.Regular(10,0,3000).Double()
                weight_sum = 0 
                for sample in ttbar_dict["ttBar"]:
                    background_input_dict[sample] = background_input_dict[sample][background_input_dict[sample].numBJet < 1] #0 BJet Multiplicity
                    background_input_dict[sample] = background_input_dict[sample][background_input_dict[sample].predictions < 0.7] #Low Purity
                    four_tau_hist.fill(background_input_dict[sample]["RecoRadion_Mass"].to_numpy(), weight = background_input_dict[sample]["weight"].to_numpy())
                    four_tau_hist_noweight.fill(background_input_dict[sample]["RecoRadion_Mass"].to_numpy())
                    weight_sum += np.sum(background_input_dict[sample]["weight"].to_numpy())
                NC = four_tau_hist.sum()
                NC_real = four_tau_hist_noweight.sum()
                NC_real_error = np.sqrt(NC_real)
                #NC_error = NC_real_error*NC
                NC_error = NC_real_error*weight_sum/NC_real
                hep.histplot(four_tau_hist)
                ax.set_title(r"$N_C$ = %.3f $\pm$ %.3f"%(NC,NC_error),loc="center")
                ax.set_yscale('log')
                print("There are %.3f events in region C +/- %.3f"%(NC,NC_error))
                plt.savefig("ttBar_region_" + region_name + "_Mass_" +mass)
                plt.close()

            if (region_name == "D"):
                fig,ax = plt.subplots()
                four_tau_hist = hist.Hist.new.Regular(10,0,3000,label="$m_{4\\tau}$ (GeV)").Double()
                four_tau_hist_noweight = hist.Hist.new.Regular(10,0,3000).Double()
                weight_sum = 0
                for sample in ttbar_dict["ttBar"]:
                    background_input_dict[sample] = background_input_dict[sample][background_input_dict[sample].numBJet >= 1] #BJet Multiplicity
                    background_input_dict[sample] = background_input_dict[sample][background_input_dict[sample].predictions < 0.7] #Low Purity
                    four_tau_hist.fill(background_input_dict[sample]["RecoRadion_Mass"].to_numpy(), weight = background_input_dict[sample]["weight"].to_numpy())
                    four_tau_hist_noweight.fill(background_input_dict[sample]["RecoRadion_Mass"].to_numpy())
                    weight_sum += np.sum(background_input_dict[sample]["weight"].to_numpy())
                ND = four_tau_hist.sum()
                ND_real = four_tau_hist_noweight.sum()
                ND_real_error = np.sqrt(ND_real/ND_real)
                #ND_error = ND_real_error*ND
                ND_error = ND_real_error*weight_sum/ND_real
                hep.histplot(four_tau_hist)
                ax.set_title(r"$N_D$ = %.3f $\pm$ %.3f"%(ND,ND_error),loc="center")
                ax.set_yscale('log')
                print("There are %.3f events in region D +/- %.3f"%(ND,ND_error))
                plt.savefig("ttBar_region_" + region_name + "_Mass_" +mass)
                plt.close()
            
            if (region_name == "A"): #Obtain old estimate for 
                fig,ax = plt.subplots()
                four_tau_hist = hist.Hist.new.Regular(10,0,3000,label="$m_{4\\tau}$ (GeV)").Double()
                four_tau_hist_noweight = hist.Hist.new.Regular(10,0,3000).Double()
                weight_sum = 0
                for sample in ttbar_dict["ttBar"]:
                    background_input_dict[sample] = background_input_dict[sample][background_input_dict[sample].numBJet < 1] #0 BJet Multiplicity
                    background_input_dict[sample] = background_input_dict[sample][background_input_dict[sample].predictions >= 0.7] #High Purity
                    four_tau_hist.fill(background_input_dict[sample]["RecoRadion_Mass"].to_numpy(), weight = background_input_dict[sample]["weight"].to_numpy())
                    four_tau_hist_noweight.fill(background_input_dict[sample]["RecoRadion_Mass"].to_numpy())
                    weight_sum += np.sum(background_input_dict[sample]["weight"].to_numpy())
                NA = four_tau_hist.sum()
                NA_real = four_tau_hist_noweight.sum()
                NA_real_error = np.sqrt(NA_real)
                #NA_error = NA_real_error*NA
                #NA_error = NA_real_error*(weight_sum)
                NA_error = 0
                hep.histplot(four_tau_hist)
                ax.set_title(r"$N_A$ = %.3f $\pm$ %.3f"%(NA,NA_error),loc="center")
                ax.set_yscale('log')
                print("There are %.3f events in region A +/- %.3f"%(NA,NA_error))
                plt.savefig("ttBar_region_" + region_name + "_Mass_" +mass)
                plt.close()
                #print("There are %f ttBar in high purity region (sans ABCD)"% four_tau_hist.sum())
                #print(str(four_tau_hist.sum()))


        #Obtain the background estimation
        Est_Dict[mass] = NB*NC/ND
        #Unct = np.sqrt((NC/ND)**2 *NB_error**2 + (NB/ND)**2* NC_error**2 + (NC*NB/(ND**2))**2*ND_error**2)
        Unct = Est_Dict[mass]*(NB_error/NB + NC_error/NC + ND_error/ND)
        print("ttBar background estimate for " + mass + " mass point: %.3f pm %.3f"%(Est_Dict[mass],Unct))

