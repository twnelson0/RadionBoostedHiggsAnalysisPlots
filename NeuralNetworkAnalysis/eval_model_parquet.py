import pandas as pd
import numpy as np
import json
import tensorflow as tf
import awkward as ak

def read_file(input_file):
    data = pd.read_parquet(input_file,engine="pyarrow")
    return data

def normalize_data(data, norm_data):
    for variable in norm_data["Unnamed: 0"]:
        mean = norm_data[norm_data["Unnamed: 0"] == variable]['mu'].values[0]
        std = norm_data[norm_data["Unnamed: 0"] == variable]['std'].values[0]
        data[variable]  = (data[variable] - mean)/ std
    return data

if __name__ == "__main__":
    #file_array = ["Data_JetHT_mass_2000GeV.parquet", "Data_SingleMuonmass_2000GeV.parquet"]
    #file_array = ["Data_JetHT","Data_SingleMuon","Signal"]
    
    file_array = ["Data_JetHT","Data_SingleMuon","Signal_mass_1000GeV","Signal_mass_2000GeV","Signal_mass_3000GeV","TTToSemiLeptonic","TTTo2L2Nu","TTToHadronic",
            "DYJetsToLL_Pt-50To100","DYJetsToLL_Pt-100To250","DYJetsToLL_Pt-250To400","DYJetsToLL_Pt-400To650","DYJetsToLL_Pt-650ToInf",
            "WZ3l1nu","WZ2l2q","WZ1l1nu2q","ZZ2l2q", "WZ1l3nu", "VV2l2nu", "Tbar-tchan","T-tchan","Tbar-tW","T-tW", 
			"WJetsToLNu_HT-100To200","WJetsToLNu_HT-200To400","WJetsToLNu_HT-400To600","WJetsToLNu_HT-600To800","WJetsToLNu_HT-800To1200","WJetsToLNu_HT-1200To2500","WJetsToLNu_HT-2500ToInf",
			"ZZ4l"]

    for file_name in file_array:
        #Get config file from JSON
        #with open("conf.json",'r') as config_file
        #    config = json.load(config_file)

        #Extract config varaibles
        #model_path = config['model_path']
        #input_tree_name = config['input_tree_name']
        #input_branches = config['input_branches']
        #evaluation_branches = config['evaluation_branches']
        #output_column_name = config['output_column_name']
        #output_tree_name = config['output_tree_name']

        #Load Neural Net Model
        model = tf.keras.models.load_model("model.keras")
        model.summary()

        #Load the data
        data_full = read_file(file_name +".parquet")
        
        #Drop 
        data = data_full.drop("weight",axis=1)
        data = data.drop("ZMult",axis=1)
        data = data.drop("RecoRadion_Mass",axis=1)

        #Read in normalization information and normalize data
        norm_data = pd.read_csv("variable_norm.csv")
        data_norm = normalize_data(data,norm_data)
        data_norm_np = data_norm.to_numpy() #Convert to numpy

        #Run stuff through the model
        print(data_norm)
        print(data_norm_np)
        print(data_norm_np.shape[0])
        print(model.input_shape)
        print(file_name)
        if (data_norm_np.shape[0] != 0):
            print("Not empty")
            predictions = model.predict(data_norm_np)
        else:
            print("Empty")
            predictions = np.array([])
        
        data_full["predictions"] = predictions.flatten()

        #Store parquet files
        data_full.to_parquet(file_name + "with_predictions.parquet")

    



