#Prepare JSON files with the list of ntuples we will nano-ify
from bbtautauAnalysisScripts.utilities.dasInterface import dasInterface
import json
import argparse

def main(args):
    #load the json
    with open(args.inputFile, 'r') as readFile:
        datasets = json.load(readFile)
    #print(json.dumps(datasets,sort_keys=True,indent=4))
    for campaign in datasets:
        for datasetName in datasets[campaign]:
            dumpTXT.write(('/'+datasets[campaign][datasetName]+'/'+campaign+'/'+args.dataTier).strip() + '\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Generate file list JSONs from a list of samples')
    parser.add_argument('--dataTier',nargs = '?', choices=['MINIAODSIM','NANOAODSIM','MINIAOD','NANOAOD'],help='Datatier to form the list of files for',default='MINIAODSIM')
    parser.add_argument('--inputFile',nargs='?',help='Input JSON file that has campaign:{name:dataset} format',required=True)
    parser.add_argument('--outputFile',nargs='?',help='Output file name to dump the final list of file in JSON format',required=True)

    args = parser.parse_args()

    with open(args.outputFile+".txt","w") as dumpTXT:
        main(args)




    