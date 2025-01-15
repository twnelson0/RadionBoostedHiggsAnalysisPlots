#!/usr/bin/env python
#script designed to do small sets of skims, with everything necessary fed into it as an argument
#so it can be suitable for 

import sys
import argparse
import re

from bbtautauAnalysisScripts.boostedTauNanoMaker.skimModules.skimManager import skimManager

def main(args):

    theSkimManager = skimManager()
    theSkimManager.skimAFile(fileName = args.inputFile,
                             branchCancelationFileName = args.branchCancelationFile,
                             theCutFile = args.theCutFile,
                             outputFileName = args.outputFileName)

if __name__ == '__main__':
    print('Command to run: '+' '.join(sys.argv))
    parser = argparse.ArgumentParser(description = 'Skim a single HFS nanoAOD ntuple down to size from arguments. For use on condor via scripts mostly')
    parser.add_argument('--inputFile',nargs='?',help='Input file to skim',required=True)
    parser.add_argument('--branchCancelationFile',nargs='?',help='File with RE\'s for canceling branches')
    parser.add_argument('--theCutFile',nargs='?',help='File containing the JSON of cuts to use',required=True)
    parser.add_argument('--outputFileName',nargs='?',help='Location/File name of the output',required=True)

    args=parser.parse_args()
    main(args)
