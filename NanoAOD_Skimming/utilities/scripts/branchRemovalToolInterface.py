#!/usr/bin/env python
#script for using the branch removal tool on specified files

import ROOT
import argparse
import json
import re
from bbtautauAnalysisScripts.utilities.branchRemovalTool import branchRemovalTool

def main(args):
    theBranchRemovalTool = branchRemovalTool()

    with open (args.fileJson) as jsonFile:
        samplesJson = json.load(jsonFile)

    for sampleKey in samplesJson:
        if not re.search(args.fileRE, sampleKey):
            continue
        theBranchRemovalTool.pruneBranches(samplesJson[sampleKey]['file'],args.branchesToRemove)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Remove specified branches from files')

    parser.add_argument('--fileJson', nargs='?', required=True, help='JSON file to load files from')
    parser.add_argument('--fileRE', nargs='?', default = '.*', help='RE to specifiy files from the JSON to remove the branch from')
    parser.add_argument('--branchesToRemove', nargs='+', required=True, help='Name of the branches to remove')

    args = parser.parse_args()

    main(args)
