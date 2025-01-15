#!/usr/bin/env python

import os
import re
import sys
import glob
import argparse


def submitJob(samplePath):
    statusDagPath1 = '%s/dag.status' % samplePath
    statusDagPath2 = '%s/dag.dag.status' % samplePath

    print ("statusDagPath1", statusDagPath1)

    errors = []
    submitted = []
    success = []
    try:
        with open(statusDagPath1,'r') as dagFile:
            errors = ["yes" for line in dagFile if (re.search('STATUS_ERROR', line)!=None)]
        with open(statusDagPath1,'r') as dagFile:
            submitted = ["yes"  for line in dagFile if(re.search('STATUS_SUBMITTED', line))!=None]
        with open(statusDagPath1,'r') as dagFile:
            success = ["yes"  for line in dagFile if(re.search('(success)', line))!=None]
    except IOError:
        try:
            with open(statusDagPath1,'r') as dagFile:
                errors = [re.search('STATUS_ERROR', line)!=None for line in dagFile]
            with open(statusDagPath1,'r') as dagFile:
                submitted = [re.search('STATUS_SUBMITTED', line)!=None for line in dagFile]
            with open(statusDagPath1,'r') as dagFile:
                success = [re.search('(success)', line)!=None for line in dagFile]
        except IOError:
            print "\t%s does not seem to be well formatted or have dag status files. Skipping..." %samplePath
            return
    print ("there are ",len(errors), " errors and ",len(submitted)-1," Jobs running currently")
    if (any(success) and len(success)==1):
        print (samplePath, " completed processing fully")

    if any(submitted) and (len(submitted)>1):
        print "\t%s is not done. Waiting to resubmit..." % samplePath
        return

    if any(errors):
        #print "\t Resubmitting %s..." % samplePath
        rescueDag = max(glob.glob('%s/*dag.rescue[0-9][0-9][0-9]' % samplePath))
        print ("file to be submitted: ",rescueDag)
        resubmitCommand = 'farmoutAnalysisJobs --rescue-dag-file=%s' % rescueDag
        os.system(resubmitCommand)
    else:
        pass

def generateSubmitDirs(jobs):
    dirs = []
    #recursive search everything under this for a dag submission/"dags" directory
    for job in jobs:
        dirs += recursivelySearchForDagDir(job)
    return dirs
    
#takes a string directory to search, and returns a list of all directories/subdirectories
#that have a "dags" dir
def recursivelySearchForDagDir(search):
    theSearch = glob.glob(search)
    results = []
    for item in theSearch:
        if re.search('.*/dags$', item):
            results.append(item)
        else:
            results+=recursivelySearchForDagDir(item+'/*')
    return results
        
    
def main(args):
    submitDirs = generateSubmitDirs(args.jobs)
    
    for directory in submitDirs:
        submitJob(directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Resubmit failed jobs to re-nano-ify files')
    parser.add_argument('--jobs',nargs='+',required=True,help='Paths to jobs to resubmit')

    args = parser.parse_args()

    main(args)
