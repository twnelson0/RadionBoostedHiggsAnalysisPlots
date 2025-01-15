#Andrew Loeliger
#tool for making chains of trees from different directories, then determining which are the uncommon branches among those trees

import ROOT
import argparse
import glob

def listXOR(listOne,listTwo):
    listThree = [x for x in listOne if x not in listTwo]
    listFour = [x for x in listTwo if x not in listOne]
    return listThree + listFour

def listIntersection(listOne, listTwo):
    return [x for x in listOne if x in listTwo]

def multiListIntersection(listOfLists):
    #we grab a list. We can do the first one without loss of generality
    theList = listOfLists[0]
    #we then make the list the intersection of this list, with all lists.
    #the intersection of itself with itself is trivial, and we're done.
    #otherwise, we do the intersection with the next in the list of lists
    #and replace
    if len(listOfLists) < 2:
        return theList
    else:
        for i in range(1,len(listOfLists)):
            theList = listIntersection(theList, listOfLists[i])
    return theList

def main(args):    
    listOfBranchLists = []
    for directory in args.directories:
        out = glob.glob(directory+"*.root")
        directoryChain = ROOT.TChain("Events")
        for fileName in out:
            directoryChain.Add(fileName)
        chainBranches = [x.GetName() for x in list(directoryChain.GetListOfBranches())]
        listOfBranchLists.append(chainBranches)

    #okay. Now we get the intersection of these lists
    commonBranches = multiListIntersection(listOfBranchLists)
    uncommonBranches = []
    for theList in listOfBranchLists:
        uncommonBranches += listXOR(commonBranches, theList)
    print("Uncommon Branches:\n")
    if args.printJSON:
        for branch in uncommonBranches:
            print('"'+branch+'":"'+branch+'",')
    else:
        print uncommonBranches

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'get uncommon branches between files in different directories')
    parser.add_argument('--directories',nargs='+',required=True, help='Different directories to read from')
    parser.add_argument('--printJSON',action='store_true', help='print the uncommon branches in a simplified JSON style')

    args = parser.parse_args()

    main(args)
