# a class based on ReweightScheme's branch removal utility to prune branches from existing files
import ROOT

class branchRemovalTool:
    def __init__(self):
        pass
        
    def pruneBranches(self, theFileName,branchesToPrune):
        pruneFile = ROOT.TFile(theFileName, "UPDATE")

        if pruneFile.IsZombie():
            raise RuntimeError("Zombie file in pruning: "+theFileName)

        alreadyGrabbedItems = []
        for keyObj in pruneFile.GetListOfKeys():
            if keyObj.GetName() not in alreadyGrabbedItems:
                obj = pruneFile.Get(keyObj.GetName())
                if type(obj) == type(ROOT.TTree()):
                    for branch in branchesToPrune:
                        try:
                            obj.GetBranch(branch).SetStatus(0)
                        except:
                            print('Didn\'t find the branch: '+branch+' in: '+obj.GetName())
                    obj = obj.CloneTree(-1,'fast')
                obj.Write(obj.GetName(), ROOT.TObject.kOverwrite | ROOT.TObject.kSingleKey)
                alreadyGrabbedItems.append(obj.GetName()) #the key obj has been clobbered, but the obj shares it's name explicitly
                    
