#Class for interfacing with das queries
import subprocess
import re

class dasInterface():
    def __init__(self):
        pass
    #perform a das query and get the string back
    def performDASQuery(self, theQuery):
        DASCommand = [
            'dasgoclient',
            '--query=%s' % theQuery,
        ]
        p = subprocess.Popen(
            DASCommand,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )

        out, err = p.communicate()
        commandExitCode = p.wait()
        
        if commandExitCode != 0:
            raise RuntimeError('DAS query failed:\n\n'+theQuery+'\n\nError:\n%s' % err+out)
        return out
        
    #perform query and create a list of files out in json/dictionary form
    def getCompleteDictionaryOfFilesFromPathList(self, DASPaths):
        theDict = {}
        listOfResults = []
        for path in DASPaths:
            try:
                assert re.match('/.*/.*/.*', path), "getCompleteDictionaryOfFilesFromQuery: Provided path doesn't match a DAS path structure"
            except AssertionError:
                print('Skipping bad path: '+path)
                continue
            listOfResults += [i.strip() for i in self.performDASQuery('dataset='+path).split('\n') if i.strip()]
        for dataset in listOfResults:
            splitDataset = dataset.split('/')
            if splitDataset[2] not in theDict:# third string is the campaign string. Second is dataset. First is empty.
                theDict[splitDataset[2]] = {}
            stringOfFiles = self.performDASQuery('file dataset='+dataset)
            listOfFiles = [i.strip() for i in stringOfFiles.split('\n') if i.strip()]
            theDict[splitDataset[2]][splitDataset[1]] = listOfFiles
        return theDict
            
