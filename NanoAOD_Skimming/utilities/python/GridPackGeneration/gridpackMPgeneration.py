## written by Fasya Khuzaimah and Shu-Xiao Liu, slightly modified by Shivani Lomte

import os
import shutil


exe1=1


def mkDir(dirName):
    if not os.path.isdir(dirName): os.mkdir(dirName)

def main():

    # First Set
    massList = [3500, 4000, 4500, 5000]

    for mass in massList:

        #dirName = "BulkGraviton_hh_GF_HH_narrow_M%s_slc7_amd64_gcc700_CMSSW_10_6_19_tarball"%mass
        dirName = "BulkGraviton_hh_GF_HH_narrow_M%s"%mass
    
        mkDir('GenerateCards/'+dirName)
        print('create '+dirName)
        shutil.copyfile('cards/BulkGraviton_hh_hdecay_narrow_Mmass_run_card.dat','GenerateCards/'+dirName+'/'+dirName+'_run_card.dat')
        shutil.copyfile('cards/BulkGraviton_hh_hdecay_narrow_Mmass_extramodels.dat','GenerateCards/'+dirName+'/'+dirName+'_extramodels.dat')
        
        f_proc0 = open('cards/BulkGraviton_hh_hdecay_narrow_Mmass_proc_card.dat','r')
        f_proc1 = open('GenerateCards/'+dirName+'/'+dirName+'_proc_card.dat','w')
        for line in f_proc0:
            f_proc1.write(line.replace('BulkGraviton_hh_hdecay_narrow_Mmass',dirName))
        f_proc0.close()
        f_proc1.close()
        
        f_cust0 = open('cards/BulkGraviton_hh_hdecay_narrow_Mmass_customizecards.dat','r')
        f_cust1 = open('GenerateCards/'+dirName+'/'+dirName+'_customizecards.dat','w')
        for line in f_cust0:
            f_cust1.write(line.replace('MASS',str(mass)))
            #if line.find('MZp') > 0: f_cust1.write(line.replace('MZp',str(MZp)))
            #elif line.find('mdm') > 0: f_cust1.write(line.replace('mdm',str(mdm)))
            #else: f_cust1.write(line)
        f_cust0.close()
        f_cust1.close()            
            
        command = '../gridpack_generation.sh ' + dirName + ' GenerateCards/' + dirName
        print(command)
        if (exe1): os.system(command)
                

if __name__ == "__main__":
    main()
#os.system('mv *.tarball.tar.xz /afs/cern.ch/work/s/slomte/public/monoHiggsZpBaryonic_gridpacks/')