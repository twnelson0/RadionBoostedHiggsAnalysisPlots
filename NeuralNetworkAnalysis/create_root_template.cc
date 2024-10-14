#include <iostream>
#include "TFile.h"
#include "TDirectory.h"

int main(){
	//std::cout << "Test" << std::endl;
	TFile *f1 = TFile::Open("HH4t_new.root","CREATE");
	TString dir_array[5] = {"TTCR_0jet", "ZCR_0jet", "highPurity_0jet", "lowPurity_0jet", "fakeCR_0jet"};
	for (auto &dir : dir_array){
		std::cout << dir << std::endl;
		f1->mkdir(dir);
	}
	f1->ls();

	f1->Close();

	return 0;	
}
