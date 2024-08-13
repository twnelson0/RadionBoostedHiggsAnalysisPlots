import inspect
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from array import array
from ROOT import *

def compare_train_test(clf, X_train, y_train, X_test, y_test, xlabel, node):
    decisions = [] # list to hold decisions of classifier
    for X,y in ((X_train, y_train), (X_test, y_test)): # train and test
        if hasattr(clf, "predict"): # if predict function exists
            d1 = clf.predict(X[y[:,0]==1])[:, node] # signal
            d2 = clf.predict(X[y[:,1]==1])[:, node] # background ttbb
            d3 = clf.predict(X[y[:,0]==1])[:, node] # background ttcc
            d4 = clf.predict(X[y[:,1]==1])[:, node] # background ttlf
        decisions += [d1, d2, d3, d4] # add to list of classifier decision
    
    highest_decision = max(np.max(d) for d in decisions) # get maximum score
    bin_edges = [] # list to hold bin edges
    bin_edge = -0.1 # start counter for bin_edges
    while bin_edge < highest_decision: # up to highest score
        bin_edge += 0.02 # increment
        bin_edges.append(bin_edge)
   
    # Unbinned Kolmogorov-Smirnov Test on two sorted arrays (requires sorting!):
    x_gaussA_array = array('d', sorted(decisions[0]))
    x_gaussB_array = array('d', sorted(decisions[4]))
    pKS = TMath.KolmogorovTest(len(x_gaussA_array), x_gaussA_array,len(x_gaussB_array), x_gaussB_array, "D")
    print('KS probability class signal :',pKS)
    x_gaussA_array = array('d', sorted(decisions[1]))   
    x_gaussB_array = array('d', sorted(decisions[5]))   
    pKS = TMath.KolmogorovTest(len(x_gaussA_array), x_gaussA_array,len(x_gaussB_array), x_gaussB_array, "D")
    print('KS probability class bkg ttbb :',pKS)
    x_gaussA_array = array('d', sorted(decisions[2]))
    x_gaussB_array = array('d', sorted(decisions[6]))
    pKS = TMath.KolmogorovTest(len(x_gaussA_array), x_gaussA_array,len(x_gaussB_array), x_gaussB_array, "D")
    print('KS probability class bkg ttcc :',pKS)
    x_gaussA_array = array('d', sorted(decisions[3]))
    x_gaussB_array = array('d', sorted(decisions[7]))
    pKS = TMath.KolmogorovTest(len(x_gaussA_array), x_gaussA_array,len(x_gaussB_array), x_gaussB_array, "D")
    print('KS probability class bkg ttlf :',pKS)

    # Plot train-test data
 
    figKS = plt.figure(figsize=(20, 12))
    #print(decisions[0])
    plt.hist(decisions[0],bins=bin_edges,density=True,histtype='stepfilled',color='blue',label='ttHbb (train)',alpha=0.5)
    plt.hist(decisions[1],bins=bin_edges,density=True,histtype='stepfilled',color='orange',label='ttbb (train)',alpha=0.5)
    plt.hist(decisions[2],bins=bin_edges,density=True,histtype='stepfilled',color='mediumpurple',label='ttcc (train)',alpha=0.5)
    plt.hist(decisions[3],bins=bin_edges,density=True,histtype='stepfilled',color='cadetblue',label='ttlf (train)',alpha=0.5)

    hist_ttHbb, bin_edges = np.histogram(decisions[4],bins=bin_edges,density=True )   
    scale = len(decisions[4]) / sum(hist_ttHbb) # between raw and normalised
    err_ttHbb = np.sqrt(hist_ttHbb * scale) / scale # error on test background
    width = 0.1 # histogram bin width
    center = (bin_edges[:-1] + bin_edges[1:]) / 2 # bin centres
    plt.errorbar(x=center, y=hist_ttHbb, yerr=err_ttHbb, fmt='o',c='blue', label='Signal (test)' ) # Signal (test)
    
    hist_ttbb, bin_edges = np.histogram(decisions[5],bins=bin_edges,density=True )
    scale = len(decisions[5]) / sum(hist_ttbb) # between raw and normalised
    err_ttbb = np.sqrt(hist_ttbb * scale) / scale # error on test background
    plt.errorbar(x=center, y=hist_ttbb, yerr=err_ttbb, fmt='o',c='orange', label='ttbb (test)' ) # ttbb (test)

    hist_ttcc, bin_edges = np.histogram(decisions[6],bins=bin_edges,density=True )
    scale = len(decisions[6]) / sum(hist_ttcc) # between raw and normalised
    err_ttcc = np.sqrt(hist_ttcc * scale) / scale # error on test background
    plt.errorbar(x=center, y=hist_ttcc, yerr=err_ttcc, fmt='o',c='mediumpurple', label='ttcc (test)' ) # ttcc (test)

    hist_ttlf, bin_edges = np.histogram(decisions[7],bins=bin_edges,density=True )
    scale = len(decisions[7]) / sum(hist_ttlf) # between raw and normalised
    err_ttlf = np.sqrt(hist_ttlf * scale) / scale # error on test background
    plt.errorbar(x=center, y=hist_ttlf, yerr=err_ttlf, fmt='o',c='cadetblue', label='ttlf (test)' ) # ttlf (test)
    

    plt.xlabel(xlabel) # write x-axis label
    plt.ylabel("Arbitrary units") # write y-axis label
    plt.legend(loc='best') # add legend
    plt.savefig('KS_node_'+str(node)+'_.png')
    plt.close(figKS)


def compare_train_test_binary(clf, X_train, y_train, X_test, y_test, xlabel):
    decisions = [] # list to hold decisions of classifier
    for X,y in ((X_train, y_train), (X_test, y_test)): # train and test
        if hasattr(clf, "predict"): # if predict function exists
            d1 = clf.predict(X[y>0.5])# signal
            d2 = clf.predict(X[y<0.5])# background ttbb
        decisions += [d1, d2] # add to list of classifier decision

    highest_decision = max(np.max(d) for d in decisions) # get maximum score
    bin_edges = [] # list to hold bin edges
    bin_edge = -0.1 # start counter for bin_edges
    while bin_edge < highest_decision: # up to highest score
        bin_edge += 0.02 # increment
        bin_edges.append(bin_edge)

    # Unbinned Kolmogorov-Smirnov Test on two sorted arrays (requires sorting!):
    x_gaussA_array = array('d', sorted(decisions[0]))
    x_gaussB_array = array('d', sorted(decisions[2]))
    pKS = TMath.KolmogorovTest(len(x_gaussA_array), x_gaussA_array,len(x_gaussB_array), x_gaussB_array, "D")
    print('KS probability class signal :',pKS)
    extraString_Signal = 'KS prob. Signal: '+ str("{:.2f}".format(pKS))
    x_gaussA_array = array('d', sorted(decisions[1]))
    x_gaussB_array = array('d', sorted(decisions[3]))
    pKS = TMath.KolmogorovTest(len(x_gaussA_array), x_gaussA_array,len(x_gaussB_array), x_gaussB_array, "D")
    print('KS probability class bkg ttlf :',pKS)
    extraString_Bkg = 'KS prob. Background: '+ str("{:.2f}".format(pKS))

    # Plot train-test data
    figKS = plt.figure(figsize=(20, 12))
    plt.hist(decisions[0],bins=bin_edges,density=True,histtype='stepfilled',color='orange',label='Signal (train)',alpha=0.5)
    plt.hist(decisions[1],bins=bin_edges,density=True,histtype='stepfilled',color='blue',label='Background (train)',alpha=0.5)

    ax = plt.gca()    
    plt.text(0.5, 1.05, "CMS Simulation (Work In Progress)      (13 TeV)", fontweight="bold", horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=28)

    hist_ttHbb, bin_edges = np.histogram(decisions[2],bins=bin_edges,density=True )
    scale = len(decisions[2]) / sum(hist_ttHbb) # between raw and normalised
    err_ttHbb = np.sqrt(hist_ttHbb * scale) / scale # error on test background
    width = 0.1 # histogram bin width
    center = (bin_edges[:-1] + bin_edges[1:]) / 2 # bin centres
    plt.errorbar(x=center, y=hist_ttHbb, yerr=err_ttHbb, fmt='o',c='orange', label='Signal (test)' ) # Signal (test)

    hist_ttbb, bin_edges = np.histogram(decisions[3],bins=bin_edges,density=True )
    scale = len(decisions[3]) / sum(hist_ttbb) # between raw and normalised
    err_ttbb = np.sqrt(hist_ttbb * scale) / scale # error on test background
    plt.errorbar(x=center, y=hist_ttbb, yerr=err_ttbb, fmt='o',c='blue', label='Background (test)' ) # tt (test)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=extraString_Signal))
    handles.append(mpatches.Patch(color='none', label=extraString_Bkg))

    plt.xlabel(xlabel,fontsize=24) # write x-axis label
    plt.ylabel("Arbitrary units",fontsize=24) # write y-axis label
    plt.legend(loc='best',handles=handles,fontsize=20) # add legend
    plt.savefig('KS_binary_classification.png')
    plt.close(figKS)


def selection_criteria(MEM,nbj,njets,mbb): 
    #if(MEM>=0. and nbj>=3 and njets>=4): return True
    #if(MEM>=0. and nbj>=3 and njets>=4 and mbb>=0.): return True
    if(nbj>=4 and njets>=4): return True
    else: return False


def plot_input_features(X, y, idx_label, xlabel):
    decisions = [] # list to hold decisions of classifier
    d1 = X[y==0,idx_label] # signal
    d2 = X[y==1,idx_label] # background ttbb
    #d3 = X[y==2,idx_label] # background ttcc
    #4 = X[y==3,idx_label] # background ttlf
    #decisions += [d1, d2, d3, d4] # add to list 
    decisions += [d1, d2] # add to list 

    highest_decision = max(np.max(d) for d in decisions) # get maximum
    lowest_decision = max(np.min(d) for d in decisions) # get minimum

    bin_edges = [] # list to hold bin edges
    bin_edge =  0.# start counter for bin_edges
    while bin_edge < highest_decision: # up to highest score
        bin_edges.append(bin_edge)
        bin_edge += 0.05*(highest_decision - lowest_decision) # increment

    #if(xlabel=='mbb'):
        #bin_edges = [0.,20.,40.,60.,80.,100.,120.,140.,160.,180.,200.,220.,240.,260.,280.,300.,320.,340.,360.,380.,400.,420.,440.,460.,480.,500.]
    if (xlabel=='vis_mass'):
        bin_edges = [0.,20.,40.,60.,80.,100.,120.,140.,160.,180.,200.,220.,240.,260.,280.,300.]
    if (xlabel=='vis_mass2'):
        bin_edges = [0.,20.,40.,60.,80.,100.,120.,140.,160.,180.,200.,220.,240.,260.,280.,300.]
    if (xlabel=='rad_eta'):
        bin_edges = [-5., -4.75, -4.5, -4.25, -4., -3.75, -3.5, -3.25, -3., -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1., 
                     -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]
    if (xlabel=='higgs2_dr'):
        bin_edges = [0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]
    if (xlabel=='higgs1_dr'):
        bin_edges = [0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]
    if (xlabel=='dphi_H1'):
        bin_edges = [-5., -4.75, -4.5, -4.25, -4., -3.75, -3.5, -3.25, -3., -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1., 
                     -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]
    if (xlabel=='dphi_H1_MET'):
        bin_edges = [-5., -4.75, -4.5, -4.25, -4., -3.75, -3.5, -3.25, -3., -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1., 
                     -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]
    if (xlabel=='dphi_H2_MET'):
        bin_edges = [-5., -4.75, -4.5, -4.25, -4., -3.75, -3.5, -3.25, -3., -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1., 
                     -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]
    if (xlabel=='dr_HH'):
        bin_edges = [1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]
    if (xlabel=='dr_H1_Rad'):
        bin_edges = [-5., -4.75, -4.5, -4.25, -4., -3.75, -3.5, -3.25, -3., -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1., 
                     -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]
    if (xlabel=='dphi_HH'):
        bin_edges = [-5., -4.75, -4.5, -4.25, -4., -3.75, -3.5, -3.25, -3., -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1., 
                     -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]
    if (xlabel=='dr_H2_Rad'):
        bin_edges = [-5., -4.75, -4.5, -4.25, -4., -3.75, -3.5, -3.25, -3., -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1., 
                     -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]
    if (xlabel=='dphi_rad_MET'):
        bin_edges = [-5., -4.75, -4.5, -4.25, -4., -3.75, -3.5, -3.25, -3., -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1., 
                     -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]
    if (xlabel=='dphi_H2'):
        bin_edges = [-5., -4.75, -4.5, -4.25, -4., -3.75, -3.5, -3.25, -3., -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1., 
                     -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]

 

    # Plot train-test data


    fig = plt.figure(figsize=(20, 12))

    ax = plt.gca()    
    plt.text(0.5, 1.05, "CMS Simulation (Work In Progress)      (13 TeV)", fontweight="bold", horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=28)

    plt.hist(decisions[0],bins=bin_edges,density=True,histtype='stepfilled',color='blue',label='Background- ZZ4l + ZZ2l2q',alpha=0.5)
    plt.hist(decisions[1],bins=bin_edges,density=True,histtype='stepfilled',color='orange',label='Signal-GluGluToRadionToHHTo4T_M-1000',alpha=0.5)
    #plt.hist(decisions[2],bins=bin_edges,density=True,histtype='stepfilled',color='mediumpurple',label='ttcc',alpha=0.5)
    #plt.hist(decisions[3],bins=bin_edges,density=True,histtype='stepfilled',color='cadetblue',label='ttlf',alpha=0.5)

    plt.xlabel(xlabel,fontsize=28) # write x-axis label
    plt.ylabel("Arbitrary units",fontsize=28) # write y-axis label
    plt.legend(loc='best',fontsize=24) # add legend
    plt.savefig('Var_'+xlabel+'_.png')
    plt.close(fig)


def AUC_ROC(X_train, y_train, X_test, y_test):

    # fit model
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    pred_prob = clf.predict(X_test)

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh ={}
    auc_ ={}
    fauc_={}

    n_class = 4

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)
        auc_[i] = auc(fpr[i], tpr[i])
        print('AUC',auc_[i])
        fauc_[i] = "{:.2f}".format(auc_[i])
    print(' Length ',len(fpr))

    # plotting
    fig = plt.figure(figsize=(16, 16))
    plt.plot(fpr[0], tpr[0], linestyle='--',linewidth=2,color='blue', label='Class ttHbb vs Rest - AUC:'+str(fauc_[0]))
    plt.plot(fpr[1], tpr[1], linestyle='--',linewidth=2,color='orange', label='Class ttbb vs Rest - AUC:'+str(fauc_[1]))
    plt.plot(fpr[2], tpr[2], linestyle='--',linewidth=2,color='mediumpurple', label='Class ttcc vs Rest - AUC:'+str(fauc_[2]))
    plt.plot(fpr[3], tpr[3], linestyle='--',linewidth=2,color='cadetblue', label='Class ttlf vs Rest - AUC:'+str(fauc_[3]))
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig('Multiclass-ROC.png',dpi=300);
    plt.close(fig)


    myfile = TFile( 'ROC_curves.root', 'RECREATE' )

    for i in range(n_class):
        gr = TGraph(len(fpr[i]),fpr[i],tpr[i] )
        gr.SetLineColor( 2 )  
        gr.SetLineWidth( 4 )
        gr.SetMarkerColor( 4 )
        gr.SetMarkerStyle( 21 )
        gr.SetTitle( 'AUC class '+str(i) )  
        gr.GetXaxis().SetTitle( 'False positive rate' )
        gr.GetYaxis().SetTitle( 'True positive rate' )
        gr.SetName('AUC_class_'+str(i))
        gr.Write()

    myfile.Close()


