# Read NanoAOD and make smaller ttree for fitting & plotting


import ROOT
from ROOT import TLorentzVector, TFile
import numpy as np
import h5py
from optparse import OptionParser

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import *
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from PhysicsTools.NanoAODTools.postprocessing.tools import *
from PhysicsTools.NanoAODTools.postprocessing.modules.jme.JetSysColl import JetSysColl, JetSysObj
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import eventLoop
from PhysicsTools.NanoAODTools.postprocessing.framework.preskimming import preSkim




def append_h5(f, name, data):
    prev_size = f[name].shape[0]
    f[name].resize(( prev_size + data.shape[0]), axis=0)
    f[name][prev_size:] = data


class Outputer:
    def __init__(self, outputFileName="out.root", batch_size = 5000, issignal = False):
        self.batch_size = batch_size
        self.output_name = outputFileName
        self.issignal = np.array([issignal]*batch_size, dtype=np.bool_)
        self.first_write = False
        self.idx = 0
        self.nBatch = 0
        self.reset()

    def reset(self):
        self.idx = 0
        n_pf_cands = 100
        self.jet1_PFCands = np.zeros((self.batch_size, n_pf_cands,4), dtype=np.float16)
        self.jet2_PFCands = np.zeros((self.batch_size, n_pf_cands, 4), dtype=np.float16)
        self.jet1_extraInfo = np.zeros((self.batch_size, 7), dtype=np.float16)
        self.jet2_extraInfo = np.zeros((self.batch_size, 7), dtype=np.float16)
        self.jet_kinematics = np.zeros((self.batch_size, 9), dtype=np.float16)
        self.event_info = np.zeros((self.batch_size, 4), dtype=np.float32)

    def fill_event(self, inTree, jet1, jet2, PFCands, mjj):

        genWeight = inTree.readBranch('genWeight')
        MET = inTree.readBranch('MET_pt')
        MET_phi = inTree.readBranch('MET_phi')
        eventNum = inTree.readBranch('event')

        event_info = [eventNum, MET, MET_phi, genWeight]


        jet_kinematics = [mjj, jet1.pt, jet1.eta, jet1.phi, jet1.msoftdrop, jet2.pt, jet2.eta, jet2.phi, jet2.msoftdrop]

        j1_nPF = min(100, jet1.nPFConstituents)
        j2_nPF = min(100, jet2.nPFConstituents)
        jet1_extraInfo = [jet1.tau1, jet1.tau2, jet1.tau3, jet1.tau4, jet1.lsf3, jet1.btagDeepB, j1_nPF]
        jet2_extraInfo = [jet2.tau1, jet2.tau2, jet2.tau3, jet2.tau4, jet2.lsf3, jet2.btagDeepB, j2_nPF]
        #print(jet1.PFConstituents_Start, jet1.PFConstituents_Start + jet1.nPFConstituents, jet2.PFConstituents_Start, jet2.PFConstituents_Start + jet2.nPFConstituents)
        range1 = range(jet1.PFConstituents_Start, jet1.PFConstituents_Start + j1_nPF, 1)
        range2 = range(jet2.PFConstituents_Start, jet2.PFConstituents_Start + j2_nPF, 1)
        jet1_PFCands = [[PFCands[idx].pt, PFCands[idx].eta, PFCands[idx].phi, PFCands[idx].mass] for idx in range1]
        jet2_PFCands = [[PFCands[idx].pt, PFCands[idx].eta, PFCands[idx].phi, PFCands[idx].mass] for idx in range2]
        



        self.event_info[self.idx] = np.array(event_info, dtype=np.float32)
        self.jet_kinematics[self.idx] = np.array(jet_kinematics, dtype = np.float16)
        self.jet1_extraInfo[self.idx] = np.array(jet1_extraInfo, dtype = np.float16)
        self.jet2_extraInfo[self.idx] = np.array(jet2_extraInfo, dtype = np.float16)
        self.jet1_PFCands[self.idx,:jet1.nPFConstituents] = np.array(jet1_PFCands, dtype = np.float16)
        self.jet2_PFCands[self.idx,:jet2.nPFConstituents] = np.array(jet2_PFCands, dtype = np.float16)

        self.idx +=1
        if(self.idx % self.batch_size == 0): self.write_out()


    def write_out(self):
        self.idx = 0
        print("Writing out batch %i \n" % self.nBatch)
        self.nBatch += 1

        if(not self.first_write):
            self.first_write = True
            print("First write, creating dataset with name %s \n" % self.output_name)
            with h5py.File(self.output_name, "w") as f:
                f.create_dataset("issignal", data=self.issignal, chunks = True, maxshape=None)
                f.create_dataset("event_info", data=self.event_info, chunks = True, maxshape=(None, self.event_info.shape[1]))
                f.create_dataset("jet_kinematics", data=self.jet_kinematics, chunks = True, maxshape=(None, self.jet_kinematics.shape[1]))
                f.create_dataset("jet1_extraInfo", data=self.jet1_extraInfo, chunks = True, maxshape=(None, self.jet1_extraInfo.shape[1]))
                f.create_dataset("jet2_extraInfo", data=self.jet2_extraInfo, chunks = True, maxshape=(None, self.jet2_extraInfo.shape[1]))
                f.create_dataset("jet1_PFCands", data=self.jet1_PFCands, chunks = True, maxshape=(None, self.jet1_PFCands.shape[1], 4))
                f.create_dataset("jet2_PFCands", data=self.jet2_PFCands, chunks = True, maxshape=(None, self.jet2_PFCands.shape[1], 4))

        else:
            with h5py.File(self.output_name, "a") as f:
                append_h5(f,'issignal',self.issignal)
                append_h5(f,'event_info',self.event_info)
                append_h5(f,'jet_kinematics',self.jet_kinematics)
                append_h5(f,'jet1_extraInfo',self.jet1_extraInfo)
                append_h5(f,'jet2_extraInfo',self.jet2_extraInfo)
                append_h5(f,'jet1_PFCands',self.jet1_PFCands)
                append_h5(f,'jet2_PFCands',self.jet2_PFCands)

        self.reset()

    def final_write_out(self):
        if(self.idx < self.batch_size):
            print("Last batch only filled %i events, shortening arrays \n" % self.idx)
            self.jet1_PFCands = self.jet1_PFCands[:self.idx]
            self.jet2_PFCands = self.jet2_PFCands[:self.idx]
            self.jet1_extraInfo = self.jet1_extraInfo[:self.idx]
            self.jet2_extraInfo = self.jet2_extraInfo[:self.idx]
            self.jet_kinematics = self.jet_kinematics[:self.idx] 
            self.event_info = self.event_info[:self.idx]

        self.write_out()








def NanoReader(inputFileName="in.root", outputFileName="out.root", cut=None, json = None):

    inputFile = TFile.Open(inputFileName)
    if(not inputFile): #check for null pointer
        print("Unable to open file %s, exting \n" % inputFileName)
        return 1

    #get input tree
    inTree = inputFile.Get("Events")
    # pre-skimming
    elist,jsonFilter = preSkim(inTree, json, cut)

    #number of events to be processed 
    nTotal = elist.GetN() if elist else inTree.GetEntries()
    
    print('Pre-select %d entries out of %s '%(nTotal,inTree.GetEntries()))


    inTree= InputTree(inTree, elist) 
    out = Outputer(outputFileName)


    # Grab event tree from nanoAOD
    eventBranch = inTree.GetBranch('event')
    treeEntries = eventBranch.GetEntries()

    filters = ["Flag_goodVertices",
    "Flag_globalSuperTightHalo2016Filter",
    "Flag_HBHENoiseFilter",
    "Flag_HBHENoiseIsoFilter",
    "Flag_EcalDeadCellTriggerPrimitiveFilter",
    "Flag_goodVertices"
    ]


    triggers = [
            'HLT_PFHT780',
            'HLT_PFHT890',
            'HLT_PFHT1050',
            'HLT_PFJet500',
            'HLT_AK8PFJet500',
            'HLT_AK8PFHT700_TrimMass50',
            'HLT_AK8PFHT800_TrimMass50',
            'HLT_AK8PFHT900_TrimMass50',
            'HLT_AK8PFJet360_TrimMass30',
            'HLT_AK8PFJet380_TrimMass30',
            'HLT_AK8PFJet400_TrimMass30',
            'HLT_AK8PFJet420_TrimMass30',
            ]

    mjj_cut = 1200.



# -------- Begin Loop-------------------------------------
    entries = inTree.entries
    count = 0
    for entry in xrange(entries):

        count   =   count + 1
        if count % 10000 == 0 :
            print('--------- Processing Event ' + str(count) +'   -- percent complete ' + str(100*count/nTotal) + '% -- ')

        # Grab the event
        event = Event(inTree, entry)



        
        passTrigger = False
        passFilter = True
        for fil in filters:
            passFilter = passFilter and inTree.readBranch(fil)
        for trig in triggers:
            passTrigger = passTrigger or inTree.readBranch(trig)
        if(not passFilter): continue
        if(not passTrigger): continue



        PFCands = Collection(event, "FatJetPFCands")
        AK8Jets = Collection(event, "FatJet")
        MuonsCol = Collection(event, "Muon")
        ElectronsCol = Collection(event, "Electron")
        PhotonsCol = Collection(event, "Photon")

        min_pt = 200
        #keep 2 jets with pt > 200, tight id and have highest softdrop mass
        jet1 = jet2 = None
    
        pf_conts_start = 0 #keep track of indices for PF candidates
        for jet in AK8Jets:
            #jetId : bit1 = loose, bit2 = tight, bit3 = tightLepVeto
            #want tight id
            if((jet.jetId & 2 == 2) and jet.pt > min_pt and abs(jet.eta) < 2.5):
                jet.PFConstituents_Start = pf_conts_start
                if(jet1 == None or jet1.msoftdrop < jet.msoftdrop):
                    jet2 = jet1
                    jet1 = jet
                elif(jet2 == None or jet2.msoftdrop < jet.msoftdrop):
                    jet2 = jet
            pf_conts_start += jet.nPFConstituents

        if(jet1 == None or jet2 == None): continue

        j1_4vec = TLorentzVector()
        j2_4vec = TLorentzVector()
        j1_4vec.SetPtEtaPhiM(jet1.pt, jet1.eta, jet1.phi, jet1.msoftdrop)
        j2_4vec.SetPtEtaPhiM(jet2.pt, jet2.eta, jet2.phi, jet2.msoftdrop)

        dijet = j1_4vec + j2_4vec
        mjj = dijet.M()

        if(mjj< mjj_cut): continue

        count+=1
        out.fill_event(inTree, jet1, jet2, PFCands, mjj)

    out.final_write_out()
    return count


parser = OptionParser()
parser.add_option("-i", "--input", dest = "fin", default = '', help="Input file name")
parser.add_option("-o", "--output", dest = "fout", default = 'test.h5', help="Output file name")
parser.add_option("-c", "--cut", default = '', help="Cut string")
parser.add_option("-j", "--json", default = '', help="Json file name")

options, args = parser.parse_args()

NanoReader(options.fin, options.fout)

