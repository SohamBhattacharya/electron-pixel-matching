import os

import FWCore.ParameterSet.Config as cms

processName = "Demo"

from Configuration.StandardSequences.Eras import eras

d_conditionOpt = {}

d_conditionOpt["Phase2HLTTDRSummer20ReRECOMiniAOD"] = {
    "GT": "auto:phase2_realistic_T15",
    "Geom": "GeometryExtended2026D49",
    "Era": eras.Phase2C9,
}

d_conditionOpt["PhaseIISpring22DRMiniAOD"] = {
    "GT": "auto:phase2_realistic_T21",
    #"GT": "123X_mcRun4_realistic_v11",
    "Geom": "GeometryExtended2026D88",
    "Era": eras.Phase2C17I13M9,
}

############################## Parse arguments ##############################

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing("analysis")

options.register("sourceFile",
    "", # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.string, # string, int, or float
    "File containing list of input files" # Description
)

options.register("dataset",
    "", # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.string, # string, int, or float
    "Dataset name; will fetch filelist from DAS" # Description
)

options.register("outputDir",
    "", # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.string, # string, int, or float
    "Output directory" # Description
)

options.register("outFileBaseName",
    "ntupleTree", # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.string, # string, int, or float
    "Output file base name. Number and extenstion will be added automatically." # Description
)

options.register("outFileNumber",
    -1, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "File number (will be added to the filename if >= 0)" # Description
)

options.register("eventRange",
    [], # Default value
    VarParsing.VarParsing.multiplicity.list, # singleton or list
    VarParsing.VarParsing.varType.string, # string, int, or float
    "Syntax: Run1:Event1-Run2:Event2 Run3:Event3-Run4:Event4(includes both)" # Description
)

options.register("debugFile",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Create debug file" # Description
)

options.register("onRaw",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Running on RAW" # Description
)

options.register("conditionOpt",
    None, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.string, # string, int, or float
    "Condition option. Choices: %s" %(", ".join(list(d_conditionOpt.keys()))) # Description
)

options.register("storeSimHit",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Store sim-hits" # Description
)

options.register("storeRecHit",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Store rec-hits" # Description
)

options.register("storeHGCALlayerClus",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Store HGCal layer clusters" # Description
)

options.register("storeSuperClusTICLclus",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Store info about TICL-electron SC, SC seed, and TICL-cluster matches" # Description
)

options.register("eleGenMatchDeltaR",
    99999, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.float, # string, int, or float
    "DeltaR to use for TICL-electron gen-matching (will store only the gen-matched ones)" # Description
)

options.register("phoGenMatchDeltaR",
    99999, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.float, # string, int, or float
    "DeltaR to use for TICL-photon gen-matching (will store only the gen-matched ones)" # Description
)

options.register("isGunSample",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Is it a particle gun sample" # Description
)

options.register("genEleFilter",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Apply gen-electron filter" # Description
)

options.register("genPhoFilter",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Apply gen-photon filter" # Description
)

options.register("genPartonFilter",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Apply gen-parton filter" # Description
)

options.register("trace",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Trace modules" # Description
)

options.register("memoryCheck",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Check memory usage" # Description
)

options.register("printTime",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Print timing information" # Description
)

options.register("depGraph",
    0, # Default value
    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
    VarParsing.VarParsing.varType.int, # string, int, or float
    "Produce dependency graph only" # Description
)

options.parseArguments()

assert(options.conditionOpt in d_conditionOpt.keys())

process = cms.Process(processName, d_conditionOpt[options.conditionOpt]["Era"])

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
#process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
#process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.StandardSequences.RawToDigi_cff')
#process.load('Configuration.StandardSequences.L1Reco_cff')
#process.load('Configuration.StandardSequences.Reconstruction_cff')
#process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('HLTrigger.Configuration.HLT_75e33_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, d_conditionOpt[options.conditionOpt]["GT"], "")
process.load('Configuration.Geometry.%sReco_cff' %(d_conditionOpt[options.conditionOpt]["Geom"]))
process.load('Configuration.Geometry.%s_cff' %(d_conditionOpt[options.conditionOpt]["Geom"]))

process.load("Geometry.HGCalGeometry.HGCalGeometryESProducer_cfi")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

if (len(options.sourceFile)) :
    
    sourceFile = options.sourceFile

fNames = []

if (len(options.inputFiles)) :
    
    fNames = options.inputFiles

elif (len(options.sourceFile.strip())) :
    
    with open(options.sourceFile) as f:
        
        fNames = f.readlines()

elif (len(options.dataset.strip())) :
    
    import Utilities.General.cmssw_das_client as cmssw_das_client
    
    #dataset = "/DYToLL_M-50_TuneCP5_14TeV-pythia8/Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1/GEN-SIM-DIGI-RAW-MINIAOD"
    
    das_result = cmssw_das_client.get_data(
        query = f"file dataset={options.dataset}",
        limit = 0,
    )
    
    for ires in das_result["data"]:
        
        fNames.append(ires["file"][0]["name"])
    #print(files)

assert(len(fNames))

for iFile, fName in enumerate(fNames) :
    
    if (
        "file:" not in fName and
        "root:" not in fName and
        not fName.startswith("/store")
    ) :
        
        fNames[iFile] = "file:%s" %(fName)

outFileSuffix = ""

if (options.onRaw) :
    
    outFileSuffix = "%s_onRaw" %(outFileSuffix)

if (options.outFileNumber >= 0) :
    
    outFileSuffix = "%s_%d" %(outFileSuffix, options.outFileNumber)

outFile = "%s%s.root" %(options.outFileBaseName, outFileSuffix)

if (len(options.outputDir)) :
    
    os.system("mkdir -p %s" %(options.outputDir))
    
    outFile = "%s/%s" %(options.outputDir, outFile)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(fNames),
    
    # Run1:Event1 to Run2:Event2
    #eventsToProcess = cms.untracked.VEventRange("1:78722-1:78722"),
    
    #duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
)


if (len(options.eventRange)) :
    
    process.source.eventsToProcess = cms.untracked.VEventRange(options.eventRange)


if (options.depGraph) :
    
    process.DependencyGraph = cms.Service("DependencyGraph")
    process.source = cms.Source("EmptySource")
    process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(0))


inputProcessName = ""

if (options.onRaw) :
    
    inputProcessName = processName


label_hgcalEle = cms.InputTag("hltEgammaGsfElectronsUnseeded", "", inputProcessName)
label_hgcalPho = cms.InputTag("photonsHGC", "", inputProcessName)

label_TICLtrackster = cms.InputTag("ticlTrackstersMerge", "", inputProcessName)
label_TICLmultiCluster = cms.InputTag("particleFlowClusterHGCal", "", inputProcessName)

label_hgcalLayerClus = cms.InputTag("hgcalLayerClusters",  "", inputProcessName),


process.treeMaker = cms.EDAnalyzer(
    "TreeMaker",
    
    ############################## My stuff ##############################
    debug = cms.bool(False),
    
    isGunSample = cms.bool(bool(options.isGunSample)),
    
    storeSimHit = cms.bool(bool(options.storeSimHit)),
    storeRecHit = cms.bool(bool(options.storeRecHit)),
    
    eleGenMatchDeltaR = cms.double(options.eleGenMatchDeltaR),
    phoGenMatchDeltaR = cms.double(options.phoGenMatchDeltaR),
    
    
    ############################## GEN ##############################
    
    label_generator = cms.InputTag("generator"),
    label_genParticle = cms.InputTag("genParticles"),
    
    
    ############################## RECO ##############################
    
    label_pileup = cms.InputTag("addPileupInfo"),
    #label_rho = cms.InputTag("fixedGridRhoFastjetAll"),
    label_rho = cms.InputTag("fixedGridRhoFastjetAllTmp"),
    
    label_EcalEBRecHit = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
    
    label_HGCEESimHit = cms.InputTag("g4SimHits", "HGCHitsEE"),
    label_HGCHEFSimHit = cms.InputTag("g4SimHits", "HGCHitsHEfront"), # HE Silicon
    label_HGCEEBSimHit = cms.InputTag("g4SimHits", "HGCHitsHEback"), # HE Scintillator
    
    label_HGCEERecHit = cms.InputTag("HGCalRecHit", "HGCEERecHits"),
    label_HGCHEFRecHit = cms.InputTag("HGCalRecHit", "HGCHEFRecHits"),
    label_HGCHEBRecHit = cms.InputTag("HGCalRecHit", "HGCHEBRecHits"),
    
    label_HGCALlayerCluster = cms.InputTag("hgcalLayerClusters"),
    label_HGCALlayerClusterTime = cms.InputTag("hgcalLayerClusters", "timeLayerCluster"),
    
    label_TICLtrackster = label_TICLtrackster,
    label_TICLmultiCluster = label_TICLmultiCluster,
    
    label_PFRecHitHGC = cms.InputTag("particleFlowRecHitHGC"),
    
    label_caloParticle = cms.InputTag("mix", "MergedCaloTruth"),
    
    label_gsfEleFromMultiClus = cms.InputTag(""),
    label_phoFromMultiClus = cms.InputTag(""),
    
    label_hgcalEle = label_hgcalEle,
    #label_hgcalElevarList = cms.vstring(l_var_TICLele),
    #label_hgcalElevarMap = cms.InputTag("HGCalElectronVarMap", process.HGCalElectronVarMap.instanceName.value(), processName),
    
    label_generalTrack = cms.InputTag("generalTracks"),
    
    #label_phoFromMultiClus = cms.InputTag("photonsHGC", "", "RECO"),
    
    label_hgcalPho = label_hgcalPho,
    #label_hgcalPhovarList = cms.vstring(l_var_TICLpho),
    #label_hgcalPhovarMap = cms.InputTag("HGCalPhotonVarMap", process.HGCalPhotonVarMap.instanceName.value(), processName),
    
    label_pixelRecHit = cms.InputTag("siPixelRecHits"),
)


########## Filters ##########
from ElectronPixelMatching.MyModules.GenParticleFilter_cfi import *

process.genEleFilter = genParticleFilter.clone(
    atLeastN = cms.int32(1),
    pdgIds = cms.vint32(11, -11),
    minPt = cms.double(5.0),
    minAbsEta = cms.double(1.4),
    maxAbsEta = cms.double(3.2),
    isGunSample = cms.bool(bool(options.isGunSample)),
    debug = cms.bool(False),
)

print("Deleting existing output file.")
os.system("rm %s" %(outFile))

# Output file name modification
if (outFile.find("/eos/cms") ==  0) :
    
    outFile = outFile.replace("/eos/cms", "root://eoscms.cern.ch//eos/cms")


# Output
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(outFile)
)

# Aging
from SLHCUpgradeSimulations.Configuration.aging import customise_aging_1000
customise_aging_1000(process)

process.reco_seq = cms.Sequence()

if (options.onRaw) :
    
    process.reco_seq = cms.Sequence(
        process.RawToDigi *
        process.reconstruction
    )
    
    process.reco_seq.associate(process.recosim)

process.p = cms.Path(
    process.genEleFilter *
    process.reco_seq *
    process.treeMaker
)

process.schedule = cms.Schedule(*[
    process.L1T_TkEm51,
    process.L1T_TkEle36,
    process.L1T_TkIsoEm36,
    process.L1T_TkIsoEle28,
    #fragment.L1T_TkEm37TkEm24,
    #fragment.L1T_TkEle25TkEle12,
    #fragment.L1T_TkIsoEm22TkIsoEm12,
    #fragment.L1T_TkIsoEle22TkEm12,
    
    process.HLT_Ele32_WPTight_Unseeded,
    
    process.p
])

print("\n")
print("*"*50)
print("process.schedule:", process.schedule)
print("*"*50)
#print "process.schedule.__dict__:", process.schedule.__dict__
#print "*"*50
print("\n")


# Tracer
if (options.trace) :
    
    process.Tracer = cms.Service("Tracer")


if (options.memoryCheck) :
    
    process.SimpleMemoryCheck = cms.Service(
        "SimpleMemoryCheck",
        moduleMemorySummary = cms.untracked.bool(True),
    )


#Timing
if (options.printTime) :

    process.Timing = cms.Service("Timing",
        summaryOnly = cms.untracked.bool(False),
        useJobReport = cms.untracked.bool(True)
    )


# Debug
if (options.debugFile) :
    
    process.out = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string("debug.root")
    )
    
    process.output_step = cms.EndPath(process.out)
    process.schedule.extend([process.output_step])


process.MessageLogger = cms.Service(
    "MessageLogger",
    destinations = cms.untracked.vstring(
        "cerr",
    ),
    cerr = cms.untracked.PSet(
        #threshold  = cms.untracked.string("ERROR"),
        DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        WARNING = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        ERROR = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    )
)


#from FWCore.ParameterSet.Utilities import convertToUnscheduled
#process = convertToUnscheduled(process)


# Add early deletion of temporary data products to reduce peak memory need
#from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
#process = customiseEarlyDelete(process)
# End adding early deletion
