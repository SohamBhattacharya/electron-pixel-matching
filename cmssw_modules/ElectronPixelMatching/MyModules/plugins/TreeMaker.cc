// -*- C++ -*-
//
// Package:    EDAnalyzers/TreeMaker
// Class:      TreeMaker
//
/**\class TreeMaker TreeMaker.cc EDAnalyzers/TreeMaker/plugins/TreeMaker.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  
//         Created:  Sat, 11 May 2019 13:14:55 GMT
//
//


// system include files
#include <limits>
#include <memory>

// user include files



#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"
#include "DataFormats/Common/interface/MapOfVectors.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/FWLite/interface/ESHandle.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "ElectronPixelMatching/MyModules/interface/Common.h"
#include "ElectronPixelMatching/MyModules/interface/Constants.h"
#include "ElectronPixelMatching/MyModules/interface/TreeOutputInfo.h"

#include <CLHEP/Matrix/Matrix.h>
#include <CLHEP/Vector/ThreeVector.h>
#include <CLHEP/Vector/ThreeVector.h>

#include <Compression.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TMatrixD.h>
#include <TTree.h> 
#include <TVector2.h> 
#include <TVectorD.h> 

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.


class TreeMaker : public edm::one::EDAnalyzer<edm::one::SharedResources>
{
    public:
    
    explicit TreeMaker(const edm::ParameterSet&);
    ~TreeMaker();
    
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    
    
    private:
    
    virtual void beginJob() override;
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;
    
    hgcal::RecHitTools recHitTools;
    
    TreeOutputInfo::TreeOutput *treeOutput;
    
    // My stuff //
    bool debug;
    bool isGunSample;
    
    bool storeSimHit;
    bool storeRecHit;
    
    double eleGenMatchDeltaR;
    double phoGenMatchDeltaR;
    
    std::vector <std::string> v_gsfEleFromTICLvar;
    std::vector <std::string> v_phoFromTICLvar;
    
    
    // GenEventInfoProduct //
    edm::EDGetTokenT <GenEventInfoProduct> tok_generator;
    
    
    // Gen particles //
    edm::EDGetTokenT <std::vector <reco::GenParticle>> tok_genParticle;
    
    
    // Pileup //
    edm::EDGetTokenT <std::vector <PileupSummaryInfo>> tok_pileup;
    
    
    // Rho //
    edm::EDGetTokenT <double> tok_rho;
    
    
    // SimHits //
    edm::EDGetTokenT <std::vector <PCaloHit>> tok_HGCEESimHit;
    
    
    // RecHits //
    edm::EDGetTokenT <SiPixelRecHitCollection> tok_pixelRecHit;;
    
    edm::EDGetTokenT <edm::SortedCollection <EcalRecHit, edm::StrictWeakOrdering <EcalRecHit>>> tok_EcalEBRechit;
    
    edm::EDGetTokenT <std::vector <reco::PFRecHit>> tok_PFRecHitHGC;
    edm::EDGetTokenT <edm::SortedCollection <HGCRecHit, edm::StrictWeakOrdering <HGCRecHit>>> tok_HGCEERecHit;
    edm::EDGetTokenT <edm::SortedCollection <HGCRecHit, edm::StrictWeakOrdering <HGCRecHit>>> tok_HGCHEFRecHit;
    edm::EDGetTokenT <edm::SortedCollection <HGCRecHit, edm::StrictWeakOrdering <HGCRecHit>>> tok_HGCHEBRecHit;
    
    
    // HGCAL layer clusters //
    edm::EDGetTokenT <reco::CaloClusterCollection> tok_HGCALlayerCluster;
    edm::EDGetTokenT <edm::ValueMap <std::pair<float,float>>> tok_HGCALlayerClusterTime;
    
    
    // TICL //
    edm::EDGetTokenT <std::vector <ticl::Trackster>> tok_TICLtrackster;
    edm::EDGetTokenT <std::vector <reco::PFCluster>> tok_TICLmultiCluster;
    //edm::EDGetTokenT <std::vector <reco::HGCalMultiCluster>> tok_TICLmultiClusterMIP;
    
    
    // Calo particles //
    edm::EDGetTokenT <std::vector <CaloParticle>> tok_caloParticle;
    
    
    // Gsf electrons from TICL //
    //edm::EDGetTokenT <std::vector <reco::GsfElectron>> tok_hgcalEle;
    edm::EDGetTokenT <std::vector <reco::Electron>> tok_hgcalEle;
    edm::EDGetTokenT <edm::MapOfVectors <std::string, double>> tok_gsfEleFromTICLvarMap;
    
    
    // Photons from TICL //
    edm::EDGetTokenT <std::vector <reco::Photon>> tok_phoFromTICL;
    edm::EDGetTokenT <edm::MapOfVectors <std::string, double>> tok_phoFromTICLvarMap;
    
    
    // General tracks //
    edm::EDGetTokenT <std::vector <reco::Track>> tok_generalTrack;
    
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TreeMaker::TreeMaker(const edm::ParameterSet& iConfig) :
    tok_geom{esConsumes<CaloGeometry, CaloGeometryRecord>()}
{
    usesResource("TFileService");
    edm::Service<TFileService> fs;
    
    // Compression
    //fs->file().SetCompressionAlgorithm(ROOT::kLZMA);
    //fs->file().SetCompressionLevel(8);
    
    //now do what ever initialization is needed
    
    treeOutput = new TreeOutputInfo::TreeOutput("tree", fs);
    
    treeOutput->registerVar<ULong64_t>("runNumber");
    treeOutput->registerVar<ULong64_t>("lumiNumber");
    treeOutput->registerVar<ULong64_t>("bxNumber");
    treeOutput->registerVar<ULong64_t>("eventNumber");
    
    treeOutput->registerVar<float>("genEventWeight");
    
    treeOutput->registerVar<int>("genEle_count");
    treeOutput->registerVar<Common::vfloat>("v_genEle_charge");
    treeOutput->registerVar<Common::vfloat>("v_genEle_energy");
    treeOutput->registerVar<Common::vfloat>("v_genEle_px");
    treeOutput->registerVar<Common::vfloat>("v_genEle_py");
    treeOutput->registerVar<Common::vfloat>("v_genEle_pz");
    treeOutput->registerVar<Common::vfloat>("v_genEle_pt");
    treeOutput->registerVar<Common::vfloat>("v_genEle_eta");
    treeOutput->registerVar<Common::vfloat>("v_genEle_phi");
    treeOutput->registerVar<Common::vfloat>("v_genEle_mass");
    treeOutput->registerVar<Common::vfloat>("v_genEle_vtx_x");
    treeOutput->registerVar<Common::vfloat>("v_genEle_vtx_y");
    treeOutput->registerVar<Common::vfloat>("v_genEle_vtx_z");
    treeOutput->registerVar<Common::vfloat>("v_genEle_vtx_rho");
    
    treeOutput->registerVar<float>("pileup_numInteractions");
    treeOutput->registerVar<float>("pileup_trueNumInteractions");
    treeOutput->registerVar<float>("rho");
    
    //treeOutput->registerVar<Common::vvfloat>("ele_hit_x");
    
    treeOutput->registerVar<int>("hgcalEle_count");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_charge");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_energy");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_px");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_py");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_pz");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_pt");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_eta");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_phi");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_mass");
    
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_vtx_x");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_vtx_y");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_vtx_z");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_vtx_rho");
    
    treeOutput->registerVar<Common::vint>("v_hgcalEle_matchedGenEle_idx");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_matchedGenEle_deltaR");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_matchedGenEle_pt");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_matchedGenEle_eta");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_matchedGenEle_phi");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_matchedGenEle_mass");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_matchedGenEle_energy");
    
    treeOutput->registerVar<Common::vint>("v_hgcalEle_gsfTrack_isValid");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_gsfTrack_p");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_gsfTrack_px");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_gsfTrack_py");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_gsfTrack_pz");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_gsfTrack_pt");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_gsfTrack_eta");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_gsfTrack_phi");
    
    treeOutput->registerVar<Common::vint>("v_hgcalEle_gsfTrack_hit_count");
    treeOutput->registerVar<Common::vvint>("vv_hgcalEle_gsfTrack_hit_detector");
    treeOutput->registerVar<Common::vvint>("vv_hgcalEle_gsfTrack_hit_isInnerTracker");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_gsfTrack_hit_globalPos_x");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_gsfTrack_hit_globalPos_y");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_gsfTrack_hit_globalPos_z");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_gsfTrack_hit_globalPos_rho");
    
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_energy");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_ET");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_rawEnergy");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_rawET");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_theta");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_eta");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_phi");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_x");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_y");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_z");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_rho");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_r");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_etaWidth");
    treeOutput->registerVar<Common::vfloat>("v_hgcalEle_SC_phiWidth");
    treeOutput->registerVar<Common::vint>("v_hgcalEle_SC_cluster_count");
    
    treeOutput->registerVar<Common::vint>("v_hgcalEle_SC_clus_count");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_clus_energy");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_clus_x");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_clus_y");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_clus_z");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_clus_theta");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_clus_eta");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_clus_phi");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_clus_rho");
    treeOutput->registerVar<Common::vvint>("vv_hgcalEle_SC_clus_detector");
    treeOutput->registerVar<Common::vvint>("vv_hgcalEle_SC_clus_layer");
    
    treeOutput->registerVar<Common::vint>("v_hgcalEle_SC_hit_count");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_hit_energy");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_hit_energyFrac");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_hit_x");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_hit_y");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_hit_z");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_hit_theta");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_hit_eta");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_hit_phi");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_hit_rho");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_hit_time");
    treeOutput->registerVar<Common::vvfloat>("vv_hgcalEle_SC_hit_timeError");
    treeOutput->registerVar<Common::vvint>("vv_hgcalEle_SC_hit_detector");
    treeOutput->registerVar<Common::vvint>("vv_hgcalEle_SC_hit_layer");
    
    // 
    treeOutput->registerVar<int>("pixelRecHit_count");
    treeOutput->registerVar<Common::vint>("v_pixelRecHit_isValid");
    treeOutput->registerVar<Common::vfloat>("v_pixelRecHit_globalPos_x");
    treeOutput->registerVar<Common::vfloat>("v_pixelRecHit_globalPos_y");
    treeOutput->registerVar<Common::vfloat>("v_pixelRecHit_globalPos_z");
    treeOutput->registerVar<Common::vfloat>("v_pixelRecHit_globalPos_rho");
    
    // My stuff //
    debug = iConfig.getParameter <bool>("debug");
    isGunSample = iConfig.getParameter <bool>("isGunSample");
    
    storeSimHit = iConfig.getParameter <bool>("storeSimHit");
    storeRecHit = iConfig.getParameter <bool>("storeRecHit");
    
    tok_generator = consumes <GenEventInfoProduct>(iConfig.getParameter <edm::InputTag>("label_generator"));
    tok_genParticle = consumes <std::vector <reco::GenParticle>>(iConfig.getParameter <edm::InputTag>("label_genParticle"));
    
    // Pileup //
    tok_pileup = consumes <std::vector <PileupSummaryInfo>>(iConfig.getParameter <edm::InputTag>("label_pileup"));
    tok_rho = consumes <double>(iConfig.getParameter <edm::InputTag>("label_rho"));
    
    // RecHits //
    tok_pixelRecHit = consumes <SiPixelRecHitCollection>(iConfig.getParameter <edm::InputTag>("label_pixelRecHit"));
    
    tok_EcalEBRechit = consumes <edm::SortedCollection <EcalRecHit, edm::StrictWeakOrdering <EcalRecHit>>>(iConfig.getParameter <edm::InputTag>("label_EcalEBRecHit"));
    
    tok_PFRecHitHGC = consumes <std::vector <reco::PFRecHit>>(iConfig.getParameter <edm::InputTag>("label_PFRecHitHGC"));
    tok_HGCEERecHit  = consumes <edm::SortedCollection <HGCRecHit, edm::StrictWeakOrdering <HGCRecHit>>>(iConfig.getParameter <edm::InputTag>("label_HGCEERecHit"));
    tok_HGCHEFRecHit = consumes <edm::SortedCollection <HGCRecHit, edm::StrictWeakOrdering <HGCRecHit>>>(iConfig.getParameter <edm::InputTag>("label_HGCHEFRecHit"));
    tok_HGCHEBRecHit = consumes <edm::SortedCollection <HGCRecHit, edm::StrictWeakOrdering <HGCRecHit>>>(iConfig.getParameter <edm::InputTag>("label_HGCHEBRecHit"));
    
    // TICL electrons //
    //tok_hgcalEle = consumes <std::vector <reco::GsfElectron>>(iConfig.getParameter <edm::InputTag>("label_hgcalEle"));
    tok_hgcalEle = consumes <std::vector <reco::Electron>>(iConfig.getParameter <edm::InputTag>("label_hgcalEle"));
    
    eleGenMatchDeltaR = iConfig.getParameter <double>("eleGenMatchDeltaR");
    phoGenMatchDeltaR = iConfig.getParameter <double>("phoGenMatchDeltaR");
}


TreeMaker::~TreeMaker()
{
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
    
    delete treeOutput;
}


//
// member functions
//


// ------------ method called for each event  ------------
void TreeMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    
    //////////////////// Run info ////////////////////
    ULong64_t runNumber = iEvent.id().run();
    ULong64_t lumiNumber = iEvent.id().luminosityBlock();
    ULong64_t bxNumber = iEvent.bunchCrossing();
    ULong64_t eventNumber = iEvent.id().event();
    
    treeOutput->clearVars();
    
    treeOutput->setVar("runNumber", runNumber);
    treeOutput->setVar("lumiNumber", lumiNumber);
    treeOutput->setVar("bxNumber", bxNumber);
    treeOutput->setVar("eventNumber", eventNumber);
    
    //for(int j = 0; j < 10; j++)
    //{
    //    treeOutput->fillVarV("genEle_energy", eventNumber+(0.1*j));
    //    
    //    vfloat v_tmp;
    //    
    //    for(int k = 0; k < 20; k++)
    //    {
    //        v_tmp.push_back(eventNumber+(0.1*j)+(0.01*k));
    //    }
    //    
    //    treeOutput->fillVarVV("ele_hit_x", v_tmp);
    //}
    
    //treeOutput->fill();
    
    const CaloGeometry *geom = &iSetup.getData(tok_geom);
    recHitTools.setGeometry(*geom);
    
    //////////////////// GenEventInfoProduct ////////////////////
    edm::Handle <GenEventInfoProduct> generatorHandle;
    iEvent.getByToken(tok_generator, generatorHandle);
    GenEventInfoProduct generator = *generatorHandle;
    
    //printf("[%llu] Gen. evt. wt. %0.4g \n", eventNumber, generator.weight());
    treeOutput->setVar("genEventWeight", generator.weight());
    
    //////////////////// Gen particle ////////////////////
    edm::Handle <std::vector <reco::GenParticle>> v_genParticle;
    iEvent.getByToken(tok_genParticle, v_genParticle);
    
    int genEle_count = 0;
    
    std::vector <reco::GenParticle> v_genEle;
    
    for(const auto& part : *v_genParticle)
    {
        int pdgId = part.pdgId();
        int status = part.status();
        
        if(
            abs(pdgId) == 11 && (
                (isGunSample && status == 1) ||
                (!isGunSample && part.isHardProcess())
            )
        )
        {
            printf(
                "[%llu, %llu, %llu] "
                "Gen-ele found: e %0.2f, pt %0.2f, eta %+0.2f, pz %+0.2f, "
                "\n",
                runNumber, lumiNumber, eventNumber,
                part.energy(), part.pt(), part.eta(), part.pz()
            );
            
            //if(fabs(part.eta())> HGCal_minEta && fabs(part.eta()) < HGCal_maxEta && part.pt()> el_minPt && part.pt() < el_maxPt)
            {
                v_genEle.push_back(part);
                
                treeOutput->fillVarV("v_genEle_charge", (float) part.charge());
                treeOutput->fillVarV("v_genEle_energy", part.energy());
                treeOutput->fillVarV("v_genEle_px", part.px());
                treeOutput->fillVarV("v_genEle_py", part.pz());
                treeOutput->fillVarV("v_genEle_pz", part.pz());
                treeOutput->fillVarV("v_genEle_pt", part.pt());
                treeOutput->fillVarV("v_genEle_eta", part.eta());
                treeOutput->fillVarV("v_genEle_phi", part.phi());
                treeOutput->fillVarV("v_genEle_mass", part.mass());
                treeOutput->fillVarV("v_genEle_vtx_x", part.vertex().x());
                treeOutput->fillVarV("v_genEle_vtx_y", part.vertex().y());
                treeOutput->fillVarV("v_genEle_vtx_z", part.vertex().z());
                treeOutput->fillVarV("v_genEle_vtx_rho", part.vertex().rho());
                
                genEle_count++;
            }
        }
    }
    
    treeOutput->setVar("genEle_count", genEle_count);
    
    
    //////////////////// Pileup ////////////////////
    edm::Handle <std::vector <PileupSummaryInfo>> hv_pileUps;
    iEvent.getByToken(tok_pileup, hv_pileUps);
    auto const& pileUpInfo = (*hv_pileUps)[Common::gerInTimePileupSummaryInfoIdx(hv_pileUps)];
    treeOutput->setVar("pileup_numInteractions", (float) pileUpInfo.getPU_NumInteractions());
    treeOutput->setVar("pileup_trueNumInteractions", (float) pileUpInfo.getTrueNumInteractions());
    
    
    //////////////////// Rho ////////////////////
    edm::Handle <double> handle_rho;
    iEvent.getByToken(tok_rho, handle_rho);
    double rho = *handle_rho;
    treeOutput->setVar("rho", rho);
    
    
    //////////////////// RecHit dictionary ////////////////////
    
    //edm::Handle <edm::SortedCollection <EcalRecHit, edm::StrictWeakOrdering <EcalRecHit>>> hv_EcalEBRecHit;
    //iEvent.getByToken(tok_EcalEBRechit, vh_EcalEBRecHit);
    
    edm::Handle <edm::SortedCollection <HGCRecHit, edm::StrictWeakOrdering <HGCRecHit>>> hv_HGCEERecHit;
    iEvent.getByToken(tok_HGCEERecHit, hv_HGCEERecHit);
    
    edm::Handle <edm::SortedCollection <HGCRecHit, edm::StrictWeakOrdering <HGCRecHit>>> hv_HGCHEFRecHit;
    iEvent.getByToken(tok_HGCHEFRecHit, hv_HGCHEFRecHit);
    
    edm::Handle <edm::SortedCollection <HGCRecHit, edm::StrictWeakOrdering <HGCRecHit>>> hv_HGCHEBRecHit;
    iEvent.getByToken(tok_HGCHEBRecHit, hv_HGCHEBRecHit);
    
    edm::Handle <std::vector <reco::PFRecHit>> hv_PFRecHitHGC;
    iEvent.getByToken(tok_PFRecHitHGC, hv_PFRecHitHGC);
    
    
    std::unordered_map <DetId, const HGCRecHit*> m_HGCRecHit;
    std::unordered_map <DetId, const HGCRecHit*> m_HGCEERecHit;
    std::unordered_map <DetId, const reco::PFRecHit*> m_PFRecHitHGC;
    
    int nHGCEERecHit = hv_HGCEERecHit->size();
    
    for(int iRecHit = 0; iRecHit < nHGCEERecHit; iRecHit++)
    {
        const HGCRecHit *recHit = &(*hv_HGCEERecHit)[iRecHit];
        
        m_HGCRecHit[recHit->id()] = recHit;
        m_HGCEERecHit[recHit->id()] = recHit;
    }
    
    int nHGCHEFRecHit = hv_HGCHEFRecHit->size();
    
    for(int iRecHit = 0; iRecHit < nHGCHEFRecHit; iRecHit++)
    {
        const HGCRecHit *recHit = &(*hv_HGCHEFRecHit)[iRecHit];
        
        m_HGCRecHit[recHit->id()] = recHit;
    }

    int nHGCHEBRecHit = hv_HGCHEBRecHit->size();
    
    for(int iRecHit = 0; iRecHit < nHGCHEBRecHit; iRecHit++)
    {
        const HGCRecHit *recHit = &(*hv_HGCHEBRecHit)[iRecHit];
        
        m_HGCRecHit[recHit->id()] = recHit;
    }
    
    int nPFRecHitHGC = hv_PFRecHitHGC->size();
    
    for(int iRecHit = 0; iRecHit < nPFRecHitHGC; iRecHit++)
    {
        const reco::PFRecHit *recHit = &(*hv_PFRecHitHGC)[iRecHit];
        
        m_PFRecHitHGC[recHit->detId()] = recHit;
    }
    
    
    edm::Handle <SiPixelRecHitCollection> hv_pixelRecHit;
    iEvent.getByToken(tok_pixelRecHit, hv_pixelRecHit);
    int nPixelRecHit = hv_pixelRecHit->size();
    
    auto const& recHits = hv_pixelRecHit.product()->data();
    
    int nRecHit = 0;
    for (auto const& recHit : recHits)
    {
        const auto& localPos = recHit.localPosition();
        const auto& globalPos = recHit.globalPosition();
        
        //printf(
        //    "SiPixelRecHit %d: "
        //    "localPos(%0.4f, %0.4f, %0.4f), "
        //    "globalPos(%0.4f, %0.4f, %0.4f), "
        //    "\n",
        //    nRecHit,
        //    localPos.x(), localPos.y(), localPos.z(),
        //    globalPos.x(), globalPos.y(), globalPos.z()
        //);
        
        nRecHit++;
        
        treeOutput->fillVarV("v_pixelRecHit_isValid", (int) recHit.isValid());
        treeOutput->fillVarV("v_pixelRecHit_globalPos_x", globalPos.x());
        treeOutput->fillVarV("v_pixelRecHit_globalPos_y", globalPos.y());
        treeOutput->fillVarV("v_pixelRecHit_globalPos_z", globalPos.z());
        treeOutput->fillVarV("v_pixelRecHit_globalPos_rho", globalPos.perp());
    }
    
    treeOutput->setVar("pixelRecHit_count", nRecHit);
    
    //////////////////// HGCal electrons ////////////////////
    //edm::Handle <std::vector <reco::GsfElectron>> hv_hgcalEle;
    edm::Handle <std::vector <reco::Electron>> hv_hgcalEle;
    iEvent.getByToken(tok_hgcalEle, hv_hgcalEle);
    const auto& v_hgcalEle = *hv_hgcalEle;
    
    //edm::Handle <edm::MapOfVectors <std::string, double>> m_gsfEleFromTICLvarMap;
    //iEvent.getByToken(tok_gsfEleFromTICLvarMap, m_gsfEleFromTICLvarMap);
    
    //int nEleFromTICL = v_gsfEleFromTICL->size();
    //
    //std::map <reco::SuperClusterRef, int> m_gsfEle_superClus;
    //
    //std::vector <CLHEP::HepLorentzVector> v_gsfEleFromTICL_4mom;
    
    int hgcalEle_count = 0;
    
    std::vector<DetId::Detector> v_HGCalDetId = {DetId::HGCalEE, DetId::HGCalHSi, DetId::HGCalHSc};
    
    std::vector <reco::Electron> v_recoEle;
    
    for(const auto& ele : v_hgcalEle)
    {
        DetId seedHitId = ele.superCluster()->seed()->seed();
        
        if(std::find(v_HGCalDetId.begin(), v_HGCalDetId.end(), seedHitId.det()) == v_HGCalDetId.end())
        {
            continue;
        }
        
        printf(
            "[%llu, %llu, %llu] "
            "HGCal ele found: e %0.2f, pt %0.2f, eta %+0.2f, pz %+0.2f, "
            "\n",
            runNumber, lumiNumber, eventNumber,
            ele.energy(), ele.pt(), ele.eta(), ele.pz()
        );
        
        v_recoEle.push_back(ele);
        
        hgcalEle_count++;
        
        treeOutput->fillVarV("v_hgcalEle_charge", (float) ele.charge());
        treeOutput->fillVarV("v_hgcalEle_energy", ele.energy());
        treeOutput->fillVarV("v_hgcalEle_px", ele.px());
        treeOutput->fillVarV("v_hgcalEle_py", ele.pz());
        treeOutput->fillVarV("v_hgcalEle_pz", ele.pz());
        treeOutput->fillVarV("v_hgcalEle_pt", ele.pt());
        treeOutput->fillVarV("v_hgcalEle_eta", ele.eta());
        treeOutput->fillVarV("v_hgcalEle_phi", ele.phi());
        treeOutput->fillVarV("v_hgcalEle_mass", ele.mass());
        
        const auto& ele_vtx_position = ele.vertex();
        treeOutput->fillVarV("v_hgcalEle_vtx_x", ele_vtx_position.x());
        treeOutput->fillVarV("v_hgcalEle_vtx_y", ele_vtx_position.y());
        treeOutput->fillVarV("v_hgcalEle_vtx_z", ele_vtx_position.z());
        treeOutput->fillVarV("v_hgcalEle_vtx_rho", ele_vtx_position.rho());
        
        //printf(
        //    "(%0.2f, %0.2f), "
        //    "(%0.2f, %0.2f), "
        //    "\n",
        //    ele.vz(), std::sqrt(ele.vx()*ele.vx() + ele.vy()*ele.vy()),
        //    ele_vtx_position.z(), ele_vtx_position.rho()
        //);
        
        // Electron GSF track //
        const auto& ele_gsfTrack = ele.gsfTrack();
        
        treeOutput->fillVarV("v_hgcalEle_gsfTrack_isValid", (int) ele_gsfTrack.isNonnull());
        treeOutput->fillVarV("v_hgcalEle_gsfTrack_p", ele_gsfTrack.isNonnull()? ele_gsfTrack->p(): FLT_MAX);
        treeOutput->fillVarV("v_hgcalEle_gsfTrack_px", ele_gsfTrack.isNonnull()? ele_gsfTrack->px(): FLT_MAX);
        treeOutput->fillVarV("v_hgcalEle_gsfTrack_py", ele_gsfTrack.isNonnull()? ele_gsfTrack->py(): FLT_MAX);
        treeOutput->fillVarV("v_hgcalEle_gsfTrack_pz", ele_gsfTrack.isNonnull()? ele_gsfTrack->pz(): FLT_MAX);
        treeOutput->fillVarV("v_hgcalEle_gsfTrack_pt", ele_gsfTrack.isNonnull()? ele_gsfTrack->pt(): FLT_MAX);
        treeOutput->fillVarV("v_hgcalEle_gsfTrack_eta", ele_gsfTrack.isNonnull()? ele_gsfTrack->eta(): FLT_MAX);
        treeOutput->fillVarV("v_hgcalEle_gsfTrack_phi", ele_gsfTrack.isNonnull()? ele_gsfTrack->phi(): FLT_MAX);
        
        int gsfTrack_nHit = 0;
        Common::vint v_hgcalEle_gsfTrack_hit_detector;
        Common::vint v_hgcalEle_gsfTrack_hit_isInnerTracker;
        Common::vfloat v_hgcalEle_gsfTrack_hit_globalPos_x;
        Common::vfloat v_hgcalEle_gsfTrack_hit_globalPos_y;
        Common::vfloat v_hgcalEle_gsfTrack_hit_globalPos_z;
        Common::vfloat v_hgcalEle_gsfTrack_hit_globalPos_rho;
        
        if(ele_gsfTrack.isNonnull())
        {
            for (trackingRecHit_iterator iHit = ele_gsfTrack->recHitsBegin(); iHit != ele_gsfTrack->recHitsEnd(); iHit++)
            {
                const TrackingRecHit *trkHit = (*iHit);
                
                if (trkHit->isValid())
                {
                    gsfTrack_nHit++;
                    const auto& hit_globalPos = trkHit->globalPosition();
                    const auto& hit_subDet = trkHit->det()->subDetector();
                    
                    printf(
                        "gsfTrack rechit: "
                        "(x %0.2f, y %0.2f, z %0.2f, rho %0.2f), "
                        "(subDet %d, isInnerTracker %d), "
                        "\n",
                        hit_globalPos.x(),
                        hit_globalPos.y(),
                        hit_globalPos.z(),
                        hit_globalPos.perp(),
                        
                        (int) hit_subDet,
                        (int) GeomDetEnumerators::isInnerTracker(hit_subDet)
                    );
                    
                    v_hgcalEle_gsfTrack_hit_detector.push_back(hit_subDet);
                    v_hgcalEle_gsfTrack_hit_isInnerTracker.push_back(GeomDetEnumerators::isInnerTracker(hit_subDet));
                    v_hgcalEle_gsfTrack_hit_globalPos_x.push_back(hit_globalPos.x());
                    v_hgcalEle_gsfTrack_hit_globalPos_y.push_back(hit_globalPos.y());
                    v_hgcalEle_gsfTrack_hit_globalPos_z.push_back(hit_globalPos.z());
                    v_hgcalEle_gsfTrack_hit_globalPos_rho.push_back(hit_globalPos.perp());
                }
            }
        }
        
        treeOutput->fillVarV("v_hgcalEle_gsfTrack_hit_count", gsfTrack_nHit);
        treeOutput->fillVarVV("vv_hgcalEle_gsfTrack_hit_detector", v_hgcalEle_gsfTrack_hit_detector);
        treeOutput->fillVarVV("vv_hgcalEle_gsfTrack_hit_isInnerTracker", v_hgcalEle_gsfTrack_hit_isInnerTracker);
        treeOutput->fillVarVV("vv_hgcalEle_gsfTrack_hit_globalPos_x", v_hgcalEle_gsfTrack_hit_globalPos_x);
        treeOutput->fillVarVV("vv_hgcalEle_gsfTrack_hit_globalPos_y", v_hgcalEle_gsfTrack_hit_globalPos_y);
        treeOutput->fillVarVV("vv_hgcalEle_gsfTrack_hit_globalPos_z", v_hgcalEle_gsfTrack_hit_globalPos_z);
        treeOutput->fillVarVV("vv_hgcalEle_gsfTrack_hit_globalPos_rho", v_hgcalEle_gsfTrack_hit_globalPos_rho);
        
        
        // Electron supercluster //
        const auto& ele_SC = ele.superCluster();
        const auto& v_superCluster_HandF = ele_SC->hitsAndFractions();
        const auto& ele_SC_position = ele_SC->position();
        
        treeOutput->fillVarV("v_hgcalEle_SC_energy", ele_SC->energy());
        treeOutput->fillVarV("v_hgcalEle_SC_ET", ele_SC->energy()*std::sin(ele_SC_position.theta()));
        treeOutput->fillVarV("v_hgcalEle_SC_rawEnergy", ele_SC->rawEnergy());
        treeOutput->fillVarV("v_hgcalEle_SC_rawET", ele_SC->rawEnergy()*std::sin(ele_SC_position.theta()));
        treeOutput->fillVarV("v_hgcalEle_SC_theta", ele_SC_position.theta());
        treeOutput->fillVarV("v_hgcalEle_SC_eta", ele_SC->eta());
        treeOutput->fillVarV("v_hgcalEle_SC_phi", ele_SC->phi());
        treeOutput->fillVarV("v_hgcalEle_SC_x", ele_SC_position.x());
        treeOutput->fillVarV("v_hgcalEle_SC_y", ele_SC_position.y());
        treeOutput->fillVarV("v_hgcalEle_SC_z", ele_SC_position.z());
        treeOutput->fillVarV("v_hgcalEle_SC_rho", std::sqrt(ele_SC_position.perp2()));
        treeOutput->fillVarV("v_hgcalEle_SC_r", std::sqrt(ele_SC_position.mag2()));
        treeOutput->fillVarV("v_hgcalEle_SC_etaWidth", ele_SC->etaWidth());
        treeOutput->fillVarV("v_hgcalEle_SC_phiWidth", ele_SC->phiWidth());
        treeOutput->fillVarV("v_hgcalEle_SC_cluster_count", (int) ele_SC->clusters().size());
        
        // SC clusters
        int nCluster = 0;
        
        Common::vfloat v_hgcalEle_SC_clus_energy;
        Common::vfloat v_hgcalEle_SC_clus_x;
        Common::vfloat v_hgcalEle_SC_clus_y;
        Common::vfloat v_hgcalEle_SC_clus_z;
        Common::vfloat v_hgcalEle_SC_clus_theta;
        Common::vfloat v_hgcalEle_SC_clus_eta;
        Common::vfloat v_hgcalEle_SC_clus_phi;
        Common::vfloat v_hgcalEle_SC_clus_rho;
        Common::vint v_hgcalEle_SC_clus_detector;
        Common::vint v_hgcalEle_SC_clus_layer;
        
        const auto& v_superCluster_clus = ele_SC->clusters();
        
        for(const auto &clus : v_superCluster_clus)
        {
            DetId seedId = clus->seed();
            DetId::Detector subDet = seedId.det();
            int seedLayer = recHitTools.getLayer(seedId);
            
            v_hgcalEle_SC_clus_energy.push_back(clus->energy());
            v_hgcalEle_SC_clus_x.push_back(clus->position().x());
            v_hgcalEle_SC_clus_y.push_back(clus->position().y());
            v_hgcalEle_SC_clus_z.push_back(clus->position().z());
            v_hgcalEle_SC_clus_theta.push_back(clus->position().theta());
            v_hgcalEle_SC_clus_eta.push_back(clus->eta());
            v_hgcalEle_SC_clus_phi.push_back(clus->phi());
            v_hgcalEle_SC_clus_rho.push_back(std::sqrt(clus->position().perp2()));
            v_hgcalEle_SC_clus_detector.push_back(subDet);
            v_hgcalEle_SC_clus_layer.push_back(seedLayer);
        }
        
        treeOutput->fillVarV("v_hgcalEle_SC_clus_count", (int) v_superCluster_clus.size());
        
        treeOutput->fillVarVV("vv_hgcalEle_SC_clus_energy", v_hgcalEle_SC_clus_energy);
        treeOutput->fillVarVV("vv_hgcalEle_SC_clus_x", v_hgcalEle_SC_clus_x);
        treeOutput->fillVarVV("vv_hgcalEle_SC_clus_y", v_hgcalEle_SC_clus_y);
        treeOutput->fillVarVV("vv_hgcalEle_SC_clus_z", v_hgcalEle_SC_clus_z);
        treeOutput->fillVarVV("vv_hgcalEle_SC_clus_theta", v_hgcalEle_SC_clus_theta);
        treeOutput->fillVarVV("vv_hgcalEle_SC_clus_eta", v_hgcalEle_SC_clus_eta);
        treeOutput->fillVarVV("vv_hgcalEle_SC_clus_phi", v_hgcalEle_SC_clus_phi);
        treeOutput->fillVarVV("vv_hgcalEle_SC_clus_rho", v_hgcalEle_SC_clus_rho);
        treeOutput->fillVarVV("vv_hgcalEle_SC_clus_detector", v_hgcalEle_SC_clus_detector);
        treeOutput->fillVarVV("vv_hgcalEle_SC_clus_layer", v_hgcalEle_SC_clus_layer);
        
        // SC hits
        int nHit = 0;
        
        Common::vfloat v_hgcalEle_SC_hit_energy;
        Common::vfloat v_hgcalEle_SC_hit_energyFrac;
        Common::vfloat v_hgcalEle_SC_hit_x;
        Common::vfloat v_hgcalEle_SC_hit_y;
        Common::vfloat v_hgcalEle_SC_hit_z;
        Common::vfloat v_hgcalEle_SC_hit_theta;
        Common::vfloat v_hgcalEle_SC_hit_eta;
        Common::vfloat v_hgcalEle_SC_hit_phi;
        Common::vfloat v_hgcalEle_SC_hit_rho;
        Common::vfloat v_hgcalEle_SC_hit_time;
        Common::vfloat v_hgcalEle_SC_hit_timeError;
        Common::vint v_hgcalEle_SC_hit_detector;
        Common::vint v_hgcalEle_SC_hit_layer;
        
        for(const auto &hnf : v_superCluster_HandF)
        {
            DetId hitId = hnf.first;
            double hitEnfrac = hnf.second;
            DetId::Detector subDet = hitId.det();
            
            int hitLayer = recHitTools.getLayer(hitId);
            
            //if (hitLayer> nLayer_)
            //{
            //    continue;
            //}
        
            //if (hitId.det() != subDet_)
            //{
            //    continue;
            //}
            
            auto pfHitIt = m_PFRecHitHGC.find(hitId);
            
            if(pfHitIt == m_PFRecHitHGC.end())
            {
                continue;
            }
            
            const reco::PFRecHit &pfRecHit = *(pfHitIt->second);
            
            auto hgcHitIter = m_HGCRecHit.find(hitId);
            
            if (hgcHitIter == m_HGCRecHit.end())
            {
                printf("HGCRecHit not found for PFRecHit: "
                    "E %0.4f, "
                    "x %0.4f, y %0.4f, z %0.4f, "
                    "eta %0.4f, phi %0.4f, "
                    "det %d, id %u, "
                    "\n",
                    pfRecHit.energy(),
                    pfRecHit.position().x(), pfRecHit.position().y(), pfRecHit.position().z(),
                    pfRecHit.positionREP().eta(), pfRecHit.positionREP().phi(),
                    (int) subDet, hitId.rawId()
                );
                continue;
            }
            
            const HGCRecHit &hgcRecHit = *(hgcHitIter->second);
            
            //printf("HGCal ele SC PFRecHit: E %0.4f, subDet %d, layer %d \n", pfRecHit.energy(), (int) subDet, hitLayer);
            
            nHit++;
            
            v_hgcalEle_SC_hit_energy.push_back(pfRecHit.energy());
            v_hgcalEle_SC_hit_energyFrac.push_back(hitEnfrac);
            v_hgcalEle_SC_hit_x.push_back(pfRecHit.position().x());
            v_hgcalEle_SC_hit_y.push_back(pfRecHit.position().y());
            v_hgcalEle_SC_hit_z.push_back(pfRecHit.position().z());
            v_hgcalEle_SC_hit_theta.push_back(pfRecHit.position().theta());
            v_hgcalEle_SC_hit_eta.push_back(pfRecHit.positionREP().eta());
            v_hgcalEle_SC_hit_phi.push_back(pfRecHit.positionREP().phi());
            v_hgcalEle_SC_hit_rho.push_back(pfRecHit.position().perp()); // positionREP().rho() returns something weird
            v_hgcalEle_SC_hit_time.push_back(hgcRecHit.time());
            v_hgcalEle_SC_hit_timeError.push_back(hgcRecHit.timeError());
            v_hgcalEle_SC_hit_detector.push_back(subDet);
            v_hgcalEle_SC_hit_layer.push_back(hitLayer);

            //v_phoFromTICL_recHit_E.push_back(pfRecHit.energy());
            //v_phoFromTICL_recHit_x.push_back(pfRecHit.position().x());
            //v_phoFromTICL_recHit_y.push_back(pfRecHit.position().y());
            //v_phoFromTICL_recHit_z.push_back(pfRecHit.position().z());
            //v_phoFromTICL_recHit_time.push_back(hgcRecHit.time());
            //v_phoFromTICL_recHit_timeError.push_back(hgcRecHit.timeError());
            //v_phoFromTICL_recHit_eta.push_back(pfRecHit.positionREP().eta());
            //v_phoFromTICL_recHit_phi.push_back(pfRecHit.positionREP().phi());
            //v_phoFromTICL_recHit_ET.push_back(pfRecHit.energy() * std::sin(pfRecHit.position().theta()));
            //v_phoFromTICL_recHit_detector.push_back(hitId.det());
            //v_phoFromTICL_recHit_layer.push_back(recHitTools.getLayer(hitId) - 1); // Start from 0
            //v_phoFromTICL_recHit_isSimHitMatched.push_back(m_simHit.find(hitId) != m_simHit.end());
            //v_phoFromTICL_recHit_SCdEta.push_back(pfRecHit.positionREP().eta() - pho.superCluster()->eta());
            //v_phoFromTICL_recHit_SCdPhi.push_back(reco::deltaPhi(pfRecHit.positionREP().phi(), pho.superCluster()->phi()));
            //v_phoFromTICL_recHit_SCdR.push_back(reco::deltaR(pfRecHit.positionREP(), *pho.superCluster()));
        }
        
        
        //treeOutput->setVar("v_hgcalEle_SC_hit_count", nHit);
        treeOutput->fillVarV("v_hgcalEle_SC_hit_count", nHit);
        
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_energy", v_hgcalEle_SC_hit_energy);
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_energyFrac", v_hgcalEle_SC_hit_energyFrac);
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_x", v_hgcalEle_SC_hit_x);
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_y", v_hgcalEle_SC_hit_y);
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_z", v_hgcalEle_SC_hit_z);
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_theta", v_hgcalEle_SC_hit_theta);
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_eta", v_hgcalEle_SC_hit_eta);
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_phi", v_hgcalEle_SC_hit_phi);
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_rho", v_hgcalEle_SC_hit_rho);
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_time", v_hgcalEle_SC_hit_time);
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_timeError", v_hgcalEle_SC_hit_timeError);
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_detector", v_hgcalEle_SC_hit_detector);
        treeOutput->fillVarVV("vv_hgcalEle_SC_hit_layer", v_hgcalEle_SC_hit_layer);
    }
    
    treeOutput->setVar("hgcalEle_count", hgcalEle_count);
    
    std::vector <int> v_hgcalEle_matchedGenEle_idx;
    std::vector <double> v_hgcalEle_matchedGenEle_deltaR;
    
    Common::getHardestInCone(
        v_recoEle,
        v_genEle,
        v_hgcalEle_matchedGenEle_idx,
        v_hgcalEle_matchedGenEle_deltaR,
        0.3
    );
    
    for(unsigned int iEle = 0; iEle < v_recoEle.size(); iEle++)
    {
        int matchedGenEle_idx = v_hgcalEle_matchedGenEle_idx.at(iEle);
        double matchedGenEle_deltaR = v_hgcalEle_matchedGenEle_deltaR.at(iEle);
        
        //bool isMatched = (matchedGenEle_idx>= 0) && (matchedGenEle_deltaR < eleGenMatchDeltaR);
        bool isMatched = (matchedGenEle_idx>= 0);
        
        const auto& matchedGenEle = isMatched? v_genEle.at(matchedGenEle_idx): reco::Electron();
        
        treeOutput->fillVarV("v_hgcalEle_matchedGenEle_idx", matchedGenEle_idx);
        treeOutput->fillVarV("v_hgcalEle_matchedGenEle_deltaR", matchedGenEle_deltaR);
        
        treeOutput->fillVarV("v_hgcalEle_matchedGenEle_pt", isMatched? matchedGenEle.pt(): FLT_MAX);
        treeOutput->fillVarV("v_hgcalEle_matchedGenEle_eta", isMatched? matchedGenEle.eta(): FLT_MAX);
        treeOutput->fillVarV("v_hgcalEle_matchedGenEle_phi", isMatched? matchedGenEle.phi(): FLT_MAX);
        treeOutput->fillVarV("v_hgcalEle_matchedGenEle_mass", isMatched? matchedGenEle.mass(): FLT_MAX);
        treeOutput->fillVarV("v_hgcalEle_matchedGenEle_energy", isMatched? matchedGenEle.energy(): FLT_MAX);
    }
    
    // Fill tree
    treeOutput->fill();
    
    //#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
    //ESHandle<SetupData> pSetup;
    //iSetup.get<SetupRecord>().get(pSetup);
    //#endif
    
    //printf("\n\n");
    
    fflush(stdout);
    fflush(stderr);
}


// ------------ method called once each job just before starting event loop  ------------
void
TreeMaker::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
TreeMaker::endJob()
{
    fflush(stdout);
    fflush(stderr);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TreeMaker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TreeMaker);
