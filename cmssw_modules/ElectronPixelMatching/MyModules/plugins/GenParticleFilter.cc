// -*- C++ -*-
//
// Package:    EDFilters/GenParticleFilter
// Class:      GenParticleFilter
// 
/**\class GenParticleFilter GenParticleFilter.cc EDFilters/GenParticleFilter/plugins/GenParticleFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Soham Bhattacharya
//         Created:  Mon, 11 Nov 2019 13:03:23 GMT
//
//


// system include files
#include <memory>

// user include files
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

//
// class declaration
//

class GenParticleFilter : public edm::stream::EDFilter<>
{
    public:
    
    explicit GenParticleFilter(const edm::ParameterSet&);
    ~GenParticleFilter();
    
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    
    private:
    
    virtual void beginStream(edm::StreamID) override;
    virtual bool filter(edm::Event&, const edm::EventSetup&) override;
    virtual void endStream() override;

    //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    
    // ----------member data ---------------------------
    
    
    // My stuff //
    bool isGunSample;
    
    int atLeastN;
    std::vector <int> v_pdgId;
    
    double minPt;
    double maxPt;
    
    double minAbsEta;
    double maxAbsEta;
    
    bool debug;
    
    
    // Gen particles //
    edm::EDGetTokenT <std::vector <reco::GenParticle> > tok_genParticle;
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
GenParticleFilter::GenParticleFilter(const edm::ParameterSet& iConfig)
{
    //now do what ever initialization is needed
    
    
    isGunSample = iConfig.getParameter <bool>("isGunSample");
    
    atLeastN = iConfig.getParameter <int>("atLeastN");
    v_pdgId = iConfig.getParameter <std::vector <int> >("pdgIds");
    
    minPt = iConfig.getParameter <double>("minPt");
    maxPt = iConfig.getParameter <double>("maxPt");
    
    minAbsEta = iConfig.getParameter <double>("minAbsEta");
    maxAbsEta = iConfig.getParameter <double>("maxAbsEta");
    
    debug = iConfig.getParameter <bool>("debug");
    
    
    // Gen particles //
    tok_genParticle = consumes <std::vector <reco::GenParticle> >(iConfig.getUntrackedParameter <edm::InputTag>("label_genParticle"));
}


GenParticleFilter::~GenParticleFilter()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool GenParticleFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    
    long long eventNumber = iEvent.id().event();
    
    //////////////////// Gen particle ////////////////////
    edm::Handle <std::vector <reco::GenParticle> > v_genParticle;
    iEvent.getByToken(tok_genParticle, v_genParticle);
    
    
    int genPart_n = 0;
    
    for(int iPart = 0; iPart < (int) v_genParticle->size(); iPart++)
    {
        reco::GenParticle part = v_genParticle->at(iPart);
        
        //int pdgId = abs(part.pdgId());
        int pdgId = part.pdgId();
        int status = part.status();
        
        bool idMatched = (std::find(v_pdgId.begin(), v_pdgId.end(), pdgId) != v_pdgId.end());
        
        if(
            idMatched && (
                (isGunSample && status == 1) ||
                (!isGunSample && part.isHardProcess())
            )
        )
        {
            if(debug)
            {
                printf("[%llu] In GenParticleFilter: PDG-ID %+d, E %0.2f, pT %0.2f, eta %+0.2f \n", eventNumber, pdgId, part.energy(), part.pt(), part.eta());
            }
            
            if(
                (fabs(part.eta()) > minAbsEta && fabs(part.eta()) < maxAbsEta) &&
                (part.pt() > minPt && part.pt() < maxPt)
            )
            {
                genPart_n++;
            }
        }
        
        
        if(genPart_n == atLeastN)
        {
            break;
        }
    }
    
    
    if(genPart_n >= atLeastN)
    {
        if(debug)
        {
            printf("Passed GenParticleFilter. \n");
        }
        
        return true;
    }
    
    
    if(debug)
    {
        printf("Failed GenParticleFilter. \n");
    }
    
    return false;
    
    
//#ifdef THIS_IS_AN_EVENT_EXAMPLE
//   Handle<ExampleData> pIn;
//   iEvent.getByLabel("example",pIn);
//#endif
//
//#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
//   ESHandle<SetupData> pSetup;
//   iSetup.get<SetupRecord>().get(pSetup);
//#endif
//   return true;
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
GenParticleFilter::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
GenParticleFilter::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
GenParticleFilter::beginRun(edm::Run const&, edm::EventSetup const&)
{ 
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
GenParticleFilter::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
GenParticleFilter::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
GenParticleFilter::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GenParticleFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    edm::ParameterSetDescription desc;
    //desc.setUnknown();
    desc.setAllowAnything();
    descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GenParticleFilter);
