import awkward as awk
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import uproot

import utils

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.weight": "bold",
    "font.size": 15,
})


def main() :
    
    l_filename = [
        #"../ntupleTree.root:treeMaker/tree",
        #"../data/ntuples/DoubleElectron_FlatPt-1To100-gun_Phase2Fall22DRMiniAOD-noPU_125X_mcRun4_realistic_v2-v1/ntupleTree.root:treeMaker/tree",
        #"../data/ntuples/RelValZEE_14_CMSSW_13_1_0-131X_mcRun4_realistic_v5_2026D95noPU-v1/ntupleTree.root:treeMaker/tree",
        "../data/ntuples/RelValZEE_14_CMSSW_13_1_0-131X_mcRun4_realistic_v5_2026D95noPU-v1/ntupleTree_numEvent200.root:treeMaker/tree",
    ]
    
    print("")
    
    fig_scatter_rhoz = plt.figure(figsize = [8, 5])
    colormap = mpl.cm.get_cmap("nipy_spectral").copy()
    
    for tree_branches in uproot.iterate(
        files = l_filename,
        expressions = [
            "genEle_count",
            "genEle_count",
            "v_genEle_charge",
            "v_genEle_pt",
            "v_genEle_eta",
            "v_genEle_phi",
            "v_genEle_mass",
            "v_genEle_energy",
            "v_genEle_vtx_z",
            "v_genEle_vtx_rho",
            
            "hgcalEle_count",
            "v_hgcalEle_charge",
            "v_hgcalEle_energy",
            "v_hgcalEle_pt",
            "v_hgcalEle_eta",
            "v_hgcalEle_phi",
            
            "v_hgcalEle_vtx_rho",
            "v_hgcalEle_vtx_z",
            
            "v_hgcalEle_matchedGenEle_idx",
            
            "v_hgcalEle_SC_eta",
            "v_hgcalEle_SC_phi",
            
            #"v_hgcalEle_SC_clus_count",
            #"vv_hgcalEle_SC_clus_energy",
            
            "v_hgcalEle_SC_hit_count",
            "vv_hgcalEle_SC_hit_energyTrue",
            #"vv_hgcalEle_SC_hit_energy",
            #"vv_hgcalEle_SC_hit_energyFrac",
            "vv_hgcalEle_SC_hit_rho",
            "vv_hgcalEle_SC_hit_x",
            "vv_hgcalEle_SC_hit_y",
            "vv_hgcalEle_SC_hit_z",
            "vv_hgcalEle_SC_hit_eta",
            "vv_hgcalEle_SC_hit_phi",
            "vv_hgcalEle_SC_hit_detector",
            "vv_hgcalEle_SC_hit_layer",
            
            "v_hgcalEle_gsfTrack_hit_count",
            "vv_hgcalEle_gsfTrack_hit_isInnerTracker",
            "vv_hgcalEle_gsfTrack_hit_globalPos_rho",
            "vv_hgcalEle_gsfTrack_hit_globalPos_x",
            "vv_hgcalEle_gsfTrack_hit_globalPos_y",
            "vv_hgcalEle_gsfTrack_hit_globalPos_z",
            
            "pixelRecHit_count",
            "v_pixelRecHit_globalPos_rho",
            "v_pixelRecHit_globalPos_x",
            "v_pixelRecHit_globalPos_y",
            "v_pixelRecHit_globalPos_z",
        ],
        aliases = {
            "vv_hgcalEle_SC_hit_energyTrue": "vv_hgcalEle_SC_hit_energy*vv_hgcalEle_SC_hit_energyFrac"
        },
        cut = "(hgcalEle_count > 0)",
        language = utils.uproot_lang,
        num_workers = 10,
        #max_num_elements = 10,
        step_size = 5,
    ) :
        
        #print(tree_branches["hgcalEle_count"])
        print(type(tree_branches))
        
        genEles = awk.zip(
            arrays = {
                "charge": tree_branches["v_genEle_charge"],
                "pt": tree_branches["v_genEle_pt"],
                "eta": tree_branches["v_genEle_eta"],
                "phi": tree_branches["v_genEle_phi"],
                "energy": tree_branches["v_genEle_energy"],
                "vtx_rho": tree_branches["v_genEle_vtx_rho"],
                "vtx_z": tree_branches["v_genEle_vtx_z"],
            }
        )
        
        hgcalEle_SC_hits = awk.zip(
            arrays = {
                "energy": tree_branches["vv_hgcalEle_SC_hit_energyTrue"],
                "rho": tree_branches["vv_hgcalEle_SC_hit_rho"],
                "x": tree_branches["vv_hgcalEle_SC_hit_x"],
                "y": tree_branches["vv_hgcalEle_SC_hit_y"],
                "z": tree_branches["vv_hgcalEle_SC_hit_z"],
                "eta": tree_branches["vv_hgcalEle_SC_hit_eta"],
                "phi": tree_branches["vv_hgcalEle_SC_hit_phi"],
                "detector": tree_branches["vv_hgcalEle_SC_hit_detector"],
                "layer": tree_branches["vv_hgcalEle_SC_hit_layer"],
            }
        )
        
        hgcalEle_gsfTrack_hits = awk.zip(
            arrays = {
                "isInnerTracker": tree_branches["vv_hgcalEle_gsfTrack_hit_isInnerTracker"],
                "rho": tree_branches["vv_hgcalEle_gsfTrack_hit_globalPos_rho"],
                "x": tree_branches["vv_hgcalEle_gsfTrack_hit_globalPos_x"],
                "y": tree_branches["vv_hgcalEle_gsfTrack_hit_globalPos_y"],
                "z": tree_branches["vv_hgcalEle_gsfTrack_hit_globalPos_z"],
            }
        )
        
        # Select the pixel (inner tracker) hits
        hgcalEle_gsfTrack_hits = hgcalEle_gsfTrack_hits[hgcalEle_gsfTrack_hits.isInnerTracker > 0]
        
        hgcalEles = awk.zip(
            arrays = {
                "genEleIdx": tree_branches["v_hgcalEle_matchedGenEle_idx"],
                "charge": tree_branches["v_hgcalEle_charge"],
                "pt": tree_branches["v_hgcalEle_pt"],
                "eta": tree_branches["v_hgcalEle_eta"],
                "phi": tree_branches["v_hgcalEle_phi"],
                "energy": tree_branches["v_hgcalEle_energy"],
                
                "SC_eta": tree_branches["v_hgcalEle_SC_eta"],
                "SC_phi": tree_branches["v_hgcalEle_SC_phi"],
                "SC_hit_count": tree_branches["v_hgcalEle_SC_hit_count"],
                "SC_hits": hgcalEle_SC_hits,
                
                "gsfTrack_hits": hgcalEle_gsfTrack_hits,
                
                "vtx_rho": tree_branches["v_hgcalEle_vtx_rho"],
                "vtx_z": tree_branches["v_hgcalEle_vtx_z"],
            },
            depth_limit = 1, # Do not broadcast
        )
        
        hgcalEles = hgcalEles[hgcalEles.genEleIdx >= 0]
        hgcalEles["genEle"] = genEles[hgcalEles.genEleIdx]
        
        pixelRecHits = awk.zip(
            arrays = {
                "rho": tree_branches["v_pixelRecHit_globalPos_rho"],
                "x": tree_branches["v_pixelRecHit_globalPos_x"],
                "y": tree_branches["v_pixelRecHit_globalPos_y"],
                "z": tree_branches["v_pixelRecHit_globalPos_z"],
            },
        )
        
        print("")
        
        # Loop over events, electrons
        # This is slow -- just to mess around with
        # Will use awkward slicing operations later
        
        assert(len(hgcalEles) == len(pixelRecHits))
        nEvent = len(hgcalEles)
        
        for iEvent in range(nEvent) :
            
            eles = hgcalEles[iEvent]
            nEle = len(eles.pt)
            
            for iEle in range(nEle) :
                
                print(iEle)
                fig_scatter_rhoz.clf()
                axes_scatter_rhoz = fig_scatter_rhoz.add_subplot(1, 1, 1)
                
                im = axes_scatter_rhoz.scatter(
                    x = eles.SC_hits[iEle].z,
                    y = eles.SC_hits[iEle].rho,
                    c = eles.SC_hits[iEle].energy,
                    norm = mpl.colors.LogNorm(),
                    cmap = colormap,
                )
                
                axes_scatter_rhoz.grid(visible = True, which = "major", axis = "both", linestyle = "--")
                
                fig_scatter_rhoz.colorbar(
                    mappable = im,
                    ax = axes_scatter_rhoz,
                    label = "HGCal hit energy [GeV]",
                    location = "right",
                    orientation="vertical",
                )
                
                # Get the indices of the pixel hits on the same half of the detector as the electron
                # That is, hit.z and ele.eta should have the same sign
                pixHits = pixelRecHits[iEvent]
                pixHits_idx = (pixHits.z * eles.eta[iEle]) > 0
                pixHits = pixHits[pixHits_idx]
                
                axes_scatter_rhoz.scatter(
                    x = pixHits.z,
                    y = pixHits.rho,
                    #c = "r",
                    edgecolors = "r",
                    facecolors = "none",
                )
                
                axes_scatter_rhoz.scatter(
                    x = eles.gsfTrack_hits[iEle].z,
                    y = eles.gsfTrack_hits[iEle].rho,
                    #c = "r",
                    marker = "s",
                    edgecolors = "b",
                    facecolors = "none",
                )
                
                plot_xrange = numpy.array([-20, 400])
                plot_yrange = numpy.array([-5, 160])
                
                if (eles.eta[iEle] < 0) :
                    
                    plot_xrange = -1*plot_xrange[::-1]
                
                ##fit_res = numpy.polyfit(x = pixHits.z, y = pixHits.rho, deg = 1)
                #fit_res = numpy.polyfit(x = eles.SC_hits[iEle].z, y = eles.SC_hits[iEle].rho, w = eles.SC_hits[iEle].energy, deg = 1)
                #fit_yval = numpy.polyval(p = fit_res, x = plot_xrange)
                #
                #axes_scatter_rhoz.plot(
                #    plot_xrange,
                #    fit_yval,
                #    "r--",
                #)
                
                # Plot the gen electron vertex
                axes_scatter_rhoz.scatter(
                    x = eles.genEle[iEle].vtx_z,
                    y = eles.genEle[iEle].vtx_rho,
                    edgecolors = "b",
                    facecolors = "none",
                )
                
                # Plot the electron vertex
                axes_scatter_rhoz.scatter(
                    x = eles.vtx_z[iEle],
                    y = eles.vtx_rho[iEle],
                    edgecolors = "magenta",
                    facecolors = "none",
                )
                
                print(
                    f"({eles.genEle[iEle].charge}, {eles.charge[iEle]}), "
                    f"({eles.genEle[iEle].eta}, {eles.eta[iEle]}), "
                    f"({eles.genEle[iEle].vtx_rho}, {eles.genEle[iEle].vtx_z})"
                )
                
                axes_scatter_rhoz.set_xlabel("z [cm]", weight='bold')
                axes_scatter_rhoz.set_ylabel(r"$\rho$ [cm]")
                
                axes_scatter_rhoz.set_xlim(plot_xrange)
                axes_scatter_rhoz.set_ylim(plot_yrange)
                
                fig_scatter_rhoz.tight_layout()
                fig_scatter_rhoz.show()#block = False)
                #plt.show(block = False)
                #plt.savefig("test.pdf")
                print("Plotted")
    
    return 0


if (__name__ == "__main__") :
    
    main()