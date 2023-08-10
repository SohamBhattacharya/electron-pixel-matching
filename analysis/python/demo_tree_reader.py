import awkward as awk
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import scipy
import scipy.linalg
import uproot

import utils

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.weight": "bold",
    "font.size": 17,
})


def main() :
    
    l_filename = [
        #"../ntupleTree.root:treeMaker/tree",
        "../data/ntuples/DoubleElectron_FlatPt-1To100-gun_Phase2Fall22DRMiniAOD-noPU_125X_mcRun4_realistic_v2-v1/ntupleTree.root:treeMaker/tree",
        #"../data/ntuples/RelValZEE_14_CMSSW_13_1_0-131X_mcRun4_realistic_v5_2026D95noPU-v1/ntupleTree.root:treeMaker/tree",
        #"../data/ntuples/RelValZEE_14_CMSSW_13_1_0-131X_mcRun4_realistic_v5_2026D95noPU-v1/ntupleTree_numEvent200.root:treeMaker/tree",
    ]
    
    outTreeFileName = "analysisOutputNtuple.root"
    outTreeFile = uproot.writing.writable.recreate(outTreeFileName)
    outTreeName = "analysisOutputTree"
    
    d_outBranch = {
        "runNumber": int,
        "lumiNumber": int,
        "eventNumber": int,
        
        "ele_SC_energy": float,
        "ele_SC_eta": float,
        "ele_SC_phi": float,
        
        "ele_vtx_rho": float,
        "ele_vtx_z": float,
        
        #"ele_wpca_eigval0": float,
        #"ele_wpca_eigval1": float,
        #
        #"ele_wpca_eigaxis0_p0": float,
        #"ele_wpca_eigaxis0_p1": float,
        #
        #"ele_wpca_eigaxis1_p0": float,
        #"ele_wpca_eigaxis1_p1": float,
        
        "ele_wlinear_dz": float,
        "ele_wlinear_centroid_dz": float,
        "ele_pca_dz": float,
        "ele_wpca_dz": float,
        "ele_wpca_inCyl_dz": float,
    }
    
    outTreeFile.mktree(
        name = outTreeName,
        branch_types = d_outBranch,
    )
    
    print("")
    
    nEvent_total = 0
    nEvent_max = 100
    step_size = int(nEvent_max/10) if (nEvent_max/100 >= 2) else 50
    stop_processing = False
    make_plots = True
    show_plots = True
    
    fig_scatter_rhoz = None
    
    if (make_plots) :
        
        fig_scatter_rhoz = plt.figure(figsize = [12, 8])
        colormap = mpl.cm.get_cmap("nipy_spectral").copy()
    
    eventId_cut = ""
    
    #runNumber = 1
    #lumiNumber = 54
    #eventNumber = 53041
    #
    #if (runNumber > 0 and lumiNumber > 0 and eventNumber > 0) :
    #    
    #    eventId_cut = f"& ((runNumber == {runNumber}) & (lumiNumber == {lumiNumber}) & (eventNumber == {eventNumber}))"
    
    tree_cut = f"(hgcalEle_count > 0) {eventId_cut}"
    
    for tree_branches in uproot.iterate(
        files = l_filename,
        expressions = [
            "runNumber",
            "lumiNumber",
            "eventNumber",
            
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
            
            "v_hgcalEle_SC_rawEnergy",
            "v_hgcalEle_SC_energy",
            "v_hgcalEle_SC_ET",
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
        cut = tree_cut,
        language = utils.uproot_lang,
        num_workers = 10,
        #max_num_elements = 1,
        step_size = step_size,
    ) :
        
        #print(tree_branches["hgcalEle_count"])
        print(type(tree_branches))
        
        # Empty the output branches
        for key in d_outBranch:
            
            d_outBranch[key] = []
        
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
        
        # Select the EE rechits
        hgcalEle_SC_hits = hgcalEle_SC_hits[hgcalEle_SC_hits.detector == 8]
        
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
        
        # Set layers
        hgcalEE_nLayer = 26
        d_layerHit = {}
        for layer in range(1, hgcalEE_nLayer+1) :
            
            d_layerHit[f"SC_hits_layer{layer}"] = hgcalEle_SC_hits[hgcalEle_SC_hits.layer == layer]
        
        hgcalEles = awk.zip(
            arrays = {
                "genEleIdx": tree_branches["v_hgcalEle_matchedGenEle_idx"],
                "charge": tree_branches["v_hgcalEle_charge"],
                "pt": tree_branches["v_hgcalEle_pt"],
                "eta": tree_branches["v_hgcalEle_eta"],
                "phi": tree_branches["v_hgcalEle_phi"],
                "energy": tree_branches["v_hgcalEle_energy"],
                
                "SC_rawEnergy": tree_branches["v_hgcalEle_SC_rawEnergy"],
                "SC_energy": tree_branches["v_hgcalEle_SC_energy"],
                "SC_ET": tree_branches["v_hgcalEle_SC_ET"],
                "SC_eta": tree_branches["v_hgcalEle_SC_eta"],
                "SC_phi": tree_branches["v_hgcalEle_SC_phi"],
                "SC_hit_count": tree_branches["v_hgcalEle_SC_hit_count"],
                "SC_hits": hgcalEle_SC_hits,
                
                "gsfTrack_hits": hgcalEle_gsfTrack_hits,
                
                "vtx_rho": tree_branches["v_hgcalEle_vtx_rho"],
                "vtx_z": tree_branches["v_hgcalEle_vtx_z"],
                
                **d_layerHit
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
            
            runNumber = tree_branches["runNumber"][iEvent]
            lumiNumber = tree_branches["lumiNumber"][iEvent]
            eventNumber = tree_branches["eventNumber"][iEvent]
            
            eles = hgcalEles[iEvent]
            nEle = len(eles.pt)
            
            for iEle in range(nEle) :
                
                tag = (
                    "["
                    f"iEvent: {nEvent_total}, "
                    f"runNumber: {runNumber}, "
                    f"lumiNumber: {lumiNumber}, "
                    f"eventNumber: {eventNumber}, "
                    f"iEle: {iEle}"
                    "]"
                )
                
                print(tag)
                
                # Get the indices of the pixel hits on the same half of the detector as the electron
                # That is, hit.z and ele.eta should have the same sign
                #pixHits = pixelRecHits[iEvent]
                #pixHits_idx = (pixHits.z * eles.eta[iEle]) > 0
                #pixHits = pixHits[pixHits_idx]
                
                if (fig_scatter_rhoz) :
                    
                    fig_scatter_rhoz.clf()
                    ax_scatter_rhoz = fig_scatter_rhoz.add_subplot(1, 1, 1)
                    #ax_scatter_rhoz_zoom = fig_scatter_rhoz.add_subplot(2, 1, 2)
                    
                    calo_img = ax_scatter_rhoz.scatter(
                        x = eles.SC_hits[iEle].z,
                        y = eles.SC_hits[iEle].rho,
                        c = eles.SC_hits[iEle].energy,
                        s = 20,
                        norm = mpl.colors.LogNorm(),
                        cmap = colormap,
                    )
                    
                    ax_scatter_rhoz.grid(visible = True, which = "major", axis = "both", linestyle = "--")
                    
                    fig_scatter_rhoz.colorbar(
                        mappable = calo_img,
                        ax = ax_scatter_rhoz,
                        label = "HGCal hit energy [GeV]",
                        location = "right",
                        orientation="vertical",
                    )
                    
                    #
                    #ax_scatter_rhoz.scatter(
                    #    x = pixHits.z,
                    #    y = pixHits.rho,
                    #    #c = "r",
                    #    edgecolors = "r",
                    #    facecolors = "none",
                    #)
                    
                    ax_scatter_rhoz.scatter(
                        x = eles.gsfTrack_hits[iEle].z,
                        y = eles.gsfTrack_hits[iEle].rho,
                        #c = "r",
                        marker = "s",
                        edgecolors = "b",
                        facecolors = "none",
                        label = "GSF track pixel hits"
                    )
                
                det_half = 1
                plot_xrange = numpy.array([-20, 400])
                plot_yrange = numpy.array([-5, 170])
                eta_xrange = numpy.array([0, plot_xrange[-1]])
                
                if (eles.eta[iEle] < 0) :
                    
                    det_half = -1
                    plot_xrange = -1*plot_xrange[::-1]
                    eta_xrange = -1*eta_xrange[::-1]
                
                
                # Weighted linear fit
                wfit_res = numpy.polyfit(x = eles.SC_hits[iEle].z, y = eles.SC_hits[iEle].rho, w = eles.SC_hits[iEle].energy, deg = 1)
                wfit_yval = numpy.polyval(p = wfit_res, x = plot_xrange)
                wfit_dz = (-wfit_res[1]/wfit_res[0]) - eles.vtx_z[iEle]
                
                #wfit_res_tmp = numpy.polyfit(x = eles.SC_hits[iEle].rho, y = eles.SC_hits[iEle].z, w = eles.SC_hits[iEle].energy, deg = 1)
                #wfit_res = [1/wfit_res_tmp[0], -wfit_res_tmp[1]/wfit_res_tmp[0]]
                #wfit_yval = numpy.polyval(p = wfit_res, x = plot_xrange)
                #wfit_dz = (-wfit_res[1]/wfit_res[0]) - eles.vtx_z[iEle]
                
                # PCA
                pca_result = utils.pca_2d(
                    x = eles.SC_hits[iEle].z,
                    y = eles.SC_hits[iEle].rho,
                )
                pca_dz = (-pca_result["eigaxes"][0][1]/pca_result["eigaxes"][0][0]) - eles.vtx_z[iEle]
                
                # Weighted PCA
                wpca_result = utils.pca_2d(
                    x = eles.SC_hits[iEle].z,
                    y = eles.SC_hits[iEle].rho,
                    w = eles.SC_hits[iEle].energy,
                )
                wpca_dz = (-wpca_result["eigaxes"][0][1]/wpca_result["eigaxes"][0][0]) - eles.vtx_z[iEle]
                
                # Weighted PCA in cylinder
                cylinder_rad = 3
                ele_SC_hits_inCylinder = eles.SC_hits[iEle][numpy.abs(numpy.polyval(wpca_result["eigaxes"][0], eles.SC_hits[iEle].z) - eles.SC_hits[iEle].rho) < cylinder_rad]
                wpca_result_inCylinder = utils.pca_2d(
                    x = ele_SC_hits_inCylinder.z,
                    y = ele_SC_hits_inCylinder.rho,
                    w = ele_SC_hits_inCylinder.energy,
                )
                wpca_inCyl_dz = (-wpca_result_inCylinder["eigaxes"][0][1]/wpca_result_inCylinder["eigaxes"][0][0]) - eles.vtx_z[iEle]
                
                
                # Do layerwise stuff
                a_layer_meanrho = numpy.zeros(hgcalEE_nLayer)
                a_layer_meanz = numpy.zeros(hgcalEE_nLayer)
                a_layer_energy = numpy.zeros(hgcalEE_nLayer)
                a_layer_z = numpy.zeros(hgcalEE_nLayer)
                
                for layer in range(1, hgcalEE_nLayer+1) :
                    
                    iLayer = layer-1
                    layer_key = f"SC_hits_layer{layer}"
                    ele_SC_hits_iLayer = eles[layer_key][iEle]
                    a_layer_energy[iLayer] = numpy.sum(ele_SC_hits_iLayer.energy)
                    
                    if (a_layer_energy[iLayer]) :
                        
                        a_layer_z[iLayer] = ele_SC_hits_iLayer.z[0]
                        a_layer_meanrho[iLayer] = numpy.average(ele_SC_hits_iLayer.rho, weights = ele_SC_hits_iLayer.energy)
                        a_layer_meanz[iLayer] = numpy.average(ele_SC_hits_iLayer.z, weights = ele_SC_hits_iLayer.energy)
                        
                
                a_layer_nonzero = numpy.argwhere(a_layer_energy > 0).flatten()
                
                #ax_scatter_rhoz.scatter(
                #    x = a_layer_meanz[a_layer_nonzero],
                #    y = a_layer_meanrho[a_layer_nonzero],
                #    marker = "x",
                #    c = "r",
                #)
                #
                #print(
                #    f"({eles.genEle[iEle].charge}, {eles.charge[iEle]}), "
                #    f"({eles.genEle[iEle].eta}, {eles.eta[iEle]}), "
                #    f"({eles.genEle[iEle].vtx_rho}, {eles.genEle[iEle].vtx_z})"
                #)
                
                wcentroidfit_res = numpy.polyfit(x = a_layer_meanz[a_layer_nonzero], y = a_layer_meanrho[a_layer_nonzero], w = a_layer_energy[a_layer_nonzero], deg = 1)
                wcentroidfit_yval = numpy.polyval(p = wcentroidfit_res, x = plot_xrange)
                wcentroidfit_dz = (-wcentroidfit_res[1]/wcentroidfit_res[0]) - eles.vtx_z[iEle]
                
                
                if (fig_scatter_rhoz) :
                    
                    ax_scatter_rhoz.plot(
                        plot_xrange,
                        wfit_yval,
                        "k--",
                        label = rf"$E$ weighted linear fit [$\Delta z={abs(wfit_dz):0.2f}$ cm]"
                    )
                    
                    ax_scatter_rhoz.plot(
                        plot_xrange,
                        wcentroidfit_yval,
                        "c--",
                        label = rf"$E$ weighted centroid linear fit [$\Delta z={abs(wcentroidfit_dz):0.2f}$ cm]"
                    )
                    
                    axis1_y = numpy.polyval(pca_result["eigaxes"][0], plot_xrange)
                    ax_scatter_rhoz.plot(
                        plot_xrange,
                        axis1_y,
                        "b--",
                        label = rf"PCA axis [$\Delta z={abs(pca_dz):0.2f}$ cm]"
                    )
                    
                    axis1_y = numpy.polyval(wpca_result["eigaxes"][0], plot_xrange)
                    ax_scatter_rhoz.plot(
                        plot_xrange,
                        axis1_y,
                        "r--",
                        label = rf"$E$ weighted PCA axis [$\Delta z={abs(wpca_dz):0.2f}$ cm]"
                    )
                    
                    axis1_y = numpy.polyval(wpca_result_inCylinder["eigaxes"][0], plot_xrange)
                    ax_scatter_rhoz.plot(
                        plot_xrange,
                        axis1_y,
                        "g--",
                        label = rf"$E$ weighted PCA axis (hits within {cylinder_rad} cm) [$\Delta z={abs(wpca_inCyl_dz):0.2f}$ cm]"
                    )
                    
                    # Plot the gen electron vertex
                    ax_scatter_rhoz.scatter(
                        x = eles.genEle[iEle].vtx_z,
                        y = eles.genEle[iEle].vtx_rho,
                        edgecolors = "b",
                        facecolors = "none",
                        #label = "Electron gen vertex",
                    )
                    
                    # Plot the electron vertex
                    ax_scatter_rhoz.scatter(
                        x = eles.vtx_z[iEle],
                        y = eles.vtx_rho[iEle],
                        edgecolors = "magenta",
                        facecolors = "none",
                        label = "Electron vertex",
                    )
                    
                    # Plot HGCal boundary eta lines
                    ax_scatter_rhoz.plot(
                        eta_xrange,
                        numpy.polyval([numpy.tan(utils.pseudorapidity_to_polar(det_half*1.479)), 0], eta_xrange),
                        "k:",
                    )
                    
                    ax_scatter_rhoz.plot(
                        eta_xrange,
                        numpy.polyval([numpy.tan(utils.pseudorapidity_to_polar(det_half*3.1)), 0], eta_xrange),
                        "k:",
                    )
                    
                    ax_scatter_rhoz.set_xlabel("$z$ [cm]", weight='bold')
                    ax_scatter_rhoz.set_ylabel(r"$\rho$ [cm]")
                    
                    ax_scatter_rhoz.set_xlim(plot_xrange)
                    ax_scatter_rhoz.set_ylim(plot_yrange)
                    
                    ax_scatter_rhoz.legend(
                        loc = "upper left" if (det_half > 0) else "upper right",
                        #fontsize = 15,
                    )
                    
                    #ax_scatter_rhoz.set_aspect("equal")
                    
                    fig_scatter_rhoz.suptitle(
                        f"Run: {runNumber}; Lumi: {lumiNumber}; Event: {eventNumber}; Electron {iEle}\n"
                        rf"$E^\mathrm{{gen}}=${eles.genEle[iEle].energy:0.2f} GeV, $p^\mathrm{{gen}}_T=${eles.genEle[iEle].pt:0.2f} GeV, $\eta^\mathrm{{gen}}=${eles.genEle[iEle].eta:0.2f}"
                    )
                    
                    fig_scatter_rhoz.tight_layout()#pad = 0)
                
                if (show_plots) :
                    fig_scatter_rhoz.canvas.draw()
                    fig_scatter_rhoz.show()#block = False)
                    fig_scatter_rhoz.canvas.flush_events()
                    
                
                
                d_outBranch["runNumber"].append(runNumber)
                d_outBranch["lumiNumber"].append(lumiNumber)
                d_outBranch["eventNumber"].append(eventNumber)
                
                d_outBranch["ele_SC_energy"].append(eles.SC_energy[iEle])
                d_outBranch["ele_SC_eta"].append(eles.SC_eta[iEle])
                d_outBranch["ele_SC_phi"].append(eles.SC_phi[iEle])
                
                d_outBranch["ele_vtx_rho"].append(eles.vtx_rho[iEle])
                d_outBranch["ele_vtx_z"].append(eles.vtx_z[iEle])
                
                #d_outBranch["ele_wpca_eigval0"].append(wpca_result["eigvals"][0])
                #d_outBranch["ele_wpca_eigval1"].append(wpca_result["eigvals"][1])
                #
                ## y = p1*x + p0
                ## The axis coefficient list is ordered as [p1, p0]
                #d_outBranch["ele_wpca_eigaxis0_p0"].append(wpca_result["eigaxes"][0][1])
                #d_outBranch["ele_wpca_eigaxis0_p1"].append(wpca_result["eigaxes"][0][0])
                #
                #d_outBranch["ele_wpca_eigaxis1_p0"].append(wpca_result["eigaxes"][1][1])
                #d_outBranch["ele_wpca_eigaxis1_p1"].append(wpca_result["eigaxes"][1][0])
                
                d_outBranch["ele_wlinear_dz"].append(wfit_dz)
                d_outBranch["ele_wlinear_centroid_dz"].append(wcentroidfit_dz)
                d_outBranch["ele_pca_dz"].append(pca_dz)
                d_outBranch["ele_wpca_dz"].append(wpca_dz)
                d_outBranch["ele_wpca_inCyl_dz"].append(wpca_inCyl_dz)
                
            
            nEvent_total += 1
            stop_processing = (nEvent_max > 0 and nEvent_total >= nEvent_max)
            
            if (stop_processing) : break
        
        # Check if all branches are filled equally
        l_branchLen = [len(_l) for _l in d_outBranch.values()]
        assert(min(l_branchLen) == max(l_branchLen))
        
        outTreeFile[outTreeName].extend(d_outBranch)
        
        if (stop_processing) : break
    
    outTreeFile.close()
    
    return 0


if (__name__ == "__main__") :
    
    main()