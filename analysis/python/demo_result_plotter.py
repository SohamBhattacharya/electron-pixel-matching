import awkward as awk
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
import numpy
import uproot

#import ROOT
#import root2matplot as r2mpl

import utils

#plt.rcParams.update({
#    "text.usetex",
#    "font.family",
#    "font.weight",
#    "font.size",
#})


def main() :
    
    l_filename = [
        "analysisOutputNtuple.root:analysisOutputTree"
    ]
    
    h1_demo = hist.Hist(
        hist.axis.Regular(bins = 100, start = -30, stop = 30, name = "x")
    )
        
    h2_demo = hist.Hist(
        hist.axis.Regular(bins = 50, start = 1.4, stop = 3.2, name = "x"),
        hist.axis.Regular(bins = 100, start = 0, stop = 100, name = "y"),
    )
    
    for tree_branches in uproot.iterate(
        files = l_filename,
        expressions = [
            "runNumber",
            "lumiNumber",
            "eventNumber",
            
            "ele_SC_energy",
            "ele_SC_eta",
            "ele_SC_phi",
            
            "ele_vtx_rho",
            "ele_vtx_z",
            
            "ele_wpca_eigval0",
            "ele_wpca_eigval1",
            
            "ele_wpca_eigaxis0_p0",
            "ele_wpca_eigaxis0_p1",
            
            "ele_wpca_eigaxis1_p0",
            "ele_wpca_eigaxis1_p1",
        ],
        language = utils.uproot_lang,
        num_workers = 10,
        #max_num_elements = 1,
        step_size = 100,
    ) :
        
        count = len(tree_branches["ele_vtx_z"])
        a_weights = numpy.ones(count) # dummy weights
        
        h1_demo.fill(
            tree_branches["ele_vtx_z"],
            weight = a_weights
        )
        
        h2_demo.fill(
            numpy.abs(tree_branches["ele_SC_eta"]),
            tree_branches["ele_SC_energy"],
            weight = a_weights
        )
    
    h1_profx = h2_demo.profile("x")
    
    fig = plt.figure(figsize = [10, 5])
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    #
    #fig2 = plt.figure(figsize = [10, 8])
    #ax2 = fig2.add_subplot(1, 1, 1)
    
    mplhep.histplot(h1_demo, ax = ax1)
    mplhep.hist2dplot(h2_demo, ax = ax2)
    mplhep.histplot(h1_profx, ax = ax3)
    
    fig.canvas.draw()
    fig.show()
    fig.canvas.flush_events()
    
    return 0


if (__name__ == "__main__") :
    
    main()