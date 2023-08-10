import awkward as awk
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
import numpy
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
        "output/analysisOutputNtuple.root:analysisOutputTree"
    ]
    
    l_xvar_name = [
        "ele_SC_eta",
    ]
    
    l_xvar_bins = [
        hist.axis.Regular(bins = 50, start = 1.4, stop = 3.2, name = "x")
    ]
    
    l_yvar_name = [
        "ele_wlinear_dz",
    ]
    
    l_yvar_bins = [
        hist.axis.Regular(bins = 100, start = -50, stop = 50, name = "y")
    ]
    
    l_hist2d = []
    
    for ixvar in range(len(l_xvar_name)) :
        for iyvar in range(len(l_yvar_name)) :
            
            h2d_tmp = hist.Hist(
                l_xvar_bins[ixvar],
                l_yvar_bins[iyvar],
            )
            
            l_hist2d.append(h2d_tmp)
            
            for tree_branches in uproot.iterate(
                files = l_filename,
                expressions = [
                    "runNumber",
                    "lumiNumber",
                    "eventNumber",
                ] + [l_xvar_name[ixvar]] + [l_yvar_name[iyvar]],
                language = utils.uproot_lang,
                num_workers = 10,
                #max_num_elements = 1,
                step_size = 100,
            ) :
                
                count = len(tree_branches["runNumber"])
                a_weights = numpy.ones(count) # dummy weights
                
                h2d_tmp.fill(
                    abs(tree_branches[l_xvar_name[ixvar]]),
                    tree_branches[l_yvar_name[iyvar]],
                    weight = a_weights
                )
    
    for hist2d in l_hist2d :
        
        fig = plt.figure(figsize = [10, 5])
        ax1 = fig.add_subplot(1, 1, 1)
        
        hist2d /= hist2d.sum()
        
        mplhep.hist2dplot(hist2d, ax = ax1, norm = mpl.colors.LogNorm())
        
        fig.canvas.draw()
        fig.show()
        fig.canvas.flush_events()
        
        print("Plotted.")
    
    return 0


if (__name__ == "__main__") :
    
    main()