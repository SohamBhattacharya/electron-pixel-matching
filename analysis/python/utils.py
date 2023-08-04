import argparse
import awkward
import ctypes
import gc
import glob
import io
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
#import mplhep
import numpy
import os
import psutil
import re
import scipy
import sortedcontainers
import sparse
import subprocess
import time
import uproot
import yaml

#import ROOT
#ROOT.gROOT.SetBatch(1)

#import CMS_lumi


#class ColorPalette :
#    
#    def __init__(
#        self,
#        a_r,
#        a_g,
#        a_b,
#        a_stop,
#    ) :
#        
#        
#        self.a_r = a_r
#        self.a_g = a_g
#        self.a_b = a_b
#        self.a_stop = a_stop
#        
#        self.nStop = len(a_stop)
#    
#    def set(self, nContour = 500) :
#        
#        ROOT.gStyle.SetNumberContours(nContour)
#        ROOT.TColor.CreateGradientColorTable(self.nStop, self.a_stop, self.a_r, self.a_g, self.a_b, nContour)
#
#
#cpalette_nipy_spectral = ColorPalette(
#    a_r = numpy.array([0.0, 0.4667, 0.5333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7333, 0.9333, 1.0, 1.0, 1.0, 0.8667, 0.8, 0.8]),
#    a_g = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.4667, 0.6, 0.6667, 0.6667, 0.6, 0.7333, 0.8667, 1.0, 1.0, 0.9333, 0.8, 0.6, 0.0, 0.0, 0.0, 0.8]),
#    a_b = numpy.array([0.0, 0.5333, 0.6, 0.6667, 0.8667, 0.8667, 0.8667, 0.6667, 0.5333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]),
#    a_stop = numpy.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
#)


uproot_lang = uproot.language.python.PythonLanguage()
uproot_lang.functions["min"] = numpy.minimum
uproot_lang.functions["max"] = numpy.maximum
uproot_lang.functions["where"] = numpy.where
uproot_lang.functions["nonzero"] = numpy.nonzero
uproot_lang.functions["ak_where"] = awkward.where
uproot_lang.functions["ak_count_nonzero"] = awkward.count_nonzero


d_datetime_fmt = {}
d_datetime_fmt[0] = "+%Y-%m-%d_%H-%M-%S"
d_datetime_fmt[1] = "+%Y-%m-%d_%H-%M-%N"


def natural_sort(l):
    
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def load_config(cfg) :
    
    content = cfg
    
    if (os.path.isfile(cfg)) :
        
        with open(cfg, "r") as fopen :
            
            content = fopen.read()
    
    print("Loading config:")
    print(content)
    
    d_loadConfig = yaml.load(content, Loader = yaml.FullLoader)
    
    d_loadConfig["content"] = content
    
    d_loadConfig = sortedcontainers.SortedDict(d_loadConfig)
    
    return d_loadConfig


def run_cmd_list(l_cmd) :
    
    for cmd in l_cmd :
        
        retval = os.system(cmd)
        
        if (retval) :
            
            exit()


def getMemoryMB(process = -1) :
    
    if (process < 0) :
        
        process = psutil.Process(os.getpid())
    
    mem = process.memory_info().rss / 1024.0**2
    
    return mem


def get_fileAndTreeNames(in_list, skip = None, filePrefix = "") :
    
    fileAndTreeNames = []
    
    skipFileList = []
    
    if skip :
        
        skipFileList = numpy.loadtxt(skip, dtype = str, delimiter = "*"*100)
    
    for entry in in_list :
        
        for fName in entry.split(",") :
            
            rootFileNames = []
            
            if (".root" in fName) :
                
                filePattern, treeName = fName.strip().split(":")
                rootFileNames = glob.glob(filePattern)
                
                for rootFileName in rootFileNames :
                    
                    if (rootFileName in skipFileList) :
                        
                        print(f"Skipping file: {rootFileName}")
                        continue
                    
                    fileAndTreeNames.append(f"{filePrefix}{rootFileName}:{treeName}")
            
            elif (".txt" in fName) :
                
                sourceFile, treeName = fName.strip().split(":")
                rootFileNames = numpy.loadtxt(sourceFile, dtype = str, delimiter = "*"*100)
                
                for rootFileName in rootFileNames :
                    
                    if (rootFileName in skipFileList) :
                        
                        print(f"Skipping file: {rootFileName}")
                        continue
                    
                    fileAndTreeNames.append(f"{filePrefix}{rootFileName}:{treeName}")
            
            else :
                
                print("Error. Invalid syntax for fileAndTreeNames: %s" %(fName))
                exit(1)
    
    return natural_sort(fileAndTreeNames)


def format_file(filename, d, execute = False) :
    
    l_cmd = []
    
    for key in d :
        
        val = d[key]
        
        l_cmd.append("sed -i \"s#{find}#{repl}#g\" {filename}".format(
            find = key,
            repl = val,
            filename = filename,
        ))
    
    if (execute) :
        
        run_cmd_list(l_cmd)
    
    else :
        
        return l_cmd


def get_timestamp(fmt_opt) :
    
    timestamp = subprocess.check_output(["date", d_datetime_fmt[fmt_opt]]).strip()
    timestamp = timestamp.decode("UTF-8") # Convert from bytes to string
    
    return timestamp


def get_name_withtimestamp(dirname) :
    
    if (os.path.exists(dirname)) :
        
        timestamp = subprocess.check_output(["date", "+%Y-%m-%d_%H-%M-%S", "-r", dirname]).strip()
        timestamp = timestamp.decode("UTF-8") # Convert from bytes to string
        
        dirname_new = "%s_%s" %(dirname, str(timestamp))
        
        return dirname_new
    
    return None


def root_TH1_fixFlowBins(hist) :
    
    nBin = hist.GetNbinsX()
    
    hist.SetBinContent(1, hist.GetBinContent(1)+hist.GetBinContent(0))
    hist.GetBinContent(0, 0)
    
    hist.SetBinContent(nBin, hist.GetBinContent(nBin)+hist.GetBinContent(nBin+1))
    hist.GetBinContent(nBin+1, 0)


#def root_TGraph_to_TH1(graph, setError = True) :
#    
#    hist = graph.GetHistogram().Clone()
#    hist.SetDirectory(0)
#    
#    nPoint = graph.GetN()
#    
#    for iPoint in range(0, nPoint) :
#        
#        if (hasattr(ROOT, "Double")) :
#            
#            pointValX = ROOT.Double(0)
#            pointValY = ROOT.Double(0)
#        
#        else :
#            
#            pointValX = ctypes.c_double(0)
#            pointValY = ctypes.c_double(0)
#        
#        graph.GetPoint(iPoint, pointValX, pointValY)
#        #print(iPoint, pointValX, pointValY)
#        
#        pointErrX = graph.GetErrorX(iPoint)
#        pointErrY = graph.GetErrorY(iPoint)
#        
#        binNum = hist.FindBin(pointValX)
#        
#        hist.SetBinContent(binNum, pointValY)
#        
#        if (setError) :
#            
#            hist.SetBinError(binNum, pointErrY)
#    
#    return hist


def wait_for_asyncpool(l_job) :
    
    l_isJobDone = [False] * len(l_job)
    
    while(False in l_isJobDone) :
        
        for iJob, job in enumerate(l_job) :
            
            if (job is None) :
                
                continue
            
            if (not l_isJobDone[iJob] and job.ready()) :
                
                l_isJobDone[iJob] = True
                
                retVal = job.get()
                
                l_job[iJob] = None
                
                if (not sum(l_isJobDone) % 10) :
                    
                    gc.collect()
    
    gc.collect()


#def load_sampleFilesAndTrees(d_sampleCfg, d_plotCfg) :
#    
#    l_tree_sample = []
#    l_tree_friend = []
#    
#    l_sample_name   = d_sampleCfg["samples"][d_plotCfg["sample"]]["names"]
#    l_sample_weight = d_sampleCfg["samples"][d_plotCfg["sample"]]["weights"]
#    l_sample_norm   = [1.0] * len(l_sample_weight)
#    
#    for iSample, sample_entry in enumerate(l_sample_name) :
#        
#        verbosetag_sample = "[sample %d/%d]" %(iSample+1, len(l_sample_name))
#        
#        sample, tag = sample_entry.split(":")
#        sample_weight = l_sample_weight[iSample]
#        
#        sample_source = d_plotCfg["source"].format(
#            sample = sample,
#            tag = tag,
#        )
#        
#        l_sample_file = numpy.loadtxt(sample_source, dtype = str, delimiter = "x"*100) ##[0: 1]
#        l_sample_filename = [entry.split("/")[-1] for entry in l_sample_file]
#        
#        print("l_sample_file:\n", "\n".join(l_sample_file))
#        
#        tree_sample = ROOT.TChain(d_plotCfg["tree"])
#        
#        for entry in l_sample_file :
#            
#            if (tag == "latest") :
#                
#                tag = entry.split(sample)[-1].split("/")[1]
#            
#            print("%s adding file: %s" %(verbosetag_sample, entry))
#            tree_sample.Add(entry)
#        
#        l_tree_friend.append([])
#        
#        for iFriend, d_friend in enumerate(d_plotCfg["friends"]) :
#            
#            verbosetag_friend = "[friend %d/%d]" %(iFriend+1, len(d_plotCfg["friends"]))
#            
#            sample_fr = sample
#            tag_fr = d_friend["tag"] if (d_friend["tag"] is not None) else tag
#            
#            if ("usefriendlist" in d_friend) :
#                
#                for fr in d_sampleCfg["friendlist"][d_friend["usefriendlist"]] :
#                    
#                    if sample in fr :
#                        
#                        sample_fr, tag_fr = fr.split(":")
#                        break
#            
#            l_sample_friend = [
#                "{dir}/{sample}_{tag}/{entry}".format(
#                    dir = d_friend["dir"],
#                    sample = sample_fr,
#                    tag = tag_fr,
#                    entry = entry,
#                ) for entry in l_sample_filename
#            ]
#            
#            print("l_sample_friend:\n", "\n".join(l_sample_friend))
#            
#            tree_friend = ROOT.TChain(d_friend["tree"])
#            
#            for entry in l_sample_friend :
#                
#                print("%s %s adding file: %s" %(verbosetag_sample, verbosetag_friend, entry))
#                tree_friend.Add(entry)
#            
#            # Need to keep a reference to the tree chain
#            l_tree_friend[-1].append(tree_friend)
#            
#            tree_sample.AddFriend(tree_friend)
#        
#        # Need to keep a reference to the tree chain
#        l_tree_sample.append(tree_sample)
#        
#        l_sample_norm[iSample] = float(sample_weight)/tree_sample.GetEntries()
#    
#    
#    d_result = {}
#    d_result["l_tree_sample"] = l_tree_sample
#    d_result["l_tree_friend"] = l_tree_friend
#    d_result["l_sample_norm"] = l_sample_norm
#    
#    return d_result
#
#
#def root_plot1D(
#    l_hist,
#    outfile,
#    xrange,
#    yrange,
#    l_graph = [],
#    l_line = [],
#    canvassize = (600, 600),
#    logx = False, logy = False,
#    title = "",
#    xtitle = "", ytitle = "",
#    xtitlescale = 1, ytitlescale = 1,
#    centertitlex = True, centertitley = True,
#    centerlabelx = False, centerlabely = False,
#    gridx = False, gridy = False,
#    #ndivisionsx = [5, 5, 0],
#    ndivisionsx = None,
#    forcedivs = False,
#    graphdrawopt = "L",
#    histdrawopt = "hist",
#    stackdrawopt = "nostack",
#    legendpos = "UR",
#    legendncol = 1,
#    legendtextsize = 0.04,
#    legendtitle = "",
#    legendheightscale = 1.0, legendwidthscale = 1.0,
#    lumiText = "(13 TeV)",
#    cmsExtraText = "Simulation Preliminary",
#) :
#    
#    ROOT.gROOT.LoadMacro("utils/tdrstyle.C")
#    ROOT.gROOT.ProcessLine("setTDRStyle()")
#    
#    ROOT.gROOT.SetStyle("tdrStyle")
#    ROOT.gROOT.ForceStyle(True)
#    
#    canvas = ROOT.TCanvas("canvas", "canvas", canvassize[0], canvassize[1])
#    canvas.UseCurrentStyle()
#    
#    canvas.SetLeftMargin(0.16)
#    canvas.SetRightMargin(0.05)
#    canvas.SetTopMargin(0.1)
#    canvas.SetBottomMargin(0.135)
#    
#    
#    legendHeight = legendheightscale * 0.06 * (len(l_hist)+len(l_graph))
#    legendWidth = legendwidthscale * 0.4
#    #legendWidth = legendwidthscale * 0.9
#    
#    padTop = 1 - canvas.GetTopMargin() - 1*ROOT.gStyle.GetTickLength("y")
#    padRight = 1 - canvas.GetRightMargin() - 0.6*ROOT.gStyle.GetTickLength("x")
#    padBottom = canvas.GetBottomMargin() + 0.6*ROOT.gStyle.GetTickLength("y")
#    padLeft = canvas.GetLeftMargin() + 0.6*ROOT.gStyle.GetTickLength("x")
#    
#    if(legendpos == "UR") :
#        
#        legend = ROOT.TLegend(padRight-legendWidth, padTop-legendHeight, padRight, padTop)
#    
#    elif(legendpos == "LR") :
#        
#        legend = ROOT.TLegend(padRight-legendWidth, padBottom, padRight, padBottom+legendHeight)
#    
#    elif(legendpos == "LL") :
#        
#        legend = ROOT.TLegend(padLeft, padBottom, padLeft+legendWidth, padBottom+legendHeight)
#    
#    elif(legendpos == "UL") :
#        
#        legend = ROOT.TLegend(padLeft, padTop-legendHeight, padLeft+legendWidth, padTop)
#    
#    else :
#        
#        print("Wrong legend position option:", legendpos)
#        exit(1)
#    
#    
#    legend.SetHeader(legendtitle, "C")
#    legend.SetNColumns(legendncol)
#    legend.SetFillStyle(0)
#    legend.SetBorderSize(0)
#    legend.SetTextSize(legendtextsize)
#    
#    stack = ROOT.THStack()
#    
#    for hist in l_hist :
#        
#        hist.GetXaxis().SetRangeUser(xrange[0], xrange[1])
#        #hist.SetFillStyle(0)
#        
#        stack.Add(hist, histdrawopt)
#        legend.AddEntry(hist, hist.GetTitle(), "LP")
#    
#    # Add a dummy histogram so that the X-axis range can be beyond the histogram range
#    h1_xRange = ROOT.TH1F("h1_xRange", "h1_xRange", 1, xrange[0], xrange[1])
#    stack.Add(h1_xRange)
#    
#    stack.Draw(stackdrawopt)
#    
#    for gr in l_graph :
#        
#        gr.GetXaxis().SetRangeUser(xrange[0], xrange[1])
#        #hist.SetFillStyle(0)
#        
#        gr.Draw("%s SAME" %(graphdrawopt))
#        legend.AddEntry(gr, gr.GetTitle(), "L")
#    
#    for ln in l_line :
#        
#        ln.Draw("L same")
#    
#    legend.Draw()
#    
#    if (ndivisionsx is not None) :
#        
#        stack.GetXaxis().SetNdivisions(ndivisionsx[0], ndivisionsx[1], ndivisionsx[2], not forcedivs)
#    
#    stack.GetXaxis().SetRangeUser(xrange[0], xrange[1])
#    stack.SetMinimum(yrange[0])
#    stack.SetMaximum(yrange[1])
#    
#    #stack.GetXaxis().SetLabelSize(ROOT.gStyle.GetLabelSize("X") * xLabelSizeScale)
#    #stack.GetYaxis().SetLabelSize(ROOT.gStyle.GetLabelSize("Y") * yLabelSizeScale)
#    
#    stack.GetXaxis().SetTitle(xtitle)
#    #stack.GetXaxis().SetTitleSize(ROOT.gStyle.GetTitleSize("X") * xTitleSizeScale)
#    stack.GetXaxis().SetTitleOffset(ROOT.gStyle.GetTitleOffset("X") * 1.1)
#    
#    stack.GetYaxis().SetTitle(ytitle)
#    #stack.GetYaxis().SetTitleSize(ROOT.gStyle.GetTitleSize("Y") * yTitleSizeScale)
#    stack.GetYaxis().SetTitleOffset(ROOT.gStyle.GetTitleOffset("Y") * 1)
#    
#    #stack.SetTitle(title)
#
#    stack.GetXaxis().CenterTitle(centertitlex)
#    stack.GetYaxis().CenterTitle(centertitley)
#    
#    stack.GetXaxis().CenterLabels(centerlabelx)
#    stack.GetYaxis().CenterLabels(centerlabely)
#    
#    canvas.SetLogx(logx)
#    canvas.SetLogy(logy)
#    
#    canvas.SetGridx(gridx)
#    canvas.SetGridy(gridy)
#    
#    CMS_lumi.CMS_lumi(pad = canvas, iPeriod = 0, iPosX = 0, CMSextraText = cmsExtraText, lumiText = lumiText)
#    
#    if ("/" in outfile) :
#        
#        outdir = outfile
#        outdir = outdir[0: outdir.rfind("/")]
#        
#        os.system("mkdir -p %s" %(outdir))
#    
#    canvas.SaveAs(outfile)
#    canvas.SaveAs(outfile.replace(".pdf", ".png"))
#    
#    canvas.Close()
#    
#    return 0
#


def pseudorapidity_to_polar(eta) :
    
    theta = 2*numpy.arctan(numpy.exp(-eta))
    
    return theta


def pca_2d(x, y, w = None) :
    
    if (w is None) :
        w = numpy.ones(len(x))
    
    mean_x = numpy.average(x, weights = w)
    mean_y = numpy.average(y, weights = w)
    
    cov_xx = numpy.average(w*(x - mean_x)**2, weights = w)
    cov_yy = numpy.average(w*(y - mean_y)**2, weights = w)
    cov_xy = numpy.average(w*(x - mean_x)*(y-mean_y), weights = w)
    
    covmat = numpy.array([
        [cov_xx, cov_xy],
        [cov_xy, cov_yy],
    ])
    
    eigvals, eigvecs = scipy.linalg.eig(covmat)
    
    # Descending
    sortidx = numpy.argsort(eigvals)[::-1]
    
    eigvals = eigvals[sortidx]
    eigvecs = eigvecs[sortidx]
    
    eig1 = eigvecs[0]
    eig2 = eigvecs[1]
    
    slope1 = eig1[1]/eig1[0]
    axis1 = [-slope1, (slope1*mean_x)+mean_y] # [p1, p0] --> y = p1*x + p0
    
    slope2 = eig2[1]/eig2[0]
    axis2 = [-slope2, (slope2*mean_x)+mean_y]
    
    result = {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "eigaxes": [axis1, axis2]
    }
    
    return result


if (__name__ == "__main__") :
    
    exit()

