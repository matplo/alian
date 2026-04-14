#!/usr/bin/env python3

"""Example script on using RDataFrame for analysis.

An example script that uses RDataFrame for analysis. RDF
implementations can only handle basic selections on cluster
and track quantities; for more complicated tasks like jet
finding or cluster-track matching, use example_analysis.py.

To be used with alian/config/example/example_rdf.yaml
"""

import argparse
from datetime import datetime as dt
from time import perf_counter as timenow

import numpy as np
import ROOT
from alian.analysis.base import AnalysisSelector, linbins, logbins, set_up_logger

ROOT.TH1.SetDefaultSumw2() # applies to TH2 and TH3 as well

class AnalysisQA:
    def __init__(self, inf, outf, cfg):
        self.logger = set_up_logger(__name__)
        self.outf = outf
        self.selector = AnalysisSelector.load(cfg)
        self.df = ROOT.RDataFrame("eventTree", inf)
        self.histos = {}

    def note_time(self, msg):
        self.logger.info(f"{msg}: -------- {timenow() - self.start_time:.3f} sec. --------", stacklevel = 2)
    def note_start(self, msg):
        self.start_time = timenow()
        self.logger.info(f"{msg}: {dt.now().replace(microsecond=0)}", stacklevel = 2)

    def analyze(self):
        self.note_start("Starting analysis")
        self.apply_cuts()
        self.create_histos()
        self.fill_and_save()
        self.note_time("Finished analysis")

    def apply_cuts(self):
        """
        Apply cuts to events, tracks, and clusters. The event selection
        is just a Filter applied on the RDF. Cluster and track selections
        are a little more complicated since they're vector columns, so
        the implementation is hidden behind apply_to(). This method
        creates a new set of columns for tracks and clusters (appended
        with `_selected`) with the relevant cuts applied to them. Use
        these column names to fill histograms from clusters or tracks
        with the cuts applied, or use the default track_pt, track_phi; etc.
        """
        self.df = self.selector.event.apply_to(self.df)
        # cluster pT isn't a column that exists already in the BerkeleyTree so we can create it:
        # self.df = self.df.Define("cluster_pt", "cluster_energy / cosh(cluster_eta)")
        # Applying the cluster selection will also automatically add it if it's not already defined:
        self.df = self.selector.cluster.apply_to(self.df)
        self.df = self.selector.track.apply_to(self.df)
        self.note_time("Cuts applied")

    def create_histos(self):
        """
        Define your histograms here. For Histo*D, the first argument is
        a tuple corresponding to what you would use for a normal TH*.
        The second is the column that your histogram will be Filled with.
        Note that the histograms themselves are not filled until
        fill_and_save() is called.
        """
        self.histos['track_pt'] = self.df.Histo1D(
            ("track_pt", 'Track #it{p}_{T};track #it{p}_{T} (GeV);Counts', 300, logbins(0.15, 200, 300)),
            "track_pt_selected"
        )
        self.histos['track_eta'] = self.df.Histo1D(
            ("track_eta", 'Track #eta;track #eta;Counts', 200, -0.9, 0.9),
            "track_eta_selected"
        )
        self.histos['track_phi'] = self.df.Histo1D(
            ("track_phi", 'Track #phi;track #phi;Counts', 200, 0, 2*np.pi),
            "track_phi_selected"
        )
        self.histos['track_eta_phi'] = self.df.Histo2D(
            ("track_eta_phi", 'Track #eta-#phi distribution;#eta;#phi', 200, -0.9, 0.9, 200, 0, 2*np.pi),
            "track_eta_selected", "track_phi_selected"
        )
        pt_bins = np.array([0.15, 0.5, 1, 5, 10, 15, 20, 30, 40, 50, 60, 100, 200])
        self.histos['track_eta_phi_pt'] = self.df.Histo3D(
            ("track_eta_phi_pT", 'Track #eta-#phi-#it{p}_{T} distribution;#eta;#phi;#it{p}_{T}',
            100, linbins(-0.9, 0.9, 100),
            100, linbins(0, 2*np.pi, 100),
            len(pt_bins) - 1, pt_bins),
            "track_eta_selected", "track_phi_selected", "track_pt_selected"
        )

        # these histograms will be filled with all clusters, not just those that pass selections,
        # since we are not using the columns with `_selected` appended
        self.histos['cluster_E'] = self.df.Histo1D(
            ("cluster_E", 'Cluster energy distribution;#it{E} (GeV);Counts', 200, 0, 100),
            "cluster_energy"
        )
        self.histos['cluster_pT'] = self.df.Histo1D(
            ("cluster_pT", 'Cluster #it{p}_{T} distribution;#it{p}_{T} (GeV/#it{c});Counts', 200, 0, 100),
            "cluster_pt"
        )
        self.histos['cluster_m02'] = self.df.Histo1D(
            ("cluster_m02", 'Cluster shape;#it{M}_{02};Counts', 400, 0, 2),
            "cluster_m02"
        )
        self.histos['cluster_time'] = self.df.Histo1D(
            ("cluster_time", 'Cluster time;time (ns);Counts', 120, -30, 30),
            "cluster_time"
        )
        self.histos['cluster_ncells'] = self.df.Histo1D(
            ("cluster_ncells", 'Cluster cell multiplicity;n_{cells};Counts', 25, -0.5, 24.5),
            "cluster_ncells"
        )

        self.note_time("Histograms defined")

    def fill_and_save(self):
        """
        Fill and save all defined histograms. Note that, for histograms
        defined in RDFs, filling is triggered when Write() is called.
        """
        self.note_time("Filling and saving histograms...")
        with ROOT.TFile(self.outf, 'recreate'):
            for h in self.histos.values():
                h.Write()
        self.note_time("Histograms saved")
        self.logger.info(f"Output saved to: {self.outf}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input",  type = str, required = True, help = "Input file or file containing a list of input files.")
    parser.add_argument("-o", "--output", type = str, required = True, help = "File to write the analysis.")
    parser.add_argument("-c", "--config", type = str, required = True, help = "YAML file describing the cuts to be applied to the data.")
    args = parser.parse_args()

    ROOT.EnableImplicitMT()
    analysis = AnalysisQA(args.input, args.output, args.config)
    analysis.analyze()