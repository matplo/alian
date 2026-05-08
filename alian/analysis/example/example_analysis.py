#!/usr/bin/env python3

"""Example script on using the alian framework for Run 3 analysis.

This script is used on the ROOT file (provided as an argument) to
perform the analysis of the events containing isolated photons and
jets. The analysis is configured with a config YAML file, creates
and fills histograms for the cleaned data, and stores them in the
output file.

To be used with alian/config/example/example_analysis.yaml
"""

import argparse
import itertools

import numpy as np
from alian.analysis.base import AnalysisBase, add_default_args, delta_R

import heppyy

fj = heppyy.load_cppyy('fastjet')


class AnalysisExample(AnalysisBase):
    _defaults = {
        'photon_jet_angle_min': 0.875 * np.pi,
        'pt_min_eec': 1.0,
    }
    def init_analysis(self, analysis_cfg: dict):
        config = self._defaults | analysis_cfg
        for setting, value in config.items():
            setattr(self, setting, value)
        self.eec_trk_selector = fj.SelectorPtMin(self.pt_min_eec)

    def analyze_event(self):
        # Analyzes this event that has passed the selection criteria
        # self.event contains the selected event
        # self.tracks contains selected tracks (i.e. after selection cuts)
        # self.clusters contains selected clusters (i.e. after selection cuts)
        # self.jets contains selected jets

        [self.hists['track_pT'].Fill(t.pt()) for t in self.tracks]
        [self.hists['jet_pT'].Fill(j.pt()) for j in self.jets]
        [self.hists['jet_pT_coarse'].Fill(j.pt()) for j in self.jets]

        for c in self.clusters:
            is_iso, iso_pt, _, _ = self.selector.isolation.selects(c, self.tracks, verbose = True)
            self.hists['tot_iso_pT'].Fill(iso_pt)
            if is_iso:
                self.hists['photon_E'].Fill(c.e())
                # find back-to-back jets
                for j in self.jets:
                    dphi = np.abs(j.delta_phi_to(c))
                    if dphi > self.photon_jet_angle_min:
                        self.hists["jet_photon_dphi"].Fill(dphi)
                        self.hists["y_jet_pT"].Fill(j.pt())
                        self.hists["xjy"].Fill(j.pt() / c.e(), c.e())
                        self.do_eec(j)

    def do_eec(self, jet):
        tracks = self.eec_trk_selector(jet.constituents())
        for p1, p2 in itertools.permutations(tracks, 2):
            ew = p1.pt() * p2.pt() / jet.pt() / jet.pt()
            angle = delta_R(p1, p2)

            self.hists["eec"].Fill(jet.pt(), angle, ew)

    def finalize(self):
        self.hists['track_pT'].Scale(1, "width")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run analysis on ROOT file using YAML configuration.")
    parser = add_default_args(parser)

    args = parser.parse_args()

    ana = AnalysisExample(args.input_file, args.output_file, args.config_file, args.tree_struct, args.nev, args.lhc_run)
    ana.run()
