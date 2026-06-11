#!/usr/bin/env python3

from __future__ import print_function
import tqdm
import argparse
import os
import numpy as np
import sys
import yasp

import heppyy.util.fastjet_cppyy
import heppyy.util.pythia8_cppyy
import heppyy.util.heppyy_cppyy

from cppyy.gbl import fastjet as fj
from cppyy.gbl import Pythia8
from cppyy.gbl.std import vector

from heppyy.util.mputils import logbins
from heppyy.pythia_util import configuration as pyconf

import ROOT
import math
import array
import itertools

from yasp import GenericObject
from alian.sandbox.root_output import SingleRootFile


def iter_all_pairs_with_symmetry(constituents):
	"""Yield unique constituent pairs with a factor reproducing ordered (i, j) counting."""
	n = len(constituents)
	for i, j in itertools.combinations_with_replacement(range(n), 2):
		factor = 1.0 if i == j else 2.0
		yield constituents[i], constituents[j], factor

def main():
	parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly', prog=os.path.basename(__file__))
	pyconf.add_standard_pythia_args(parser)
	parser.add_argument('-v', '--verbose', help="be verbose", default=False, action='store_true')
	parser.add_argument('-o','--output', help='root output filename', default='eec_pPb_escale_test.root', type=str)
	parser.add_argument('--jet-pt-min', help='jet pt min', default=100.0, type=float)
	parser.add_argument('--jet-pt-max', help='jet pt max', default=120.0, type=float)
	parser.add_argument('--etadet', help='detector eta', default=1.0, type=float)
	parser.add_argument('--jet-radius', help='jet radius', default=0.4, type=float)
	parser.add_argument('--scale-shift', help='scale shift for jets', default=0.6, type=float)
	args = parser.parse_args()
	print(args)
 
	pythia = Pythia8.Pythia()

	# jet finder
	# print the banner first
	fj.ClusterSequence.print_banner()
	print()

	scale_shift = args.scale_shift
	args.output = args.output.replace(".root", f"_scale_shift_{scale_shift}.root")

	jet_def = fj.JetDefinition(fj.antikt_algorithm, args.jet_radius)
	jet_selector_0 = fj.SelectorPtRange(args.jet_pt_min, args.jet_pt_max) * fj.SelectorAbsEtaMax(args.etadet - args.jet_radius)
	jet_selector_scale_shift = fj.SelectorPtRange(args.jet_pt_min+scale_shift, args.jet_pt_max+scale_shift) * fj.SelectorAbsEtaMax(args.etadet)
	part_selector = fj.SelectorPtRange(1.0, 1e10)
	fout = SingleRootFile(args.output)
	fout.root_file.cd()

	nbins = 7
	# EEC
	eec_hist = ROOT.TH1F("eec", "EEC", nbins, logbins(0.01, 1.0, nbins))
	eec_hist.GetXaxis().SetTitle("R_{L}")
	eec_hist.GetYaxis().SetTitle("EEC")
	eec_hist.Sumw2()

	eec_hist_scale_shift = ROOT.TH1F("eec_scale_shift", "EEC Scale Shift", nbins, logbins(0.01, 1.0, nbins))
	eec_hist_scale_shift.GetXaxis().SetTitle("R_{L}")
	eec_hist_scale_shift.GetYaxis().SetTitle("EEC")
	eec_hist_scale_shift.Sumw2()

	eec_hist_scale_shift_weight_only = ROOT.TH1F("eec_scale_shift_weight_only", "EEC Scale Shift Weight Only", nbins, logbins(0.01, 1.0, nbins))
	eec_hist_scale_shift_weight_only.GetXaxis().SetTitle("R_{L}")
	eec_hist_scale_shift_weight_only.GetYaxis().SetTitle("EEC")
	eec_hist_scale_shift_weight_only.Sumw2()

	mycfg = []
	pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)
	if not pythia:
		print("[e] pythia initialization failed.")
		return
	if args.nev < 10:
		args.nev = 10

	count_jets = 0
	count_jets_scale_shift = 0
 
	pbar = tqdm.tqdm(total=args.nev)
	while pbar.n < args.nev:
		if not pythia.next():
			print('[w] pythia event generation failed, skipping event.')
			continue
		parts = vector[fj.PseudoJet]([fj.PseudoJet(p.px(), p.py(), p.pz(), p.e()) for p in pythia.event if p.isFinal() and p.isCharged()])
		jets_all = jet_def(parts)
		jets = jet_selector_0(jets_all)
		if len(jets) == 0:
			continue
		for jet in jets:
			count_jets += 1
			# compute EEC for this jet
			constituents = list(part_selector(jet.constituents()))
			jet_pt = jet.perp()
			den_nom = jet_pt * jet_pt
			den_shift = (jet_pt + scale_shift) * (jet_pt + scale_shift)
			for p1, p2, factor in iter_all_pairs_with_symmetry(constituents):
				dR = p1.delta_R(p2)
				eec = factor * (p1.perp() * p2.perp()) / den_nom
				eec_hist.Fill(dR, eec)
				eec_reweight = factor * (p1.perp() * p2.perp()) / den_shift
				eec_hist_scale_shift_weight_only.Fill(dR, eec_reweight)

		jets = jet_selector_scale_shift(jets_all)
		if len(jets) == 0:
			continue
		for jet in jets:
			count_jets_scale_shift += 1
			# compute EEC for this jet
			constituents = list(part_selector(jet.constituents()))
			den_nom = jet.perp() * jet.perp()
			for p1, p2, factor in iter_all_pairs_with_symmetry(constituents):
				dR = p1.delta_R(p2)
				# eec = factor * (p1.perp() * p2.perp()) / ((jet.perp() + scale_shift) * (jet.perp() + scale_shift))
				eec = factor * (p1.perp() * p2.perp()) / den_nom
				eec_hist_scale_shift.Fill(dR, eec)
     
		pbar.update(1)
   
	print(f"Total jets: {count_jets}")
	eec_hist.Scale(1.0 / count_jets)
	eec_hist_scale_shift.Scale(1.0 / count_jets_scale_shift)
	eec_hist_scale_shift_weight_only.Scale(1.0 / count_jets)
	fout.write()
	fout.close()
 
if __name__ == "__main__":	main()