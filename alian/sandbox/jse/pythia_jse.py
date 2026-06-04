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

from yasp import GenericObject
from alian.sandbox.root_output import SingleRootFile

# we measured for k=1 and a=1,2,3
def angularity(jet, a, k, jetR):
	ang = 0.0
	for p in jet.constituents():
		dr = jet.delta_R(p) / jetR
		pt = p.perp() / jet.perp()
		ang += ((dr)**a) * ((pt)**k)
	return ang

def mass(jet):
	m2 = jet.e()**2 - jet.px()**2 - jet.py()**2 - jet.pz()**2
	if m2 > 0:
		return math.sqrt(m2)
	return 0.0


def main():
	parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly', prog=os.path.basename(__file__))
	pyconf.add_standard_pythia_args(parser)
	parser.add_argument('-v', '--verbose', help="be verbose", default=False, action='store_true')
	parser.add_argument('--ncorrel', help='max n correlator', type=int, default=2)
	parser.add_argument('-o','--output', help='root output filename', default='pythia_jse_output.root', type=str)
	parser.add_argument('--jet-pt-min', help='jet pt min', default=100.0, type=float)
	parser.add_argument('--jet-pt-max', help='jet pt max', default=120.0, type=float)
	parser.add_argument('--etadet', help='detector eta', default=2.5, type=float)
	parser.add_argument('--shape', help='fill the jet shape histograms', action='store_true', default=False)
	args = parser.parse_args()

	pythia = Pythia8.Pythia()

	# jet finder
	# print the banner first
	fj.ClusterSequence.print_banner()
	print()

	Rs = [0.2, 0.4, 0.6]
	jet_defs = {}
	jet_selectors = {}
	for R in Rs:
		jet_defs[R] = fj.JetDefinition(fj.antikt_algorithm, R)
		jet_selectors[R] = fj.SelectorPtMin(args.jet_pt_min) * fj.SelectorAbsEtaMax(args.etadet - R * 1.05)
		if args.jet_pt_max > 0:
			jet_selectors[R] *= fj.SelectorPtMax(args.jet_pt_max) * fj.SelectorPtMin(args.jet_pt_min) * fj.SelectorAbsEtaMax(args.etadet - R * 1.05)
	
	jet_def_wta = fj.JetDefinition(fj.cambridge_algorithm, 1.0)
	jet_def_wta.set_recombination_scheme(fj.WTA_pt_scheme)
	reclusterer_wta =  fj.contrib.Recluster(jet_def_wta)

	sd01 = fj.contrib.SoftDrop(0, 0.1, 1.0)
	sd02 = fj.contrib.SoftDrop(0, 0.2, 1.0)
 
	fout = SingleRootFile(args.output)
	fout.root_file.cd()
	tnR02 = ROOT.TNtuple("tnR02", "tnR02", "pt:eta:phi:mass:wtastd:wtasd01:wtasd02:angk1a1:angk1a2:angk1a3:mjet")
	tnR04 = ROOT.TNtuple("tnR04", "tnR04", "pt:eta:phi:mass:wtastd:wtasd01:wtasd02:angk1a1:angk1a2:angk1a3:mjet")
	tnR06 = ROOT.TNtuple("tnR06", "tnR06", "pt:eta:phi:mass:wtastd:wtasd01:wtasd02:angk1a1:angk1a2:angk1a3:mjet")
	tn = {0.2: tnR02, 0.4: tnR04, 0.6: tnR06}
	for r, tx in tn.items():
		fout.root_file.cd()
		fout.add(tx)

	shape_hists = {}
	shape_hists_uw = {}
	shape_hists_dydphi_low_uw = {}
	shape_hists_dydphi_high_uw = {}
	low_high_cut = {}
	low_high_cut['angk1a1'] = 0.25
	low_high_cut['angk1a2'] = 0.15
	low_high_cut['angk1a3'] = 0.08
	low_high_cut['mjet'] = 3.0
	if args.shape:
		for R in Rs:
			shape_hists[R] = {}
			shape_hists_uw[R] = {}
			shape_hists_dydphi_low_uw[R] = {}
			shape_hists_dydphi_high_uw[R] = {}
			for angs in ['angk1a1', 'angk1a2', 'angk1a3', 'mjet']:
				ymin = 0.0
				ymax = 1.0
				if angs == 'mjet':
					ymin = 0.0
					ymax = 100.0
				fout.root_file.cd()

				shape_hists[R][angs] = ROOT.TH2F(f"shape_{R}_{angs}", f"shape_{R}_{angs}", 100, 0, 1.0, 100, ymin, ymax)
				shape_hists[R][angs].SetDirectory(0)
				fout.add(shape_hists[R][angs])

				shape_hists_uw[R][angs] = ROOT.TH2F(f"shape_{R}_{angs}_uw", f"shape_{R}_{angs}", 100, 0, 1.0, 100, ymin, ymax)
				shape_hists_uw[R][angs].SetDirectory(0)
				fout.add(shape_hists_uw[R][angs])

				shape_hists_dydphi_low_uw[R][angs] = ROOT.TH2F(f"shape_{R}_{angs}_dydphi_low_uw", f"shape_{R}_{angs}", 41, -R-0.1, R+0.1, 41, -R-0.1, R+0.1)
				shape_hists_dydphi_low_uw[R][angs].SetDirectory(0)
				fout.add(shape_hists_dydphi_low_uw[R][angs])

				shape_hists_dydphi_high_uw[R][angs] = ROOT.TH2F(f"shape_{R}_{angs}_dydphi_high_uw", f"shape_{R}_{angs}", 41, -R-0.1, R+0.1, 41, -R-0.1, R+0.1)
				shape_hists_dydphi_high_uw[R][angs].SetDirectory(0)
				fout.add(shape_hists_dydphi_high_uw[R][angs])

 
	print(shape_hists)

	mycfg = []
	pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)
	if not pythia:
		print("[e] pythia initialization failed.")
		return
	if args.nev < 10:
		args.nev = 10
	count_jets = 0
	pbar = tqdm.tqdm(total=args.nev)
	while pbar.n < args.nev:
		if not pythia.next():
			continue
		parts = vector[fj.PseudoJet]([fj.PseudoJet(p.px(), p.py(), p.pz(), p.e()) for p in pythia.event if p.isFinal() and p.isCharged()])
		# parts = pythiafjext.vectorize(pythia, True, -1, 1, False)
		for R in Rs:
			jet_def = jet_defs[R]
			jet_selector = jet_selectors[R]
			jets = jet_selector(jet_def(parts))
			if R == 0.4 and len(jets) > 0:
				pbar.update(1)
			for j in jets:
				count_jets += 1
				jet_wta = reclusterer_wta.result(j)
				jet_sd01 = sd01.result(j)
				jet_sd02 = sd02.result(j)
				wtastd = jet_wta.delta_R(j)
				wtasd01 = jet_sd01.delta_R(j)
				wtasd02 = jet_sd02.delta_R(j)
				angk1a1 = angularity(j, 1.0, 1.0, R)
				angk1a2 = angularity(j, 2.0, 1.0, R)
				angk1a3 = angularity(j, 3.0, 1.0, R)
				mjet = mass(j)
				tn[R].Fill(j.perp(), j.eta(), j.phi(), j.m(), wtastd, wtasd01, wtasd02, angk1a1, angk1a2, angk1a3, mjet)
				if args.shape:
					for c in j.constituents():
						dR = c.delta_R(j)
						z = c.perp() / j.perp()
						for angs in ['angk1a1', 'angk1a2', 'angk1a3', 'mjet']:
							shape_hists[R][angs].Fill(dR, eval(angs), z)
							shape_hists_uw[R][angs].Fill(dR, eval(angs), 1.0)
							if eval(angs) < low_high_cut[angs]:
								shape_hists_dydphi_low_uw[R][angs].Fill(c.rap() - j.rap(), c.phi() - j.phi(), 1.0)
							else:
								shape_hists_dydphi_high_uw[R][angs].Fill(c.rap() - j.rap(), c.phi() - j.phi(), 1.0)

	pythia.stat()

	print(type(pythia))
	if args.shape:
		for R in Rs:
			for angs in ['angk1a1', 'angk1a2', 'angk1a3', 'mjet']:
				shape_hists[R][angs].Scale(1.0 / count_jets)
				shape_hists_uw[R][angs].Scale(1.0 / count_jets)
				shape_hists_dydphi_low_uw[R][angs].Scale(1.0 / count_jets)
				shape_hists_dydphi_high_uw[R][angs].Scale(1.0 / count_jets)
	fout.close()

if __name__ == '__main__':
	main()