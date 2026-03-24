#!/usr/bin/env python3
"""
lund_eec.py — Primary Lund plane + subjet-resolved E2C analysis.

Input modes
-----------
  Pythia8  (default, standard pyconf args: --nev, --py-pthatmin, cmnd file, etc.)
  File     (--input file.root [file2.root ...] or glob "jewel_*.root"; enables file mode)
            --max-events N  : max events per file (not total)

Jet selection
-------------
  --jetR         jet radius (default 0.4)
  --jet-pt-min / --jet-pt-max   (default 100 / 120 GeV)
  --etadet       detector eta (default 2.5)

Splitting selections → E2C
--------------------------
  --kt-cuts 0,1,2,5       threshold on kT [GeV]; all splittings passing kt > threshold
  --kappa-cuts 0,...       paired kappa threshold (matched to --kt-cuts, padded with 0)
  --max-kt-eec             single splitting: highest kT per jet
  --soft-drop-eec          single splitting: first z > --z-sd (default 0.1)
  --symmetric-eec          all splittings with z > --z-sym (default 0.3)

For each selected splitting, four E2C distributions are filled (binned in ln ΔR):
  eec_AA  : pairs both within harder subjet A
  eec_BB  : pairs both within softer subjet B
  eec_AB  : cross pairs — one from A, one from B
  eec_all : all pairs from A ∪ B (AA + BB + AB, each unordered pair once)
  weight  : pT_i * pT_j / pT_jet²,  normalised per jet

Output files  (<stem> = --output without .parquet extension)
--------------------------------------------------------------
  <stem>_eec.parquet            EEC histograms, all selections, tidy DataFrame
                                columns: ln_dR, bin_lo, bin_hi, eec_AA, eec_BB,
                                         eec_AB, eec_all, n_jets, n_splits,
                                         label, kt_cut, kappa_cut, selection,
                                         jet_pt_min, jet_pt_max
  <stem>_lundplane.npy          2-D density Lund plane H[ix,iy] (weight=1/splitting)
  <stem>_lundplane_xedges.npy   x bin edges: ln(1/Δ)
  <stem>_lundplane_yedges.npy   y bin edges: ln(kt)
  <stem>_lundplane_weighted.npy 2-D energy-weighted Lund plane (weight=pT_A*pT_B/pT_jet²)

Parquet selection column values
--------------------------------
  'threshold'            kt/kappa threshold-based (one row-group per kt_cut value)
  'max_kt'               max-kT splitting
  'soft_drop_z>{z_sd}'   soft-drop selected splitting
  'symmetric_z>{z_sym}'  symmetric splittings (all z > z_sym)
"""

from __future__ import print_function
import os
import math
import argparse
import numpy as np
import tqdm
import pandas as pd

import heppyy.util.fastjet_cppyy
import heppyy.util.heppyy_cppyy

from cppyy.gbl import fastjet as fj
from cppyy.gbl.std import vector


# ── E2C accumulator ──────────────────────────────────────────────────────────

class EECAccum:
	"""
	Accumulate weighted E2C pairs into ln(ΔR) histograms.

	Convention: each unordered pair (i,j) with i<j is counted once,
	weighted by pT_i * pT_j / pT_jet^2.
	"""

	def __init__(self, nbins=30, lndr_min=-5.0, lndr_max=1.5):
		self.edges    = np.linspace(lndr_min, lndr_max, nbins + 1)
		self.nbins    = nbins
		self.h        = {t: np.zeros(nbins) for t in ('AA', 'BB', 'AB', 'all')}
		self.n_jets   = 0
		self.n_splits = 0
		self.sum_kt   = 0.0

	def fill(self, parts_a, parts_b, jet_pt2):
		"""
		Fill all four E2C types for one Lund splitting.

		Parameters
		----------
		parts_a  : list of fj.PseudoJet  (harder subjet constituents)
		parts_b  : list of fj.PseudoJet  (softer subjet constituents)
		jet_pt2  : float  (jet pT² for normalisation)
		"""
		def _add(p1, p2, key):
			dR = p1.delta_R(p2)
			if dR <= 0:
				return
			idx = np.searchsorted(self.edges, math.log(dR), side='right') - 1
			if 0 <= idx < self.nbins:
				w = p1.pt() * p2.pt() / jet_pt2
				self.h[key][idx]   += w
				self.h['all'][idx] += w

		# AA: pairs within harder subjet
		for i in range(len(parts_a)):
			for j in range(i + 1, len(parts_a)):
				_add(parts_a[i], parts_a[j], 'AA')

		# BB: pairs within softer subjet
		for i in range(len(parts_b)):
			for j in range(i + 1, len(parts_b)):
				_add(parts_b[i], parts_b[j], 'BB')

		# AB: cross pairs
		for pa in parts_a:
			for pb in parts_b:
				_add(pa, pb, 'AB')

		self.n_splits += 1

	def normalise(self):
		if self.n_jets > 0:
			for k in self.h:
				self.h[k] = self.h[k] / self.n_jets

	def to_df(self, label=None, kt_cut=None):
		centers = 0.5 * (self.edges[:-1] + self.edges[1:])
		avg_kt  = self.sum_kt / self.n_splits if self.n_splits > 0 else float('nan')
		d = {
			'ln_dR':    centers,
			'bin_lo':   self.edges[:-1],
			'bin_hi':   self.edges[1:],
			'eec_AA':   self.h['AA'],
			'eec_BB':   self.h['BB'],
			'eec_AB':   self.h['AB'],
			'eec_all':  self.h['all'],
			'n_jets':   self.n_jets,
			'n_splits': self.n_splits,
			'avg_kt':   avg_kt,
		}
		if label   is not None: d['label']  = label
		if kt_cut  is not None: d['kt_cut'] = kt_cut
		return pd.DataFrame(d)


# ── Lund-plane accumulator ────────────────────────────────────────────────────

class LundPlaneAccum:
	"""2-D histogram of ln(1/Δ) vs ln(kt) for the primary Lund plane."""

	def __init__(self, nbins=50, lninvd_range=(0, 5), lnkt_range=(-4, 6)):
		self.xedges = np.linspace(*lninvd_range, nbins + 1)  # ln(1/Δ)
		self.yedges = np.linspace(*lnkt_range,   nbins + 1)  # ln(kt)
		self.h2     = np.zeros((nbins, nbins))

	def fill(self, lund_seq, jet_pt2=None, local_weight=False):
		"""
		Fill the histogram.

		jet_pt2=None, local_weight=False  → weight = 1 per splitting (density)
		jet_pt2 provided                  → weight = pT_A * pT_B / pT_jet²
		local_weight=True                 → weight = pT_A * pT_B / pT_pair²  (z*(1-z) local)
		"""
		for l in lund_seq:
			if l.Delta() <= 0 or l.kt() <= 0:
				continue
			if local_weight:
				pair_pt2 = l.pair().pt() ** 2
				w = l.harder().pt() * l.softer().pt() / pair_pt2 if pair_pt2 > 0 else 0.0
			elif jet_pt2 is not None:
				w = l.harder().pt() * l.softer().pt() / jet_pt2
			else:
				w = 1.0
			xi = np.searchsorted(self.xedges, math.log(1.0 / l.Delta()), side='right') - 1
			yi = np.searchsorted(self.yedges, math.log(l.kt()),           side='right') - 1
			if 0 <= xi < self.h2.shape[0] and 0 <= yi < self.h2.shape[1]:
				self.h2[xi, yi] += w


# ── single-splitting selectors ────────────────────────────────────────────────

def select_max_kt(lund_seq):
	"""Return the primary Lund splitting with the largest kT, or None."""
	best = None
	for l in lund_seq:
		if best is None or l.kt() > best.kt():
			best = l
	return best


def select_soft_drop(lund_seq, z_cut):
	"""
	Return the first primary Lund splitting satisfying z > z_cut (soft-drop
	condition with beta=0), or None if no splitting passes.
	"""
	for l in lund_seq:
		if l.z() > z_cut:
			return l
	return None


def select_symmetric_splittings(lund_seq, z_min):
	"""
	Return all primary Lund splittings with z > z_min (symmetric selection).

	z is the softer-branch momentum fraction (range 0–0.5); larger z means
	a more equal pT sharing between the two subjets.  Typical choices:
	  z_min = 0.2  — moderately symmetric
	  z_min = 0.3  — fairly symmetric
	  z_min = 0.4  — nearly equal pT split
	"""
	return [l for l in lund_seq if l.z() > z_min]


# ── per-jet analysis ──────────────────────────────────────────────────────────

def _fill_splitting(l, accum, jet_pt2, pt_min=0.0):
	"""Fill accum from a single LundDeclustering object.
	Only particles with pT > pt_min are included in the EEC pairs.
	"""
	try:
		parts_a = [p for p in l.harder().constituents() if p.pt() > pt_min]
		parts_b = [p for p in l.softer().constituents() if p.pt() > pt_min]
	except Exception:
		return
	if not parts_a and not parts_b:
		return
	accum.fill(parts_a, parts_b, jet_pt2)
	accum.sum_kt += l.kt()


def process_jet(jet, lund_gen, eec_accums, lund_accum, kt_cuts, kappa_cuts,
                eec_maxkt=None, eec_maxkt_thr=None,
                eec_sd_dict=None, lund_accum_w=None,
                eec_sym=None, z_sym=0.3, z_sym_kt_cut=0.0,
                lund_accum_lw=None, pt_particle_min=0.0):
	"""
	Run the Lund + E2C analysis for a single jet.

	Parameters
	----------
	jet              : fj.PseudoJet
	lund_gen         : fj.contrib.LundGenerator
	eec_accums       : dict { (kt_cut, kappa_cut) -> EECAccum }  threshold-based
	lund_accum       : LundPlaneAccum   density (weight=1 per splitting)
	kt_cuts          : list of float
	kappa_cuts       : list of float
	eec_maxkt        : EECAccum or None   filled with the max-kT splitting
	eec_maxkt_thr    : dict { (kt_cut, kappa_cut) -> EECAccum } or None
	                   max-kT splitting that also passes each kt/kappa threshold
	eec_sd_dict      : dict { z_sd -> EECAccum } or None  soft-drop per z_sd value
	lund_accum_w     : LundPlaneAccum or None  energy-weighted Lund plane (pT_A*pT_B/pT_jet²)
	eec_sym          : EECAccum or None   filled with splittings where z > z_sym
	z_sym            : float              symmetry z threshold (default 0.3)
	z_sym_kt_cut     : float              additional kT cut for symmetric splittings (default 0.0)
	lund_accum_lw    : LundPlaneAccum or None  local-weight Lund plane (pT_A*pT_B/pT_pair²)
	pt_particle_min  : float              only particles with pT > this enter EEC pairs (default 0)
	"""
	lund_seq = lund_gen.result(jet)
	jet_pt2  = jet.pt() ** 2
	lund_accum.fill(lund_seq)
	if lund_accum_w is not None:
		lund_accum_w.fill(lund_seq, jet_pt2=jet_pt2)
	if lund_accum_lw is not None:
		lund_accum_lw.fill(lund_seq, local_weight=True)

	# ── threshold-based (all splittings passing cuts) ──────────────────────
	for (kt_cut, kappa_cut), accum in eec_accums.items():
		accum.n_jets += 1
		for l in lund_seq:
			if l.kt()    < kt_cut:    continue
			if l.kappa() < kappa_cut: continue
			_fill_splitting(l, accum, jet_pt2, pt_particle_min)

	# ── max-kT splitting ───────────────────────────────────────────────────
	if eec_maxkt is not None or eec_maxkt_thr is not None:
		l = select_max_kt(lund_seq)
		if eec_maxkt is not None:
			eec_maxkt.n_jets += 1
			if l is not None:
				_fill_splitting(l, eec_maxkt, jet_pt2, pt_particle_min)
		if eec_maxkt_thr is not None:
			for (kt_cut, kappa_cut), accum in eec_maxkt_thr.items():
				accum.n_jets += 1
				if l is not None and l.kt() >= kt_cut and l.kappa() >= kappa_cut:
					_fill_splitting(l, accum, jet_pt2, pt_particle_min)

	# ── soft-drop splittings (one accumulator per z_sd) ────────────────────
	if eec_sd_dict is not None:
		for z_sd, accum in eec_sd_dict.items():
			accum.n_jets += 1
			l = select_soft_drop(lund_seq, z_sd)
			if l is not None:
				_fill_splitting(l, accum, jet_pt2, pt_particle_min)

	# ── symmetric splittings (z > z_sym, optionally kT > z_sym_kt_cut) ────
	if eec_sym is not None:
		eec_sym.n_jets += 1
		for l in select_symmetric_splittings(lund_seq, z_sym):
			if l.kt() < z_sym_kt_cut:
				continue
			_fill_splitting(l, eec_sym, jet_pt2, pt_particle_min)


# ── event sources ─────────────────────────────────────────────────────────────

def pythia_events(args):
	"""Yield vector<PseudoJet> per event from Pythia8."""
	import heppyy.util.pythia8_cppyy
	from cppyy.gbl import Pythia8
	from heppyy.pythia_util import configuration as pyconf

	pythia = pyconf.create_and_init_pythia_from_args(args, [])
	if not pythia:
		raise RuntimeError('[e] Pythia initialisation failed.')

	pbar = tqdm.tqdm(total=args.nev)
	while pbar.n < args.nev:
		if not pythia.next():
			continue
		yield vector[fj.PseudoJet]([
			fj.PseudoJet(p.px(), p.py(), p.pz(), p.e())
			for p in pythia.event if p.isFinal() and p.isVisible()
			# for p in pythia.event if p.isFinal() and p.isCharged()
		])
		pbar.update(1)
	pythia.stat()


def file_events(file_paths, max_events=None):
	"""
	Yield vector<PseudoJet> per event from one or more JEWEL ROOT files.

	file_paths : list of str  — explicit paths or glob patterns.
	max_events : int or None  — max events *per file* (None = all).
	"""
	import uproot, glob as _glob

	# expand any glob patterns and sort for reproducibility
	expanded = []
	for p in file_paths:
		matches = sorted(_glob.glob(p))
		expanded.extend(matches if matches else [p])

	if not expanded:
		raise FileNotFoundError(f'No files matched: {file_paths}')

	print(f'[i] processing {len(expanded)} file(s):')
	for p in expanded:
		print(f'    {p}')

	for file_path in expanded:
		print(f'[i] reading {file_path}')
		with uproot.open(file_path) as f:
			tracks_df     = f['tracks'].arrays(library='pd')
			event_info_df = f['event_info'].arrays(library='pd')
		events_df = pd.merge(tracks_df, event_info_df, on='eventID', how='inner')

		for n, (_, ev) in enumerate(tqdm.tqdm(events_df.groupby('eventID'),
		                                       desc=os.path.basename(file_path))):
			if max_events is not None and n >= max_events:
				break
			yield vector[fj.PseudoJet]([
				fj.PseudoJet(px, py, pz, e)
				for px, py, pz, e in zip(ev['px'], ev['py'], ev['pz'], ev['energy'])
			])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
	parser = argparse.ArgumentParser(
		description='Lund plane + subjet E2C analysis',
		prog=os.path.basename(__file__))

	try:
		from heppyy.pythia_util import configuration as pyconf
		pyconf.add_standard_pythia_args(parser)
	except Exception:
		pass

	parser.add_argument('--input',       default=None, nargs='+',
	                    help='JEWEL ROOT file(s) or glob pattern(s); enables file mode')
	parser.add_argument('--max-events',  default=None,      type=int,
	                    help='Max events per file (applied independently to each input file)')
	parser.add_argument('--output',      default='lund_eec_output.parquet', type=str)
	parser.add_argument('--jetR',        default=0.4,       type=float)
	parser.add_argument('--jet-pt-min',  default=100.0,     type=float)
	parser.add_argument('--jet-pt-max',  default=120.0,     type=float)
	parser.add_argument('--etadet',      default=2.5,       type=float)
	parser.add_argument('--kt-cuts',     default='0,1,2,5', type=str,
	                    help='Comma-separated kT minima [GeV] (applied to each Lund splitting)')
	parser.add_argument('--kappa-cuts',  default='0',       type=str,
	                    help='Comma-separated kappa minima (matched to --kt-cuts, padded with 0)')
	parser.add_argument('--label',       default='',        type=str)
	parser.add_argument('--lund-local-weight', action='store_true', default=False,
	                    help='Also compute local-weight Lund plane (pT_A*pT_B/pT_pair² per splitting)')
	parser.add_argument('--max-kt-eec',  action='store_true', default=False,
	                    help='Also compute E2C for the max-kT primary splitting per jet')
	parser.add_argument('--soft-drop-eec', action='store_true', default=False,
	                    help='Also compute E2C for the soft-drop selected splitting per jet')
	parser.add_argument('--z-sd',          default='0.1,0.2,0.3',     type=str,
	                    help='Comma-separated soft-drop z thresholds (default 0.1, used with --soft-drop-eec)')
	parser.add_argument('--symmetric-eec', action='store_true', default=False,
	                    help='Also compute E2C for all splittings with z > --z-sym')
	parser.add_argument('--z-sym',         default=0.3,       type=float,
	                    help='Symmetry z threshold (default 0.3, used with --symmetric-eec)')
	parser.add_argument('--z-sym-kt-cut',  default=0.0,       type=float,
	                    help='Additional kT cut [GeV] for symmetric splittings (default 0, disabled)')
	parser.add_argument('--eec-nbins',          default=30,  type=int,
	                    help='Number of ln(ΔR) bins for EEC histograms (default 30)')
	parser.add_argument('--pt-particle-min',    default=0.0, type=float,
	                    help='Min particle pT [GeV] for EEC pairs (default 0, all particles)')
	args = parser.parse_args()

	# jet finder + Lund generator
	fj.ClusterSequence.print_banner()
	jetR     = args.jetR
	jet_def  = fj.JetDefinition(fj.antikt_algorithm, jetR)
	jet_sel  = fj.SelectorPtMin(args.jet_pt_min) * fj.SelectorAbsEtaMax(args.etadet - jetR * 1.05)
	if args.jet_pt_max > 0:
		jet_sel *= fj.SelectorPtMax(args.jet_pt_max)
	lund_gen = fj.contrib.LundGenerator()
	print('[i] LundGenerator:', lund_gen)

	# build cut pairs
	kt_cuts    = [float(x) for x in args.kt_cuts.split(',')]
	kappa_vals = [float(x) for x in args.kappa_cuts.split(',')]
	kappa_cuts = [kappa_vals[i] if i < len(kappa_vals) else 0.0 for i in range(len(kt_cuts))]

	eec_accums    = {(kt, kp): EECAccum(nbins=args.eec_nbins) for kt, kp in zip(kt_cuts, kappa_cuts)}
	lund_accum    = LundPlaneAccum()
	lund_accum_w  = LundPlaneAccum()
	lund_accum_lw = LundPlaneAccum() if args.lund_local_weight else None
	eec_maxkt     = EECAccum(nbins=args.eec_nbins) if args.max_kt_eec    else None
	eec_maxkt_thr = {(kt, kp): EECAccum(nbins=args.eec_nbins) for kt, kp in zip(kt_cuts, kappa_cuts)} \
	                if args.max_kt_eec else None
	z_sd_vals     = [float(x) for x in args.z_sd.split(',')]
	eec_sd_dict   = {z: EECAccum(nbins=args.eec_nbins) for z in z_sd_vals} if args.soft_drop_eec else None
	eec_sym       = EECAccum(nbins=args.eec_nbins) if args.symmetric_eec else None

	if lund_accum_lw: print('[i] local-weight Lund plane enabled  (pT_A*pT_B/pT_pair²)')
	if args.pt_particle_min > 0:
		print(f'[i] EEC particle pT cut: pT > {args.pt_particle_min} GeV')
	if eec_maxkt:   print('[i] max-kT EEC enabled (+ combined max_kt & kt_cut)')
	if eec_sd_dict: print(f'[i] soft-drop EEC enabled  z_sd values: {z_sd_vals}')
	if eec_sym:
		msg = f'[i] symmetric EEC enabled  (z_sym > {args.z_sym})'
		if args.z_sym_kt_cut > 0:
			msg += f'  AND kT > {args.z_sym_kt_cut} GeV'
		print(msg)

	# event loop
	events = file_events(args.input, max_events=args.max_events) if args.input \
	         else pythia_events(args)

	for parts in events:
		cseq = fj.ClusterSequence(parts, jet_def)
		jets = fj.sorted_by_pt(jet_sel(cseq.inclusive_jets()))
		for jet in jets:
			process_jet(jet, lund_gen, eec_accums, lund_accum, kt_cuts, kappa_cuts,
			            eec_maxkt=eec_maxkt, eec_maxkt_thr=eec_maxkt_thr,
			            eec_sd_dict=eec_sd_dict,
			            lund_accum_w=lund_accum_w, lund_accum_lw=lund_accum_lw,
			            eec_sym=eec_sym, z_sym=args.z_sym,
			            z_sym_kt_cut=args.z_sym_kt_cut,
			            pt_particle_min=args.pt_particle_min)

	# ── output ────────────────────────────────────────────────────────────────
	stem = os.path.splitext(args.output)[0]

	bare_label = args.label if args.label else 'default'
	frames = []
	for (kt_cut, kappa_cut), accum in eec_accums.items():
		accum.normalise()
		df = accum.to_df(label=bare_label, kt_cut=kt_cut)
		df['kappa_cut']  = kappa_cut
		df['selection']  = 'threshold'
		df['jet_pt_min'] = args.jet_pt_min
		df['jet_pt_max'] = args.jet_pt_max
		frames.append(df)

	if eec_maxkt is not None:
		eec_maxkt.normalise()
		df  = eec_maxkt.to_df(label=bare_label)
		df['kt_cut']    = float('nan')
		df['kappa_cut'] = float('nan')
		df['selection'] = 'max_kt'
		df['jet_pt_min'] = args.jet_pt_min
		df['jet_pt_max'] = args.jet_pt_max
		frames.append(df)

	if eec_maxkt_thr is not None:
		for (kt_cut, kappa_cut), accum in eec_maxkt_thr.items():
			accum.normalise()
			df = accum.to_df(label=bare_label, kt_cut=kt_cut)
			df['kappa_cut']  = kappa_cut
			df['selection']  = f'max_kt_kt>{kt_cut}'
			df['jet_pt_min'] = args.jet_pt_min
			df['jet_pt_max'] = args.jet_pt_max
			frames.append(df)

	if eec_sd_dict is not None:
		for z_sd, accum in eec_sd_dict.items():
			accum.normalise()
			df = accum.to_df(label=bare_label)
			df['kt_cut']     = float('nan')
			df['kappa_cut']  = float('nan')
			df['selection']  = f'soft_drop_z>{z_sd}'
			df['jet_pt_min'] = args.jet_pt_min
			df['jet_pt_max'] = args.jet_pt_max
			frames.append(df)

	if eec_sym is not None:
		eec_sym.normalise()
		df  = eec_sym.to_df(label=bare_label)
		df['kt_cut']     = args.z_sym_kt_cut if args.z_sym_kt_cut > 0 else float('nan')
		df['kappa_cut']  = float('nan')
		sel = f'symmetric_z>{args.z_sym}'
		if args.z_sym_kt_cut > 0:
			sel += f'_kt>{args.z_sym_kt_cut}'
		df['selection']  = sel
		df['jet_pt_min'] = args.jet_pt_min
		df['jet_pt_max'] = args.jet_pt_max
		frames.append(df)

	eec_df   = pd.concat(frames, ignore_index=True)
	eec_path = stem + '_eec.parquet'
	eec_df.to_parquet(eec_path, engine='pyarrow')
	print(f'[i] EEC histograms  -> {eec_path}')

	lund_path = stem + '_lundplane.npy'
	np.save(lund_path,                        lund_accum.h2)
	np.save(stem + '_lundplane_xedges.npy',   lund_accum.xedges)
	np.save(stem + '_lundplane_yedges.npy',   lund_accum.yedges)
	print(f'[i] Lund plane (density)  -> {lund_path}')

	lund_w_path = stem + '_lundplane_weighted.npy'
	np.save(lund_w_path,                      lund_accum_w.h2)
	print(f'[i] Lund plane (weighted) -> {lund_w_path}')

	if lund_accum_lw is not None:
		lund_lw_path = stem + '_lundplane_localweighted.npy'
		np.save(lund_lw_path, lund_accum_lw.h2)
		print(f'[i] Lund plane (local-weighted) -> {lund_lw_path}')


if __name__ == '__main__':
	main()
