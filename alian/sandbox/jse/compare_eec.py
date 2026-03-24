#!/usr/bin/env python3
"""
compare_eec.py — compare EEC distributions and Lund planes across multiple runs.

Usage:
    python compare_eec.py --config compare_eec_example.yaml [--show] [--no-save]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yaml


# ── secondary-axis helpers ────────────────────────────────────────────────────

def _log_secaxis(sec, axis, label):
    _ax = getattr(sec, f'{axis}axis')
    _ax.set_major_locator(mticker.LogLocator(base=10, numticks=8))
    _ax.set_major_formatter(mticker.LogFormatterMathtext(labelOnlyBase=False))
    getattr(sec, f'set_{axis}label')(label)

def add_lund_secaxis(ax):
    secx = ax.secondary_xaxis('top',
        functions=(lambda x: np.exp(-np.clip(x, -30, 30)),
                   lambda r: -np.log(np.clip(r, 1e-9, None))))
    _log_secaxis(secx, 'x', r'$\Delta$ [rad]')

def add_lund_secyaxis(ax):
    secy = ax.secondary_yaxis('right',
        functions=(lambda y: np.exp(np.clip(y, -30, 30)),
                   lambda k: np.log(np.clip(k, 1e-9, None))))
    _log_secaxis(secy, 'y', r'$k_t$ [GeV]')

def add_eec_secaxis(ax):
    secx = ax.secondary_xaxis('top',
        functions=(lambda x: np.exp(np.clip(x, -30, 30)),
                   lambda r: np.log(np.clip(r, 1e-9, None))))
    _log_secaxis(secx, 'x', r'$\Delta R$ [rad]')

def add_z_half_line(ax, xedges, jet_pt_rep):
    x_arr = np.array([xedges[0], xedges[-1]])
    ax.plot(x_arr, -x_arr + np.log(0.5 * jet_pt_rep),
            color='red', lw=1.5, ls='-', label=r'$z=0.5$')


# ── Lund type registry ────────────────────────────────────────────────────────

LUND_TYPES = {
    'density':        ('_lundplane.npy',
                       'counts',
                       r'density'),
    'weighted':       ('_lundplane_weighted.npy',
                       r'$\sum p_{T,A}p_{T,B}/p_{T,j}^2$',
                       r'energy-weighted'),
    'local_weighted': ('_lundplane_localweighted.npy',
                       r'$\sum p_{T,A}p_{T,B}/p_{T,\rm pair}^2$',
                       r'local-weighted'),
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_run(path):
    """Load EEC parquet and all available Lund-plane arrays from a run directory."""
    stem = os.path.join(path, 'lund_eec_output')
    data = {}

    eec_path = stem + '_eec.parquet'
    if os.path.exists(eec_path):
        data['eec_df'] = pd.read_parquet(eec_path)
    else:
        data['eec_df'] = None
        print(f'[w] no EEC parquet: {eec_path}')

    for key in ('_lundplane_xedges.npy', '_lundplane_yedges.npy'):
        p = stem + key
        attr = 'xedges' if 'xedges' in key else 'yedges'
        data[attr] = np.load(p) if os.path.exists(p) else None

    for ltype, (suffix, _, _) in LUND_TYPES.items():
        p = stem + suffix
        data[f'lund_{ltype}'] = np.load(p) if os.path.exists(p) else None

    return data


# ── save / show ───────────────────────────────────────────────────────────────

def _finish(fig, name, cfg, show, save):
    # sanitise filename
    safe = name.replace('>', 'gt').replace('<', 'lt').replace(' ', '_').replace('/', '_')
    if save:
        outdir = cfg.get('output_dir', 'compare_figures')
        os.makedirs(outdir, exist_ok=True)
        for ext in cfg.get('formats', ['pdf', 'png']):
            fig.savefig(os.path.join(outdir, f'{safe}.{ext}'), bbox_inches='tight')
            print(f'[i] saved {safe}.{ext}')
    if show:
        plt.show()
    plt.close(fig)


# ── EEC helpers ───────────────────────────────────────────────────────────────

_eec_names = {'eec_all': r'A∪B (all)', 'eec_AA': r'A×A',
              'eec_BB': r'B×B',        'eec_AB': r'A×B (cross)'}
_eec_ls    = {'eec_all': '-', 'eec_AA': '--', 'eec_BB': '-.', 'eec_AB': ':'}


def _get_eec_group(eec_df, sel):
    """Return sub-df for a selection string.
    Shorthand 'threshold_ktX' maps to selection='threshold' with kt_cut=X."""
    if sel.startswith('threshold_kt'):
        kt = float(sel.split('threshold_kt')[1])
        return eec_df[(eec_df['selection'] == 'threshold') & (eec_df['kt_cut'] == kt)]
    return eec_df[eec_df['selection'] == sel]


def _eec_uncert(vals, n_jets):
    """Per-bin statistical uncertainty: σ ≈ sqrt(h / n_jets).

    Approximation: each jet is an independent contributor; variance per bin
    scales as the mean contribution divided by n_jets.  Valid for n_jets >> 1
    and smooth distributions.  Requires no additional stored quantities.
    """
    if n_jets <= 0:
        return np.zeros_like(vals)
    return np.sqrt(np.clip(vals, 0.0, None) / n_jets)


def _apply_eec_ylim(ax, cfg, is_ratio=False):
    """Apply y-axis range from config (eec.ratio_ylim or eec.ylim)."""
    eec_cfg = cfg.get('eec', {})
    ylim = eec_cfg.get('ratio_ylim' if is_ratio else 'ylim', None)
    if ylim is not None:
        ax.set_ylim(ylim)


def _apply_eec_xlim(ax, cfg):
    """Apply x-axis range from config eec.xlim (given as ΔR values, converted to ln(ΔR))."""
    xlim = cfg.get('eec', {}).get('xlim', None)
    if xlim is not None:
        ax.set_xlim(np.log(xlim[0]), np.log(xlim[1]))


def _ratio_uncert(h_a, n_a, h_b, n_b, ratio):
    """Stat uncertainty on ratio A/B via error propagation.

    σ_R/R = sqrt(σ_A²/A² + σ_B²/B²)
    With σ_X = sqrt(X/n_X):  σ_R = R * sqrt(1/(A*n_A) + 1/(B*n_B))
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_var = (np.where(h_a > 0, 1.0 / (h_a * n_a), 0.0) +
                   np.where(h_b > 0, 1.0 / (h_b * n_b), 0.0))
        sigma = np.where(np.isfinite(ratio), ratio * np.sqrt(rel_var), np.nan)
    return sigma


# ── EEC: overlay ──────────────────────────────────────────────────────────────

def plot_eec_overlay(runs, selection, quantities, cfg, show, save):
    """All runs overlaid; one panel per quantity, with optional uncertainty bands."""
    show_unc = cfg.get('uncertainties', True)
    fig, axes = plt.subplots(1, len(quantities),
                             figsize=(6 * len(quantities), 5), sharex=True)
    if len(quantities) == 1:
        axes = [axes]
    colors = plt.cm.tab10.colors
    for qi, qty in enumerate(quantities):
        ax = axes[qi]
        for ci, run in enumerate(runs):
            if run['eec_df'] is None:
                continue
            grp = _get_eec_group(run['eec_df'], selection)
            if grp.empty:
                continue
            vals   = grp[qty].values
            x      = grp['ln_dR'].values
            color  = run.get('color') or colors[ci % 10]
            ax.step(x, vals, where='mid', label=run['label'],
                    color=color, ls=_eec_ls.get(qty, '-'))
            if show_unc:
                n_jets = grp['n_jets'].iloc[0]
                sigma  = _eec_uncert(vals, n_jets)
                ax.fill_between(x, vals - sigma, vals + sigma,
                                alpha=0.15, color=color, step='mid')
        ax.set_title(_eec_names.get(qty, qty))
        ax.set_xlabel(r'$\ln(\Delta R)$')
        if qi == 0:
            ax.set_ylabel('E2C')
        ax.legend(fontsize=9)
        add_eec_secaxis(ax)
        _apply_eec_ylim(ax, cfg)
        _apply_eec_xlim(ax, cfg)
    plt.suptitle(f'E2C overlay — {selection}')
    plt.tight_layout()
    _finish(fig, f'eec_overlay_{selection}_{"_".join(quantities)}', cfg, show, save)


# ── EEC: ratio ────────────────────────────────────────────────────────────────

def plot_eec_ratio(runs, ref_run, selection, quantities, cfg, show, save):
    """Each non-reference run divided by the reference; one panel per quantity.
    Uncertainty bands from error propagation: σ_R = R·sqrt(1/(A·n_A) + 1/(B·n_B))."""
    non_ref = [r for r in runs if r is not ref_run]
    if not non_ref:
        return
    show_unc = cfg.get('uncertainties', True)
    fig, axes = plt.subplots(1, len(quantities),
                             figsize=(6 * len(quantities), 5), sharex=True)
    if len(quantities) == 1:
        axes = [axes]
    colors = plt.cm.tab10.colors
    ref_grp = _get_eec_group(ref_run['eec_df'], selection) \
              if ref_run['eec_df'] is not None else None

    for qi, qty in enumerate(quantities):
        ax = axes[qi]
        if ref_grp is None or ref_grp.empty:
            ax.set_title(f'{_eec_names.get(qty, qty)} — no reference data')
            continue
        h_b  = ref_grp[qty].values
        n_b  = ref_grp['n_jets'].iloc[0]
        x    = ref_grp['ln_dR'].values
        for ci, run in enumerate(non_ref):
            if run['eec_df'] is None:
                continue
            grp = _get_eec_group(run['eec_df'], selection)
            if grp.empty:
                continue
            h_a   = grp[qty].values
            n_a   = grp['n_jets'].iloc[0]
            color = run.get('color') or colors[ci % 10]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(h_b > 0, h_a / h_b, np.nan)
            ax.step(x, ratio, where='mid',
                    label=f'{run["label"]} / {ref_run["label"]}', color=color)
            if show_unc:
                sigma = _ratio_uncert(h_a, n_a, h_b, n_b, ratio)
                ax.fill_between(x, ratio - sigma, ratio + sigma,
                                alpha=0.2, color=color, step='mid')
        ax.axhline(1.0, ls=':', color='grey', lw=1)
        ax.set_title(_eec_names.get(qty, qty))
        ax.set_xlabel(r'$\ln(\Delta R)$')
        if qi == 0:
            ax.set_ylabel('ratio')
        ax.legend(fontsize=9)
        add_eec_secaxis(ax)
        _apply_eec_ylim(ax, cfg, is_ratio=True)
        _apply_eec_xlim(ax, cfg)
    plt.suptitle(f'E2C ratio — {selection}  (ref: {ref_run["label"]})')
    plt.tight_layout()
    _finish(fig, f'eec_ratio_{selection}_{"_".join(quantities)}', cfg, show, save)


# ── EEC: side by side ─────────────────────────────────────────────────────────

def plot_eec_side_by_side(runs, selection, quantities, cfg, show, save):
    """Grid: rows = runs, columns = quantities, with uncertainty bands."""
    show_unc = cfg.get('uncertainties', True)
    nr, nq = len(runs), len(quantities)
    fig, axes = plt.subplots(nr, nq, figsize=(6 * nq, 4 * nr),
                             sharey='row', sharex=True, squeeze=False)
    for ri, run in enumerate(runs):
        color = run.get('color') or 'black'
        for qi, qty in enumerate(quantities):
            ax = axes[ri][qi]
            if run['eec_df'] is not None:
                grp = _get_eec_group(run['eec_df'], selection)
                if not grp.empty:
                    vals = grp[qty].values
                    x    = grp['ln_dR'].values
                    ax.step(x, vals, where='mid', color=color)
                    if show_unc:
                        sigma = _eec_uncert(vals, grp['n_jets'].iloc[0])
                        ax.fill_between(x, vals - sigma, vals + sigma,
                                        alpha=0.2, color=color, step='mid')
            if ri == 0:
                ax.set_title(_eec_names.get(qty, qty))
            if qi == 0:
                ax.set_ylabel(run['label'], fontsize=9)
            if ri == nr - 1:
                ax.set_xlabel(r'$\ln(\Delta R)$')
            add_eec_secaxis(ax)
            _apply_eec_ylim(ax, cfg)
            _apply_eec_xlim(ax, cfg)
    plt.suptitle(f'E2C side-by-side — {selection}')
    plt.tight_layout()
    _finish(fig, f'eec_sbs_{selection}_{"_".join(quantities)}', cfg, show, save)


# ── Lund: individual ──────────────────────────────────────────────────────────

def _lund_mesh(run):
    xedges, yedges = run['xedges'], run['yedges']
    Xc = 0.5 * (xedges[:-1] + xedges[1:])
    Yc = 0.5 * (yedges[:-1] + yedges[1:])
    return np.meshgrid(Xc, Yc, indexing='ij'), xedges, yedges

def _jet_pt_rep(run):
    eec_df = run.get('eec_df')
    if eec_df is None:
        return None
    return 0.5 * (eec_df['jet_pt_min'].iloc[0] + eec_df['jet_pt_max'].iloc[0])


def plot_lund_individual(runs, lund_type, cfg, show, save):
    """One panel per run."""
    _, cbar_label, type_title = LUND_TYPES[lund_type]
    valid = [r for r in runs if r.get(f'lund_{lund_type}') is not None]
    if not valid:
        print(f'[w] no {lund_type} Lund data in any run — skipping individual plot')
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(7 * len(valid), 6),
                             layout='constrained')
    if len(valid) == 1:
        axes = [axes]

    for ax, run in zip(axes, valid):
        h2 = run[f'lund_{lund_type}']
        (X, Y), xedges, yedges = _lund_mesh(run)
        pcm = ax.pcolormesh(X, Y, h2, cmap='Greys', shading='auto')
        plt.colorbar(pcm, ax=ax, label=cbar_label)
        ax.set_xlabel(r'$\ln(1/\Delta)$')
        ax.set_ylabel(r'$\ln(k_t)$')
        ax.set_title(run['label'])
        pt = _jet_pt_rep(run)
        if pt is not None:
            add_z_half_line(ax, xedges, pt)
            ax.legend(fontsize=9, loc='upper right')
        add_lund_secaxis(ax)
        add_lund_secyaxis(ax)

    plt.suptitle(f'Lund plane — {type_title}')
    _finish(fig, f'lund_{lund_type}_individual', cfg, show, save)


# ── Lund: ratio ───────────────────────────────────────────────────────────────

def plot_lund_ratio(run_a, run_b, lund_type, cfg, show, save):
    """log(run_a / run_b) for one Lund type.

    Bins where the reference (run_b) falls below lund.ratio_min_count are
    masked (shown as white / NaN).  An optional second panel shows the
    Poisson significance (h_a − h_b) / sqrt(h_a + h_b) per bin.
    """
    h2_a = run_a.get(f'lund_{lund_type}')
    h2_b = run_b.get(f'lund_{lund_type}')
    if h2_a is None or h2_b is None:
        print(f'[w] skipping Lund ratio {lund_type}: missing data in one run')
        return

    lund_cfg    = cfg.get('lund', {})
    min_count   = lund_cfg.get('ratio_min_count', 5)
    show_sig    = lund_cfg.get('show_significance', True)

    # mask bins with too few reference counts
    valid = h2_b >= min_count

    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.where(valid, np.log(np.where(valid, h2_a / h2_b, 1.0)), np.nan)

    # Poisson significance: (A − B) / sqrt(A + B)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.sqrt(h2_a + h2_b)
        significance = np.where(valid & (denom > 0), (h2_a - h2_b) / denom, np.nan)

    # symmetric color scale at 95th percentile of |log ratio|
    finite = log_ratio[np.isfinite(log_ratio)]
    vmax = np.percentile(np.abs(finite), 95) if finite.size > 0 else 1.0
    vmax = max(vmax, 0.1)

    ref_run = run_a if run_a['xedges'] is not None else run_b
    (X, Y), xedges, _ = _lund_mesh(ref_run)
    pt = _jet_pt_rep(run_a) or _jet_pt_rep(run_b)

    ncols = 2 if show_sig else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6), layout='constrained')
    if ncols == 1:
        axes = [axes]

    # ── panel 1: log ratio ────────────────────────────────────────────────────
    ax = axes[0]
    pcm = ax.pcolormesh(X, Y, log_ratio, cmap='RdBu_r', shading='auto',
                        vmin=-vmax, vmax=vmax)
    plt.colorbar(pcm, ax=ax,
                 label=rf'$\ln({run_a["label"]} / {run_b["label"]})$')
    ax.set_xlabel(r'$\ln(1/\Delta)$')
    ax.set_ylabel(r'$\ln(k_t)$')
    ax.set_title(rf'{run_a["label"]} / {run_b["label"]}  (log ratio)')
    if pt is not None:
        add_z_half_line(ax, xedges, pt)
        ax.legend(fontsize=9, loc='upper right')
    add_lund_secaxis(ax)
    add_lund_secyaxis(ax)

    # ── panel 2: significance ─────────────────────────────────────────────────
    if show_sig:
        ax2 = axes[1]
        sig_finite = significance[np.isfinite(significance)]
        smax = np.percentile(np.abs(sig_finite), 95) if sig_finite.size > 0 else 3.0
        smax = max(smax, 1.0)
        pcm2 = ax2.pcolormesh(X, Y, significance, cmap='RdBu_r', shading='auto',
                               vmin=-smax, vmax=smax)
        plt.colorbar(pcm2, ax=ax2,
                     label=r'$(A-B)/\sqrt{A+B}$  [Poisson $\sigma$]')
        ax2.set_xlabel(r'$\ln(1/\Delta)$')
        ax2.set_ylabel(r'$\ln(k_t)$')
        ax2.set_title(rf'{run_a["label"]} vs {run_b["label"]}  (significance)')
        if pt is not None:
            add_z_half_line(ax2, xedges, pt)
            ax2.legend(fontsize=9, loc='upper right')
        add_lund_secaxis(ax2)
        add_lund_secyaxis(ax2)

    plt.suptitle(f'Lund plane ratio — {lund_type}  '
                 rf'(masked: $n_{{ref}} < {min_count}$)')
    tag = f'{run_a["label"]}_over_{run_b["label"]}'.replace(' ', '_')
    _finish(fig, f'lund_ratio_{lund_type}_{tag}', cfg, show, save)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare EEC & Lund plane outputs across runs')
    parser.add_argument('--config',   required=True, help='YAML config file')
    parser.add_argument('--show',     action='store_true', help='Show plots interactively')
    parser.add_argument('--no-save',  action='store_true', help='Disable saving figures')
    args = parser.parse_args()
    save = not args.no_save

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── load runs ──────────────────────────────────────────────────────────────
    runs = []
    for rc in cfg.get('runs', []):
        print(f'[i] loading  {rc["path"]}')
        d = load_run(rc['path'])
        d['label'] = rc.get('label', os.path.basename(rc['path']))
        d['color'] = rc.get('color', None)
        runs.append(d)

    if not runs:
        print('[e] no runs loaded'); return

    ref_label = cfg.get('reference', runs[0]['label'])
    ref_run   = next((r for r in runs if r['label'] == ref_label), runs[0])
    print(f'[i] reference run: {ref_run["label"]}')

    # ── EEC ────────────────────────────────────────────────────────────────────
    selections = cfg.get('selections', [])
    quantities = cfg.get('quantities', ['eec_all'])
    eec_modes  = cfg.get('eec', {}).get('plot_modes', ['overlay'])

    for sel in selections:
        if 'overlay' in eec_modes:
            plot_eec_overlay(runs, sel, quantities, cfg, args.show, save)
        if 'ratio' in eec_modes:
            plot_eec_ratio(runs, ref_run, sel, quantities, cfg, args.show, save)
        if 'side_by_side' in eec_modes:
            plot_eec_side_by_side(runs, sel, quantities, cfg, args.show, save)

    # ── Lund planes ────────────────────────────────────────────────────────────
    lund_cfg   = cfg.get('lund', {})
    lund_types = lund_cfg.get('types', ['density'])
    lund_modes = lund_cfg.get('plot_modes', ['individual'])

    for ltype in lund_types:
        if 'individual' in lund_modes:
            plot_lund_individual(runs, ltype, cfg, args.show, save)
        if 'ratio' in lund_modes:
            non_ref = [r for r in runs if r is not ref_run]
            for run in non_ref:
                plot_lund_ratio(run, ref_run, ltype, cfg, args.show, save)


if __name__ == '__main__':
    main()
