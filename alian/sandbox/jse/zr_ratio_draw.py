import argparse
import ROOT
import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser(description='Draw normalized z_r distributions and ratios.')
parser.add_argument(
    '--normalize-zr-gt', '--norm-zr-min',
    dest='norm_zr_gt',
    nargs='?',
    const=0.7,
    default=None,
    type=float,
    help='Normalize each distribution to the selected yield with that z_r variable above this value. '
         'If passed without a value, uses 0.7.',
)
args = parser.parse_args()

branches = ['pt', 'lzr01', 'lzr02']
printed_norms = set()


def normalize_hist(raw_counts, values, var, sample):
    if args.norm_zr_gt is None:
        norm = len(values)
    else:
        norm = np.count_nonzero((values > args.norm_zr_gt) & (values <= bins[-1]))
        norm_key = (sample, var)
        if norm_key not in printed_norms:
            print(f'Normalization entries for {sample} {var} > {args.norm_zr_gt:g}: {norm}')
            printed_norms.add(norm_key)

    if norm <= 0:
        raise ValueError(f'No entries available for {sample} {var} normalization')

    hist = raw_counts.astype(float) / norm
    err = np.sqrt(raw_counts).astype(float) / norm
    return hist, err


def norm_ylabel(var):
    if args.norm_zr_gt is None:
        return f'(1/N) dN/d{var}'
    return f'(1/N({var}>{args.norm_zr_gt:g})) dN/d{var}'


def norm_title(var):
    if args.norm_zr_gt is None:
        return 'normalized'
    return f'normalized to {var} > {args.norm_zr_gt:g}'


def ratio_title(var, den_low, den_high):
    if args.norm_zr_gt is None:
        return f'{var} ratio [100,150)/[{den_low},{den_high})'
    return f'{var} ratio [100,150)/[{den_low},{den_high}), norm {var}>{args.norm_zr_gt:g}'


def compute_ratio(num, num_err, den, den_err):
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(den > 0, num / den, np.nan)
        err = np.where(
            (num > 0) & (den > 0),
            ratio * np.sqrt((num_err / np.where(num > 0, num, np.nan))**2 +
                            (den_err / np.where(den > 0, den, np.nan))**2),
            np.nan
        )
    return ratio, err

with uproot.open('pythia_zr_pthatmin100.root') as f:
    arr100 = f['tnR04'].arrays(branches, library='np')

with uproot.open('pythia_zr_pthatmin120.root') as f:
    arr120 = f['tnR04'].arrays(branches, library='np')

with uproot.open('pythia_zr_pthatmin115.root') as f:
    arr115 = f['tnR04'].arrays(branches, library='np')

sel100 = (arr100['pt'] > 100) & (arr100['pt'] < 150)
sel120 = (arr120['pt'] > 120) & (arr120['pt'] < 170)
sel115 = (arr115['pt'] > 115) & (arr115['pt'] < 165)

n100 = sel100.sum()
n120 = sel120.sum()
n115 = sel115.sum()
print(f'Selected entries: pthat100 file = {n100}, pthat120 file = {n120}, pthat115 file = {n115}')
if n100 <= 0:
    raise ValueError('No selected entries in pthat100 file for ratio reporting')
print(f'Jet count ratio n120/n100 = {n120 / n100:.6g}')
print(f'Jet count ratio n115/n100 = {n115 / n100:.6g}')

zr_max = 1.001
zr_bin_width = 0.035
n_full_bins = int(np.floor(zr_max / zr_bin_width))
bins = np.concatenate((
    [0.0],
    np.round(zr_max - zr_bin_width * np.arange(n_full_bins, -1, -1), 12)
))

# --- individual plots with ratio panels ---
for var in ['lzr01', 'lzr02']:
    vals100 = arr100[var][sel100]
    vals120 = arr120[var][sel120]
    vals115 = arr115[var][sel115]
    h100_raw, edges = np.histogram(vals100, bins=bins)
    h120_raw, _     = np.histogram(vals120, bins=bins)
    h115_raw, _     = np.histogram(vals115, bins=bins)

    h100, e100 = normalize_hist(h100_raw, vals100, var, 'pthat100')
    h120, e120 = normalize_hist(h120_raw, vals120, var, 'pthat120')
    h115, e115 = normalize_hist(h115_raw, vals115, var, 'pthat115')

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths  = np.diff(edges)

    ratio120, e_ratio120 = compute_ratio(h100, e100, h120, e120)
    ratio115, e_ratio115 = compute_ratio(h100, e100, h115, e115)

    fig = plt.figure(figsize=(7, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    ax0.bar(centers, h100, width=widths, alpha=0.4, color='steelblue', label=r'$p_T \in [100,150)$ GeV')
    ax0.errorbar(centers, h100, yerr=e100, fmt='none', ecolor='steelblue', capsize=3)
    ax0.bar(centers, h120, width=widths, alpha=0.4, color='tomato', label=r'$p_T \in [120,170)$ GeV')
    ax0.errorbar(centers, h120, yerr=e120, fmt='none', ecolor='tomato', capsize=3)
    ax0.bar(centers, h115, width=widths, alpha=0.4, color='seagreen', label=r'$p_T \in [115,165)$ GeV')
    ax0.errorbar(centers, h115, yerr=e115, fmt='none', ecolor='seagreen', capsize=3)
    ax0.set_ylabel(norm_ylabel(var))
    ax0.legend()
    ax0.set_title(f'tnR04 — {var}, {norm_title(var)}')
    plt.setp(ax0.get_xticklabels(), visible=False)

    ax1.axhline(1, color='k', lw=0.8, ls='--')
    ax1.errorbar(centers, ratio120, yerr=e_ratio120, fmt='o', color='dimgray', capsize=3, ms=4,
                 label='[100,150)/[120,170)')
    ax1.errorbar(centers, ratio115, yerr=e_ratio115, fmt='s', color='seagreen', capsize=3, ms=4,
                 label='[100,150)/[115,165)')
    ax1.set_ylabel('ratio')
    ax1.set_xlabel(var)
    ax1.set_ylim(0, 3)
    ax1.set_xlim(edges[0], edges[-1])
    ax1.legend()

    plt.tight_layout()
    plt.savefig(f'{var}_ratio.pdf', bbox_inches='tight')
    print(f'Saved {var}_ratio.pdf')
    plt.close()

# --- combined ratio plot ---
fig, ax = plt.subplots(figsize=(7, 4))
colors = {'lzr01': 'steelblue', 'lzr02': 'tomato'}

fout = ROOT.TFile('zr_ratios.root', 'RECREATE')

for var in ['lzr01', 'lzr02']:
    vals100 = arr100[var][sel100]
    vals120 = arr120[var][sel120]
    vals115 = arr115[var][sel115]
    h100_raw, edges = np.histogram(vals100, bins=bins)
    h120_raw, _     = np.histogram(vals120, bins=bins)
    h115_raw, _     = np.histogram(vals115, bins=bins)

    h100, e100 = normalize_hist(h100_raw, vals100, var, 'pthat100')
    h120, e120 = normalize_hist(h120_raw, vals120, var, 'pthat120')
    h115, e115 = normalize_hist(h115_raw, vals115, var, 'pthat115')

    centers = 0.5 * (edges[:-1] + edges[1:])
    ex      = np.zeros_like(centers)

    ratio120, e_ratio120 = compute_ratio(h100, e100, h120, e120)
    ratio115, e_ratio115 = compute_ratio(h100, e100, h115, e115)

    ax.errorbar(centers, ratio120, yerr=e_ratio120, fmt='o', color=colors[var],
                capsize=3, ms=4, label=f'{var} [100,150)/[120,170)')
    ax.errorbar(centers, ratio115, yerr=e_ratio115, fmt='s', color=colors[var],
                capsize=3, ms=4, mfc='white', label=f'{var} [100,150)/[115,165)')

    # TGraphErrors for ratios
    for suffix, den_low, den_high, ratio_vals, ratio_errs in [
        ('120170', 120, 170, ratio120, e_ratio120),
        ('115165', 115, 165, ratio115, e_ratio115),
    ]:
        g = ROOT.TGraphErrors(len(centers),
                              centers.astype('d'),
                              np.nan_to_num(ratio_vals).astype('d'),
                              ex.astype('d'),
                              np.nan_to_num(ratio_errs).astype('d'))
        g.SetName(f'ratio_{var}_{suffix}')
        g.SetTitle(f'{ratio_title(var, den_low, den_high)};z_{{r}};ratio')
        g.Write()

    # TH1D for normalized distributions
    nbins = len(edges) - 1
    for tag, vals, errs in [('100', h100, e100), ('120', h120, e120), ('115', h115, e115)]:
        th = ROOT.TH1D(f'h_{var}_{tag}', f'{var} pthat{tag};z_{{r}};{norm_ylabel(var)}',
                       nbins, edges.astype('d'))
        for i, (v, e) in enumerate(zip(vals, errs), start=1):
            th.SetBinContent(i, v)
            th.SetBinError(i, e)
        th.Write()

    # TH1D for ratios
    for suffix, den_low, den_high, ratio_vals, ratio_errs in [
        ('120170', 120, 170, ratio120, e_ratio120),
        ('115165', 115, 165, ratio115, e_ratio115),
    ]:
        th_ratio = ROOT.TH1D(f'h_ratio_{var}_{suffix}',
                             f'{ratio_title(var, den_low, den_high)};z_{{r}};ratio',
                             nbins, edges.astype('d'))
        for i, (v, e) in enumerate(zip(np.nan_to_num(ratio_vals), np.nan_to_num(ratio_errs)), start=1):
            th_ratio.SetBinContent(i, v)
            th_ratio.SetBinError(i, e)
        th_ratio.Write()

fout.Close()
print('Saved zr_ratios.root')

ax.axhline(1, color='k', lw=0.8, ls='--')
ax.set_xlabel('z_r')
ax.set_ylabel('ratio')
if args.norm_zr_gt is None:
    ax.set_title('Ratios of normalized distributions (tnR04)')
else:
    ax.set_title(f'Ratios of distributions normalized to z_r > {args.norm_zr_gt:g} (tnR04)')
ax.set_ylim(0.5, 1.5)
ax.set_xlim(0.7, 1.001)
ax.legend()
plt.tight_layout()
plt.savefig('lzr_ratio_combined.pdf', bbox_inches='tight')
print('Saved lzr_ratio_combined.pdf')
plt.close()

# --- envelope-based mean ratio objects from the two denominator choices ---
fupdate = ROOT.TFile('zr_ratios.root', 'UPDATE')

for var in ['lzr01', 'lzr02']:
    h_a = fupdate.Get(f'h_ratio_{var}_120170')
    h_b = fupdate.Get(f'h_ratio_{var}_115165')
    if not h_a or not h_b:
        raise RuntimeError(f'Missing input ratio histograms for {var} in zr_ratios.root')

    nbins = h_a.GetNbinsX()
    if h_b.GetNbinsX() != nbins:
        raise RuntimeError(f'Inconsistent binning between ratio histograms for {var}')

    h_mean = h_a.Clone(f'h_ratio_mean_{var}')
    h_mean.Reset('ICES')
    h_mean.SetTitle(f'{var} mean ratio with envelope error;z_{{r}};ratio')

    x = np.zeros(nbins, dtype='d')
    ex = np.zeros(nbins, dtype='d')
    y = np.zeros(nbins, dtype='d')
    eyl = np.zeros(nbins, dtype='d')
    eyh = np.zeros(nbins, dtype='d')

    for i in range(1, nbins + 1):
        r1 = h_a.GetBinContent(i)
        e1 = h_a.GetBinError(i)
        r2 = h_b.GetBinContent(i)
        e2 = h_b.GetBinError(i)

        m = 0.5 * (r1 + r2)
        low_edge = min(r1 - e1, r2 - e2)
        high_edge = max(r1 + e1, r2 + e2)
        dev_low = m - low_edge
        dev_high = high_edge - m
        dev_sym = max(dev_low, dev_high)

        h_mean.SetBinContent(i, m)
        h_mean.SetBinError(i, dev_sym)

        x[i - 1] = h_a.GetBinCenter(i)
        y[i - 1] = m
        eyl[i - 1] = dev_low
        eyh[i - 1] = dev_high

    h_mean.Write('', ROOT.TObject.kOverwrite)

    g_mean_env = ROOT.TGraphAsymmErrors(nbins, x, y, ex, ex, eyl, eyh)
    g_mean_env.SetName(f'gr_ratio_mean_env_{var}')
    g_mean_env.SetTitle(f'{var} mean ratio envelope;z_{{r}};ratio')
    g_mean_env.Write('', ROOT.TObject.kOverwrite)

fupdate.Close()
print('Saved envelope-based mean ratio objects to zr_ratios.root')
