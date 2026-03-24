pthatmin=100
ptjetmin=$pthatmin
ptjetmax=$(( ptjetmin + ptjetmin/10 ))
nevents=100000
# process="py-hardQCDbeauty"
process="py-hardQCD"

label="pp"
jetR="0.6"
etadet="2.5"
kt_cuts="0,1,2"
kappa_cuts="0"
z_sd="0.1,0.2,0.3"
z_sym="0.3"
z_sym_kt_cut="0.0"
pt_particle_min="1.0"   # min particle pT [GeV] for EEC pairs; 0 = all particles

# input for file mode (leave empty for Pythia mode)
input=""
data_dir="/Users/ploskon/data/jewel/fromLuisa"
# input="${data_dir}/jewel_100GeV_medium_99.root"
# input="${data_dir}/jewel_100GeV_vacuum_99.root"
# input="jewel_*.root"
# input="${data_dir}/jewel_100GeV_vacuum_9*.root"
input="${data_dir}/jewel_100GeV_medium_9*.root"
max_events=""  # e.g. 1000; empty = all
max_events=${nevents}  # override max_events if nev is set (for Pythia mode)

# ── auto output directory ─────────────────────────────────────────────────────
jetR_tag="${jetR/./}"                       # 0.6 → 06
pt_tag="ptpmin${pt_particle_min/./p}"       # 0.0 → ptpmin0p0, 0.5 → ptpmin0p5

if [ -n "$input" ]; then
    src_tag=$(basename "${input%% *}" .root)
    outdir="${src_tag}_R${jetR_tag}_pt${ptjetmin}_${ptjetmax}_${pt_tag}_${label}"
else
    outdir="${process}_pthat${pthatmin}_R${jetR_tag}_pt${ptjetmin}_${ptjetmax}_N${nevents}_${pt_tag}_${label}"
fi
mkdir -p "$outdir"
output="$outdir/lund_eec_output.parquet"
echo "[i] output dir: $outdir"

# ── run ───────────────────────────────────────────────────────────────────────
if [ -n "$input" ]; then
    mode_args="--input $input"
    [ -n "$max_events" ] && mode_args="$mode_args --max-events $max_events"
else
    mode_args="--nev $nevents --py-pthatmin $pthatmin --$process"
fi

python lund_eec.py $mode_args \
--jet-pt-min $ptjetmin --jet-pt-max $ptjetmax \
--etadet $etadet \
--label $label \
--jetR $jetR \
--output $output \
--kt-cuts $kt_cuts \
--kappa-cuts $kappa_cuts \
--pt-particle-min $pt_particle_min \
--inclusive-eec \
--max-kt-eec \
--soft-drop-eec --z-sd $z_sd \
--symmetric-eec --z-sym $z_sym --z-sym-kt-cut $z_sym_kt_cut \
--lund-local-weight
