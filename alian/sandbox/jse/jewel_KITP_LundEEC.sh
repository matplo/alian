# JEWEL file
data_dir="/Users/ploskon/data/jewel/fromLuisa"
# python lund_eec.py --input ${data_dir}/jewel_vac_100GeV.root --max-events 500 --kt-cuts 0,2 --label pp
# fname="jewel_100GeV_vacuum_99.root"
fname="jewel_100GeV_medium_99.root"

pthatmin=100
ptjetmin=$pthatmin
ptjetmax=$(( ptjetmin + ptjetmin/10 ))
nevents=10000
process="py-hardQCDbeauty"
process="py-hardQCD"
python lund_eec.py --nev $nevents \
--input ${data_dir}/${fname} \
--jet-pt-min $ptjetmin --jet-pt-max $ptjetmax \
--$process \
--label ppB \
--jetR 0.6 \
--kt-cuts 0,1,2 \
--max-kt-eec \
--soft-drop-eec --z-sd 0.1 \
--symmetric-eec --z-sym 0.3 \
--lund-local-weight

