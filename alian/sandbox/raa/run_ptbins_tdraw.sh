#!/bin/bash

inputfile=$1

if [ -z "$inputfile" ]; then
	inputfile=pythia8_simple_eec_output.root
fi


fquench=0
for jetpt in 20 40 60 80 100
do
	jetptmin=$jetpt
	jetptmax=$((jetpt + 20))
	clone_fname="PbPb_data.root"
	clone_hname="h1d_EEC_${jetptmin}${jetptmax}_pp"
	if [ ${jetpt} == "20" ];
	then
		jetptmax=150
		clone_hname="h1d_EEC_4060_pp"
	fi
	outputfile="${inputfile%.root}_${jetptmin}_${jetptmax}_h.root"
	for fq in 0 7 10 12 14 20
	do
		outputfile_q="${outputfile%.root}_${fq}.root"
		# ./exec_tdraw.sh ${inputfile} ${jetptmin} ${jetptmax} ${outputfile_q} ${fquench} ${fq}
		draw_from_yaml.py -c tdraw_eec_eloss.yaml -d output=${outputfile_q} input=${inputfile} jetptmin=${jetptmin} jetptmax=${jetptmax} fquench=${fquench} fquenchconst=${fq} clone_fname=${clone_fname} clone_hname=${clone_hname}
	done
done

