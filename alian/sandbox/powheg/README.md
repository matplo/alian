# how to use powheg + pythia

- note: this is a copy of https://github.com/matplo/heppyy/blob/main/heppyy/example/powheg_pythia/README.md 
	- with additional Zjet that requires 
- this relies on heppyy installed (https://github.com/matplo/heppyy) as the rest of the alian

## compile some powheg components

- we assume you are using yasp (https://github.com/matplo/yasp)

```
# build fastjet
yasp -mi fastjet
module load fastjet
# build LHAPDF6
yasp -mi LHAPDF6
module load LHAPDF6
```

- now powheg for dijet, hvq, and Zjet processess

```
yasp -mi powheg --define proc=dijet,hvq,Zjet
module load powheg
```

## setup a powheg run

- note things got installed into `$POWHEG_DIR` - including the compiled execs in bin directory but also the code was synced so you can try examples setup - for example `$POWHEG_DIR/dijet/testrun-lhc/powheg.input`

- we have a local copy `powheg_dijet_lhc.input`

- lets soft link it to the `powheg.input`

```
ln -s powheg_dijet_lhc.input powheg.input
```

- we use cteq6l1 PDFs from LHAPDF - let's install with lhapdf

```
lhapdf install cteq6l1
```

## run powheg

- dijet
```
pwhg_dijet
```

- charm or beauty

```
lhapdf install cteq66
ln -sf powheg_hvq_c_lhc.input powheg.input
# or for b quarks
# ln -sf powheg_hvq_b_lhc.input powheg.input
pwhg_hvq
```

## shower+hadronize (analyze) with pythia8

### install and load pythia

```
# build pythia8
yasp -mi pythia8
module load pythia8
```

# python code

- load lhe file to pythia and run for some events (see [pythia8_powheg.py](https://github.com/matplo/heppyy/blob/main/heppyy/example/powheg_pythia/pythia8_powheg.py))
	- note we generated 1000 events as per powheg config file...
	- we read .lhe file + run pythia and fastjet for each event... 

```
./pythia8_powheg.py events.lhe --nev 1000 --verbose
```

# one liners

- to run Zjet look into `run_Zj.sh` or just run it... similar example for b's `run_b.sh`
