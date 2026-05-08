#!/usr/bin/bash

# A simple script to merge files within a directory. Feel free to make a copy
# and customize to your needs.

show_help() {
    cat << EOF
Usage:
  $0 -i|--input INPUT -o|--output OUTPUT [-n|--name FILENAME] [-f|--force] [-h|--help]

Searches for files in a given directory INPUT and a given filename FILENAME,
and merges them with ROOT hadd into a specified output file OUTPUT.
The list of files that were merged will be placed in a text file in the
same directory as the output file.

Options:
  -i, --input INPUT    Input directory to search for files
  -o, --output OUTPUT  Filename for merged output
  -n, --name NAME      Search for NAME in INPUT (default: analysis.root)
  -j, --jobs [N]       Parallelize merging with N jobs (default: system max $(nproc))
  -f, --force          Overwrite existing output file (default: false)
  -h, --help           Show this help message

EOF
}

if ! command -v hadd >/dev/null 2>&1; then echo "hadd not available, crashing." && exit 127; fi

if [ $# -eq 0 ]; then
    show_help && exit 0
fi

PARSEDARGS=$(getopt -o i:o:n:j:fh \
                    --long input:,output:,name:,jobs:,force,help \
                    -n 'merge' -- "$@")
PARSE_EXIT=$?
if [ $PARSE_EXIT -ne 0 ] ; then exit $PARSE_EXIT ; fi

# NOTE: quotes around PARSEDARGS essential to parse spaces between parsed options
eval set -- "$PARSEDARGS"

input=
output=
filename=analysis.root
force_opt=
njobs=$(nproc)
echo "Maximum cores: $njobs"

while true; do
    case "$1" in
        -i | --input )  input="$2"; shift 2;;
        -o | --output ) output="$2"; shift 2;;
        -n | --name )   filename="$2"; shift 2;;
        -j | --jobs )   if (( "$2" < njobs )); then
                            njobs="$2";
                        else
                            echo "Not enough cores for $2 jobs, setting to maximum $njobs."; fi
                        shift 2;;
        -f | --force )  force_opt="-f"; shift ;;
        -h | --help )   show_help; exit 0 ;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

if [[ -z "$input" ]]; then
    echo "Input directory must be passed:"
    show_help
    exit 1
fi
if [[ -z "$output" ]]; then
    echo "Output filename must be passed:"
    show_help
    exit 1
fi

filelist="$(dirname -- "$(realpath "$output")")/list.txt"

if [[ -z "$force_opt" ]]; then
    if [[ -f $filelist ]]; then
        echo "Filelist $filelist already exists; pass -f or --force to overwrite."
        exit 1
    fi
    if [[ -f $output ]]; then
        echo "Output file $output already exists; pass -f or --force to overwrite."
        exit 1
    fi
fi

echo "Searching in $input for files named $filename..."
find "$input" -name "$filename" -type f > "$filelist"
echo "Found $(wc -l < "$filelist") files to merge. Merging..."

hadd -v 99 $force_opt -j "$njobs" "$output" @"$filelist"
exitcode=$?
if [[ $exitcode -ne 0 ]]; then
    echo "Merging failed."
    exit $exitcode
fi
echo "Merging completed."
