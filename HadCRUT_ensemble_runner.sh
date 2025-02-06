#!/bin/bash
syr=$1
eyr=$2
log_file=$3

config_file="./config_HadCRUT.yaml"

module load Python
source ./.venv/bin/activate

for ensemble in {1..200}; do
    echo "==============================="
    echo "Doing ensemble member $ensemble"
    ./main_HadCRUT.py -config $config_file -year_start $syr -year_stop $eyr -member $ensemble -variable "sst" -log_file "$log_file"
    echo "==============================="
    echo ""
    echo ""
done
