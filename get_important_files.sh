#!/bin/bash
declare -A dictionary
dictionary["data_partition_paper"]="1z7TNKzTmPeTR2kJA8kDg7c0KGyn-UrBi"
dictionary["results_rf_paper"]="1FmXMEDezyeBAGVcG9gVjqsxo_1IVmfKq"
dictionary["results_paper"]="1GZmfvDBjwZhD3J_8JYKyonUIHUCofWef"
dictionary["dynamic_features"]="1IPNoHHtokHy6DIq_6IWwuOxrrqfSFfHO"
dictionary["data_original_fits/FULL_ELASTICC_TRAIN"]="1ci53lW7n1ccyPgQ3pMToH-gAfCZqAplW"

FILEID=${dictionary[$1]}
echo $FILEID

OUTFILE="$1.zip"  # Ensure the variable is enclosed in quotes

if [[ "$1" == "data_original_fits/FULL_ELASTICC_TRAIN" ]]; then
    mkdir -p data_original_fits
fi

# Ensure the URL and options are correctly quoted
gdown "https://drive.google.com/uc?id=$FILEID" -O "$OUTFILE"

# Ensure the python command argument is correctly quoted
python unzip.py "$OUTFILE"