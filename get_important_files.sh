#!/bin/bash
declare -A dictionary
dictionary["data_partition_paper"]="1nFqNBgSAr0MBGxQbBIsvA_OLIh4KMG9S"
dictionary["results_rf_paper"]="1tfiUvdFyTSJp_HPdEaUYkxq-QHdOBK9b"
dictionary["results_paper"]="1teIi3GfPbYOZXaIHRAa_OCPTY9QYXTU1"
dictionary["dynamic_features"]="1tXqx7JTaMeO_sDBQ5RK4cLKw_1lUxZmn"
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