#!/bin/sh

main_dir=$(pwd)
model_file=$1
atom_coor_dir=$2
out_dir=$3
tmp_dir=$4




if [ -s $atom_coor_dir/$model_file ]; then
    awk '{print $2,$3,$4}' $atom_coor_dir/$model_file > $tmp_dir/$model_file
    ./main $tmp_dir/$model_file $out_dir/$model_file 3 3 fslices
else
    echo "File is empty"
fi

rm $tmp_dir/$model_file


