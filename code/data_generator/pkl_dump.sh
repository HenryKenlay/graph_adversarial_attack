#!/bin/bash

min_n=90
max_n=100
p=0.02
output_root=../../dropbox/data/components

# make output directory if it doesnt exist
if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

for n_comp in 1 2 3 4 5; do
    python gen_er_components.py \
        --save_dir $output_root \
        --max_n $max_n \
        --min_n $min_n \
        --num_graph 5000 \
        --p $p \
        --n_comp $n_comp
done
