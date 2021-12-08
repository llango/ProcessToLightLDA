#!/bin/bash

seeds=(
0
1
2
3
)
python chunk.py ./test_Out/docword.libsvm 4
for seed in ${seeds[*]}
do
./lightLDABin/dump_binary ./docword_$seed.libsvm ./docs/millitary.word_id.dict ./model_block_path/ $seed
done