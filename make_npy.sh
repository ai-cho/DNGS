DATASET="nerfbusters-dataset"

for SCENE in "aloe" "art" "century" "flowers" "garbage" "picnic" "roses" ; do
    python make_npy.py --basedir /workspace/nerfbusters-dataset/${SCENE}

done