datapath=/home/kjw/git-repo/SimpleNet/data/SimpleNet-Dataset-100
datasets=('only-a' 'only-b' 'only-c' 'only-d' 'only-e' 'only-f' 'only-g' 'only-h' 'only-i' 'only-j' 'only-k' )
datasets=('only-h')

dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 main.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_StitchingNet \
--log_project StitchingNet_Results \
--results_path results \
--run_name run \
net \
-b resnet18 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 50000 \
--embedding_size 256 \
--gan_epochs 2 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
dataset \
--batch_size 4 \
--resize 224 \
--imagesize 224 "${dataset_flags[@]}" StitchingNet  $datapath
