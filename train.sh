### model settings ###
qwen=Qwen/Qwen2.5-7B-Instruct
llama=meta-llama/Llama-3.2-3B-Instruct

#### Stage 1 ####
pretrained=None
batchsize=16
run_name=stage1_pretrain
ver=plain
acc_step=1

sh scripts/train/stage1_pretrain.sh $llama $pretrained $batchsize $run_name $ver $acc_step


#### Stage 2 ####
pretrained=/pretrained_path
batchsize=1
run_name=stage2_longv
ver=plain
acc_step=1

sh scripts/train/stage2_longv.sh $llama $pretrained $batchsize $run_name $ver $acc_step

#### Stage 3 ####
pretrained=None
batchsize=4
run_name=stage3_llama
ver=llama_3_2 # or qwen_2_5
acc_step=3

sh scripts/train/stage3_llama.sh $llama $pretrained $batchsize $run_name $ver $acc_step