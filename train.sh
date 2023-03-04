# Training parameters
num_epochs=1
batch_size=64
lr=1e-3
weight_decay=1e-4
momentum=0.9

betas_lower=0.9
betas_upper=0.999
gamma=0.5
step_size=64
# Saving parameters
ckpt_save_path='./ckpts'
ckpt_prefix='cktp_epoch_'
ckpt_save_freq=10
report_path="./reports"
all_data_pathes="./data/tiny-imagenet-200/train/"
regex_for_category="\/data\/tiny-imagenet-200\/train\/(.*)\/images\/.*"


python train.py --batch-size $batch_size \
                --lr $lr \
                --weight-decay $weight_decay \
                --num-epochs $num_epochs \
                --ckpt-save-path $ckpt_save_path \
                --ckpt-prefix $ckpt_prefix \
                --ckpt-save-freq $ckpt_save_freq \
                --report-path $report_path \
                --all-data-pathes $all_data_pathes \
                --regex-for-category $regex_for_category \
                --momentum $momentum \
                --betas $betas_lower $betas_upper \
                --gamma $gamma \
                --step-size $step_size
