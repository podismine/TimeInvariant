python run_class_finetuning.py --mixup 0.8 --shuffle_seed 4 --finetune checkpoints_small_mask0.15/checkpoint-399.pth


python -m torch.distributed.launch --nproc_per_node=6  run_mae_pretraining.py --batch_size 512  --pin_mem  --num_workers 3



python run_class_finetuning.py --mixup 0.8  --finetune checkpoints_pretrain_mask25_lambda10/checkpoint-499.pth  --epochs 100 --shuffle_seed 8


python -m torch.distributed.launch --nproc_per_node=6  run_mae_pretraining.py --batch_size 512  --pin_mem  --num_workers 3 --output_dir checkpoints_pretrain_mask25_lambda10 --log_dir logs_pretrain_mask25_lambda10