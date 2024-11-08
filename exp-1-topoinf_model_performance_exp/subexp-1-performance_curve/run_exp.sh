### `exp_run.sh` File ###
### TO RUN: `sh exp_run.sh` ###

python topoinf_train_model.py \
	--dataset cora \
	--model-list GCN SGC APPNP MLP \
	--num-layers 3 \
	--delete-unit mode_ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--delete-rate-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
	--k-order 3 \
	--coefficients 0.0 0.0 1.0 \
	--topoinf-threshold 0.0005 \


python topoinf_train_model.py \
	--dataset citeseer \
	--model-list GCN SGC APPNP MLP \
	--num-layers 3 \
	--delete-unit mode_ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--delete-rate-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
	--k-order 3 \
	--coefficients 0.0 0.0 1.0 \
	--topoinf-threshold 0.0005 \


python topoinf_train_model.py \
	--dataset pubmed \
	--model-list GCN SGC APPNP MLP \
	--num-layers 3 \
	--delete-unit mode_ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--delete-rate-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
	--k-order 3 \
	--coefficients 0.0 0.0 1.0 \
	--topoinf-threshold 0.0005 \


python topoinf_train_model.py \
	--dataset photo \
	--model-list GCN SGC APPNP MLP \
	--num-layers 3 \
	--delete-unit mode_ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--delete-rate-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
	--k-order 3 \
	--coefficients 0.0 0.0 1.0 \
	--topoinf-threshold 0.0005 \


python topoinf_train_model.py \
	--dataset computers \
	--model-list GCN SGC APPNP MLP \
	--num-layers 3 \
	--delete-unit mode_ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--delete-rate-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
	--k-order 3 \
	--coefficients 0.0 0.0 1.0 \
	--topoinf-threshold 0.0005 \


python topoinf_train_model.py \
	--dataset actor \
	--model-list GCN SGC APPNP MLP \
	--num-layers 3 \
	--delete-unit mode_ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--delete-rate-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
	--k-order 3 \
	--coefficients 0.0 0.0 1.0 \
	--topoinf-threshold 0.0005 \


###################################
#      NUM TOTAL TASKS:    6      #
###################################
