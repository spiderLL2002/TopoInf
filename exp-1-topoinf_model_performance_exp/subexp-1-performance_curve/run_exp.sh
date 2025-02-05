### `run_exp.sh` File ###
### TO RUN: `sh run_exp.sh` ###



python topoinf_train_model.py \
	--dataset cora \
	--model-list  GCN APPNP MLP  SGC  \
	--num-layers 3 \
	--delete-unit mode_ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--k-order 3 \
	--coefficients 0.0 0.0 1.0 \
	--topoinf-threshold 0.0005 \
	--output "output1/"\

python topoinf_train_model.py \
	--dataset pubmed \
	--model-list  APPNP  \
	--num-layers 3 \
	--delete-unit mode_ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--k-order 3 \
	--coefficients 0.0 0.0 1.0 \
	--topoinf-threshold 0.0005 \
	--output "output1/"\
	
:<< "a"


 


python topoinf_train_model.py \
	--dataset citeseer \
	--model-list GCN SGC APPNP MLP \
	--num-layers 3 \
	--delete-unit mode_ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--k-order 3 \
	--coefficients 0.0 0.0 1.0 \
	--topoinf-threshold 0.0005 \
	--output "output1/"\


python topoinf_train_model.py \
	--dataset photo \
	--model-list GCN SGC APPNP MLP \
	--num-layers 3 \
	--delete-unit mode_ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--k-order 3 \
	--coefficients 0.0 0.0 1.0 \
	--topoinf-threshold 0.0005 \
	--output "output1/"\



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
	--output "output1/"\

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
	--output "output1/"\


###################################
#      NUM TOTAL TASKS:    6      #
###################################
a