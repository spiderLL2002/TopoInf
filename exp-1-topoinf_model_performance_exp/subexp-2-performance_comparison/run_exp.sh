### `exp_1_run.sh` File ###
### TO RUN: `sh exp_1_run.sh` ###

python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset cora \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode-list pos neg \
	--delete-strategy all_random \
	--k-order 3 \
	--topoinf-threshold 0.0005 \
	--output "output2/"\

python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset cora \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--k-order 3 \
	--topoinf-threshold 0.0005 \
	--output "output2/"\


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset cora \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode-list pos neg \
	--delete-strategy label \
	--k-order 3 \
	--topoinf-threshold 0.0005 \
	--output "output2/"\


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset citeseer \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode-list pos neg \
	--delete-strategy all_random \
	--k-order 3 \
	--topoinf-threshold 0.0005 \
	--output "output2/"\


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset citeseer \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--k-order 3 \
	--topoinf-threshold 0.0005 \
	--output "output2/"\


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset citeseer \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode-list pos neg \
	--delete-strategy label \
	--k-order 3 \
	--topoinf-threshold 0.0005 \
	--output "output2/"\


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset pubmed \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode-list pos neg \
	--delete-strategy all_random \
	--k-order 3 \
	--topoinf-threshold 0.0005 \
	--output "output2/"\


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset pubmed \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--k-order 3 \
	--topoinf-threshold 0.0005 \
	--output "output2/"\


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset pubmed \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode-list pos neg \
	--delete-strategy label \
	--k-order 3 \
	--topoinf-threshold 0.0005 \
	--output "output2/"\


###################################
#      NUM TOTAL TASKS:    9      #
###################################
