### `exp_1_run.sh` File ###
### TO RUN: `sh exp_1_run.sh` ###

python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset cora \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit number \
	--delete-mode-list pos neg \
	--delete-strategy all_random \
	--delete-num-list 100 200 300 400 500 600 \
	--k-order 3 \
	--topoinf-threshold 0.0005 \


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset cora \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit number \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--delete-num-list 100 200 300 400 500 600 \
	--k-order 3 \
	--topoinf-threshold 0.0005 \


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset cora \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit number \
	--delete-mode-list pos neg \
	--delete-strategy label \
	--delete-num-list 100 200 300 400 500 600 \
	--k-order 3 \
	--topoinf-threshold 0.0005 \


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset citeseer \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit number \
	--delete-mode-list pos neg \
	--delete-strategy all_random \
	--delete-num-list 100 200 300 400 500 600 \
	--k-order 3 \
	--topoinf-threshold 0.0005 \


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset citeseer \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit number \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--delete-num-list 100 200 300 400 500 600 \
	--k-order 3 \
	--topoinf-threshold 0.0005 \


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset citeseer \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit number \
	--delete-mode-list pos neg \
	--delete-strategy label \
	--delete-num-list 100 200 300 400 500 600 \
	--k-order 3 \
	--topoinf-threshold 0.0005 \


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset pubmed \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit number \
	--delete-mode-list pos neg \
	--delete-strategy all_random \
	--delete-num-list 500 1000 1500 2000 2500 3000 \
	--k-order 3 \
	--topoinf-threshold 0.0005 \


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset pubmed \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit number \
	--delete-mode-list pos neg \
	--delete-strategy topoinf \
	--delete-num-list 500 1000 1500 2000 2500 3000 \
	--k-order 3 \
	--topoinf-threshold 0.0005 \


python ../subexp-1-performance_curve/topoinf_train_model.py \
	--dataset pubmed \
	--model-list GCN SGC APPNP \
	--num-layers 3 \
	--delete-unit number \
	--delete-mode-list pos neg \
	--delete-strategy label \
	--delete-num-list 500 1000 1500 2000 2500 3000 \
	--k-order 3 \
	--topoinf-threshold 0.0005 \


###################################
#      NUM TOTAL TASKS:    9      #
###################################
