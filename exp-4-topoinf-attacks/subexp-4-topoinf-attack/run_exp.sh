### `run_exp.sh` File ###
### TO RUN: `sh run_exp.sh` ###

python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--delete-num-list 100 100 100 100 100 100 \
	--one-hot-mask train_mask \
	--entropy-aware True \
	--entropy-coefficient 0.02\

:<<"a"
python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 100 100 100 100 100 100 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 100 200 300 400 500 600 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 100 200 300 400 500 600 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 100 200 300 400 500 600 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 100 200 300 400 500 600 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 100 200 300 400 500 600 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 100 200 300 400 500 600 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 100 200 300 400 500 600 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 100 200 300 400 500 600 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 100 200 300 400 500 600 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 100 200 300 400 500 600 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 500 1000 1500 2000 2500 3000 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 500 1000 1500 2000 2500 3000 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 500 1000 1500 2000 2500 3000 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 500 1000 1500 2000 2500 3000 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 500 1000 1500 2000 2500 3000 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-num-list 500 1000 1500 2000 2500 3000 \
	--one-hot-mask train_mask val_mask \


###################################
#      NUM TOTAL TASKS:   18      #
###################################
a