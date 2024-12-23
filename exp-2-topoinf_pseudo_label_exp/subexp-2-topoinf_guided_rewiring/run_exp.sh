### `run_exp.sh` File ###
### TO RUN: `sh run_exp.sh` ###



python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2  \
	--one-hot-mask train_mask \
	--point-num 2000\
	--edge-num 30000\

<<:
python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 2 2 2 2 2 2 2 2 2 2 2 2 2 2 \
	--one-hot-mask train_mask \
	--point-num 2000\
	--edge-num 30000\




python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 2 2 2 2 2 2 2 2 2 2 2 2 2 2 \
	--one-hot-mask train_mask \
	--point-num 2000\
	--edge-num 30000\



python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 2 2 2 2 2 2 2 2 2 2 2 2 2 2 \
	--one-hot-mask train_mask \
	--point-num 2500\
	--edge-num 50000\





python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 2 2 2 2 2 2 2 2 2 2 2 2 2 2 \
	--one-hot-mask train_mask val_mask \
	--point-num 2500\
	--edge-num 50000\


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 2 2 2 2 2 2 2 2 2 2 2 2 2 2 \
	--one-hot-mask train_mask \
	--point-num 2500\
	--edge-num 50000\



python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 2 2 2 2 2 2 2 2 2 2 2 2 2 2 \
	--one-hot-mask train_mask \
	--point-num 7000\
	--edge-num 100000\



python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 2 2 2 2 2 2 2 2 2 2 2 2 2 2 \
	--one-hot-mask train_mask \
	--point-num 7000\
	--edge-num 100000\



python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 2 2 2 2 2 2 2 2 2 2 2 2 2 2 \
	--one-hot-mask train_mask \
	--point-num 7000\
	--edge-num 100000



###################################
#      NUM TOTAL TASKS:   18      #
###################################
