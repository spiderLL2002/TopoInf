### `run_exp.sh` File ###
### TO RUN: `sh run_exp.sh` ###


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 1 1 1 1 1 1 1 1 1 1  \
	--one-hot-mask train_mask \
	--point-num 1000\
	--edge-num 10000\
	--ratio-coefficient 1\
	--degree-coefficient 3\


<<: "a"
python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 1 1 1 1 1 1 1 1 1 1  \
	--one-hot-mask train_mask \
	--point-num 1500\
	--edge-num 20000\
	--ratio-coefficient 1\
	--degree-coefficient 3\


python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 1 1 1 1 1 1 1 1 1 1  \
	--one-hot-mask train_mask \
	--point-num 3000\
	--edge-num 40000\
	--ratio-coefficient 1\
	--degree-coefficient 3\


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 1 1 1 1 1 1 1 1 1 1  \
	--one-hot-mask train_mask val_mask \
	--point-num 1500\
	--edge-num 20000\
	--ratio-coefficient 1\
	--degree-coefficient 3\






python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 1 1 1 1 1 1 1 1 1 1  \
	--one-hot-mask train_mask \
	--point-num 1500\
	--edge-num 20000\
	--ratio-coefficient 1\
	--degree-coefficient 3\








python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 1 1 1 1 1 1 1 1 1 1  \
	--one-hot-mask train_mask \
	--point-num 3000\
	--edge-num 40000\
	--ratio-coefficient 1\
	--degree-coefficient 3\



python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 1 1 1 1 1 1 1 1 1 1  \
	--one-hot-mask train_mask \
	--point-num 3000\
	--edge-num 40000\
	--ratio-coefficient 1\
	--degree-coefficient 3




python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 1 1 1 1 1 1 1 1 1 1  \
	--one-hot-mask train_mask \
	--point-num 1000\
	--edge-num 10000\
	--ratio-coefficient 1\
	--degree-coefficient 3\


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode pos \
	--delete-rate-list 1 1 1 1 1 1 1 1 1 1  \
	--one-hot-mask train_mask \
	--point-num 1000\
	--edge-num 10000\
	--ratio-coefficient 1\
	--degree-coefficient 3\





###################################
#      NUM TOTAL TASKS:   18      #
###################################
