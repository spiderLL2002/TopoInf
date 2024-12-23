### `exp_1_run.sh` File ###
### TO RUN: `sh exp_1_run.sh` ###

python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0 \
	--one-hot-mask train_mask \

:<< "a"
python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset cora \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset citeseer \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model GCN \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model SGC \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask val_mask \


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask \


python topoinf_pseudo_label_train_model.py \
	--model APPNP \
	--dataset pubmed \
	--num-layers 3 \
	--delete-unit ratio \
	--delete-mode neg \
	--entropy-aware True \
	--entropy-coefficient 0.02 \
	--one-hot-mask train_mask val_mask \

