run_exp: 
	sh run_exp_2.sh

run_exp_tee: 
	sh run_exp_2.sh 2>&1 | tee run_exp_out.log

test_exp:
	python topoinf_pseudo_label_guided_dropedge.py \
		--seed 2024 \
		--dataset cora \
		--model GCN \
		--n-epochs	100 \
		--early_stopping	50 \
		--perf-save-root-dir ./temp_test_dir \
		| tee temp_test_out.log

test_specified_model:
	python topoinf_pseudo_label_guided_dropedge.py \
		--model GCN \
		--num-layers 2 \
		--k-order 3 \
		--coefficients aver \
		--lr 0.005 \
		--dataset cora \
		--dropout 0.0 \
		--dropedge_rate 0.4 \
		--dropedge-temperature 0.5 \
		--one-hot-mask train_mask \
		--perf-save-root-dir ./temp_test_dir \
		| tee temp_test_out.log
