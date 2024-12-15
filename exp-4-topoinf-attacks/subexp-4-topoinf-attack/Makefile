run_exp: 
	sh run_exp.sh

run_exp_tee: 
	sh run_exp.sh 2>&1 | tee run_exp_out.log

test_exp:
	python topoinf_pseudo_label_train_model.py \
		--dataset cora \
		--model GCN \
		--n-runs 2 \
		--delete-num-list  300 \
		--n-epochs	100 \
		--early_stopping	50 \
		--perf-save-root-dir ./temp_test_dir \
		| tee temp_test_out.log
