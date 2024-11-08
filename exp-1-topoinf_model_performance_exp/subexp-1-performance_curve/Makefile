run_exp: 
	sh run_exp.sh

run_exp_tee: 
	sh run_exp.sh 2>&1 | tee run_exp_out.log

test_exp:
	python topoinf_train_model.py \
		--dataset actor \
		--model-list GCN SGC \
		--n-runs 2 \
		--seed 2023 \
		--delete-rate-list  0.25 0.5 0.75 1 \
		--n-epochs	100 \
		--early_stopping	50 \
		--perf-save-root-dir ./temp_test_dir \
		| tee temp_test_out.log

