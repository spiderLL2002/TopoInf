run_exp: 
	sh run_exp.sh

run_exp_tee: 
	sh run_exp.sh 2>&1 | tee run_exp_out.log

test_exp:
	python ../subexp-1-performance_curve/topoinf_train_model.py \
		--dataset cora \
		--model-list SGC \
		--n-runs 2 \
		--seed 2024 \
		--delete-strategy all_random \
		--delete-unit number \
		--delete-num-list 100 200 \
		--delete-mode-list pos neg \
		--n-epochs	100 \
		--early_stopping	50 \
		--perf-save-root-dir ./temp_test_dir \
		| tee temp_test_out.log