SHELL = /bin/bash

default: results

J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace

synthetic: samples jitter noise dropout warping

allen: decoding

samples:
	$(JN) figure_2_num_samples.ipynb
	git commit -m 'results : samples' figure_2_num_samples.ipynb
	git push
multi:
	$(JN) results_num_patterns.ipynb
	git commit -m 'results : multiple patterns' results_num_patterns.ipynb
	git push
noise:
	$(JN) figure_3_background_noise.ipynb
	git commit -m 'results : additive noise' figure_3_background_noise.ipynb
	git push
warping:
	$(JN) figure_3_warping.ipynb
	git commit -m 'results : warping' figure_3_warping.ipynb
	git push
jitter:
	$(JN) figure_3_jitter.ipynb
	git commit -m 'results : jitter' figure_3_jitter.ipynb
	git push
dropout:
	$(JN) figure_3_dropout.ipynb
	git commit -m 'results : dropout' figure_3_dropout.ipynb
	git push
allen: 
	$(JN) annexe_allendataset_results_id_vs_ood.ipynb
	git commit -m 'results : allen ' annexe_allendataset_results_id_vs_ood.ipynb
	git push
decoding: 
	$(JN) figure_4_allendataset_decoding.ipynb
	git commit -m 'results : allen ' figure_4_allendataset_decoding.ipynb
	git push