env:
	pip install -r requirements.txt

train-mnist:
	python -m tcuq.scripts.train_backbone --cfg configs/mnist.yaml

train-cifar10:
	python -m tcuq.scripts.train_backbone --cfg configs/cifar10.yaml

fit-head:
	python -m tcuq.scripts.fit_tcuq_head --cfg configs/tcuq_head.yaml

eval-accdrop:
	python -m tcuq.scripts.eval_accdrop --cfg configs/eval_accdrop.yaml

eval-failure:
	python -m tcuq.scripts.eval_failure --cfg configs/eval_failure.yaml

eval-calib:
	python -m tcuq.scripts.eval_calibration --cfg configs/cifar10.yaml

plots:
	python -m tcuq.scripts.plot_curves --in outputs/metrics.jsonl --out outputs/figs/

ci:
	pytest -q
