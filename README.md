# "A Statistical Approach to Assessing Neural Network Robustness"

Here you can find a [PyTorch](http://pytorch.org/) implementation of [Adaptive Multilevel splitting](http://www.nowozin.net/sebastian/blog/multilevel-splitting.html) and code to reproduce the experiments in our paper, ["A Statistical Approach to Assessing Neural Network Robustness"](https://arxiv.org/abs/1811.07209), to appear at the 7th International Conference on Representation Learning (ICLR 2019).

## Instructions

For all experiments:
1. Install the Python package `seaborn`, and the latest versions of `NumPy`/`PyTorch`.
2. Install [Gurobi](https://www.gurobi.com). (This is required for the PLNN package to load, but is not actually used by the core code itself.)
3. Activate Gurobi license.

### Experiment 6.2 Ablation Study:

From the base directory:
	* `python -m exp_6_2.process_data`
	* `python -m exp_6_2.run_exp`
	* `python -m exp_6_2.plot_exp`

### Experiment 6.3 MNIST/CIFAR10:

From the base directory:
	* `python -m exp_6_3_mnist.train`
	* `python -m exp_6_3_mnist.run_baseline`
	* `python -m exp_6_3_mnist.run_exp`
	* `python -m exp_6_3_mnist.plot_exp`

Repeat for CIFAR10 by replacing `mnist` with `cifar10`.

### Experiment 6.3 CIFAR100:

This experiment uses a pretrained DenseNet contained in the repo.

From the base directory:
	* `python -m exp_6_3_cifar100.run_baseline`
	* `python -m exp_6_3_cifar100.run_exp`
	* `python -m exp_6_3_cifar100.plot_exp`

### Experiment 6.4 Robust Training:

From the base directory:
1. Train the network by the standard method:
	* `python -m exp_6_4.train --epochs 100 --starting_epsilon 0.01 --prefix mnist --schedule_length 50 --method 'baseline' --cuda_ids '0'`
2. Continue training by the robust method:
	* `python -m exp_6_4.train --epochs 100 --starting_epsilon 0.01 --prefix 'mnist_robustified' --schedule_length 50 --method 'robust' --cuda_ids '0' --model_snapshot './snapshots/mnist_baseline_batch_size_50_epochs_100_lr_0.001_opt_adam_real_time_False_seed_0_checkpoint_99.pth'`
3. Run and plot the experiment:
	* `python -m exp_6_4.run_exp`
	* `python -m exp_6_4.plot_exp`

Change `--cuda_ids` to your desired GPU card index.

## Pertinent files

* `/exp_6_2`
	* `plot_exp.py` Produces Figures 1 and 4 from the paper for Experiment 6.2.
	* `process_data.py` Converts the CollisionDetection property files into pickled objects and a summary file. Also produces naive MC estimates.
	* `run_exp.py` Runs AMLS on the 500 properties to produce results for Experiment 6.1 and 6.2.
* `/exp_6_3_mnist`
	* `plot_exp.py` Produces the MNIST panel of Figure 2.
	* `run_baseline.py` Produces naive MC estimates of adversarial properties of varying rareness.
	* `run_exp.py` Runs AMLS on adversarial properties varying the hyperparameters to produce results for Ex 6.3.
	* `train.py` Trains the simple feedforward classifier on MNIST.
* `/exp_6_4`
	
	*	`train.py` Trains the network using standard or robust training.
	* `plot_exp.py` Produces Figures 3 and 5.
* `/plnn` A modified version of the [PLNN package](https://github.com/oval-group/PLNN-verification) with some bug fixes for the latest PyTorch.
* `/convex_adversarial` A modified version of the [convex_adversarial package](https://github.com/locuslab/convex_adversarial).

Note that `exp_6_4` contain modified code from [here](https://github.com/locuslab/convex_adversarial/tree/master/examples)

The directories `/exp_6_3_cifar10` and `/exp_6_3_cifar100` are analogous to `exp_6_3_mnist`.

## Notes

* Please contact me if you would like to obtain the code for Section 6.1.
* The code to run the experiments was not completely rerun due to the need for computational resources, so contact me if you are having trouble with them.
* You may wish to modify the code so that it runs different combinations of hyperparamaters in parallel across multiple GPUs to reduce the total experiment time.

## TODO

* Experiment code for Section 6.4.