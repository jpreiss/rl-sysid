import itertools as it
import json
import os
import typing
from typing import List

import numpy as np
import tensorflow as tf

import libexperiment as lib
import logger
from sysid_utils import *


def clone_experts(
	experts_dir: lib.MultiSpecDir,
	multispec: lib.Spec, clone_spec: lib.Spec,
	epochs: int, save_path: lib.SaverDir):

	# load the trajectories collected from testing the experts
	results = lib.load_test_results(multispec, experts_dir)
	specs, segsegs = zip(*results)
	segs = list(it.chain.from_iterable(segsegs))

	# get the cloning target data into big flat arrays
	keys = ("ob", "ac", "logp", "acmean", "ep_rews")
	for seg in segs:
		seg_flatten_batches(seg, keys=keys)
	*data, ep_rew = [lib.ndarray_key(segs, key).astype(np.float32) for key in keys]

	if False:
		nan_checks = [np.isnan(d) for d in data]
		any_nan = np.any(np.column_stack(nan_checks), axis=1)
		step_ok = np.logical_not(any_nan)
		data = [d[step_ok] for d in data]
		print(f"{np.sum(any_nan)} steps contain NaN data")

	obs, acs, logps, acmeans = data
	nsteps = logps.size
	nbytes = sum(a.nbytes for a in data)
	print(f"training data: {nsteps:,} timesteps, {nbytes:,} bytes")
	print(f"expert rew: mean = {ep_rew.mean()}, std = {ep_rew.std()}")

	# env only used to construct dim (not great, yeah)
	seed = clone_spec["seed"]
	env = lib.make_env(clone_spec)
	env.seed(seed)
	dim = lib.get_dim(clone_spec, env)

	# set up tf graph for training the clone
	ob_ph = tf.placeholder(tf.float32, (None, dim.ob_concat), "ob")
	ac_ph = tf.placeholder(tf.float32, (None, dim.ac), "ac")
	logp_ph = tf.placeholder(tf.float32, (None,), "logp")
	acmean_ph = tf.placeholder(tf.float32, (None, dim.ac), "acmean")
	ob_traj_ph = tf.placeholder(tf.float32, (None, dim.window, dim.ob), "ob_traj")
	ac_traj_ph = tf.placeholder(tf.float32, (None, dim.window, dim.ac), "ac_traj")

	minibatch = clone_spec["minibatch"]
	dataset = tf.data.Dataset.from_tensor_slices((ob_ph, ac_ph, logp_ph, acmean_ph))
	dataset = dataset.shuffle(nsteps).batch(minibatch)
	iter = dataset.make_initializable_iterator()
	ob, expert_ac, expert_logp, expert_acmean = iter.get_next()

	np_random = np.random.RandomState(seed)
	with tf.variable_scope("pi"):
		pi = lib.make_batch_policy_fn(np_random, clone_spec, dim, test=False)(
			env.observation_space, env.action_space,
			ob, ob_traj_ph, ac_traj_ph)

	#expert_pdf = tf.distributions.Normal(expert_mu, tf.exp(expert_logstd))
	#kl_loss = tf.reduce_mean(expert_pdf.kl_divergence(pi.pdf.pdf))
	pi_logp = pi.logp(expert_ac)
	kl_loss = tf.reduce_mean(expert_logp - pi_logp)

	# diagnostics
	max_logp = tf.reduce_max(pi_logp)
	min_logp = tf.reduce_min(pi_logp)
	mean_logp, var_logp = tf.nn.moments(pi_logp, axes=[0])
	std_logp = tf.sqrt(var_logp)
	diagnostics = [max_logp, min_logp, mean_logp, std_logp]
	diag_names = ["max_logp", "min_logp", "mean_logp", "std_logp"]

	mu_loss = tf.reduce_mean((expert_acmean - pi.ac_mean)**2)
	entropy = tf.reduce_mean(pi.entropy)
	clone_loss = mu_loss
	clone_loss -= 0.1 * entropy

	rate = clone_spec["learning_rate"]
	print("clone lr:", rate)
	adam = tf.train.AdamOptimizer(rate)
	opt_op = adam.minimize(clone_loss, var_list=pi.policy_vars)

	# train the clone
	tf.set_random_seed(seed)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("beginning training")
		progress = lib.Progress(epochs * nsteps, print_every_percent=5)
		for epoch in range(epochs):
			sess.run(iter.initializer, {
				ob_ph: obs,
				ac_ph: acs,
				logp_ph: logps,
				acmean_ph: acmeans,
			})
			while True:
				try:
					loss, losskl, lossmu, ent,  _, *diag_vals = sess.run(
						[clone_loss, kl_loss, mu_loss, entropy, opt_op] + diagnostics)
					do_print, pct = progress.add(minibatch)
					if do_print:
						logger.record_tabular("% done", pct)
						logger.record_tabular("epoch", epoch)
						logger.record_tabular("kl_loss", losskl)
						logger.record_tabular("mu_loss", lossmu)
						logger.record_tabular("entropy", ent)
						logger.record_tabular("clone_loss", loss)
						for n, v in zip(diag_names, diag_vals):
							logger.record_tabular(n, v)
						logger.dump_tabular()
				except tf.errors.OutOfRangeError:
					break

		pi.save(sess, save_path)


def test_clone_on_expert_envs(
	specs: List[lib.Spec], clone_spec: lib.Spec, load_clone_path: lib.SaverDir,
	save_results_path: lib.TestPath, n_procs: int):

	os.makedirs(save_results_path, exist_ok=True)

	n_sysid_samples = 1  # only test on expert environments
	infer_sysid = False  # use the true SysID value instead of the estimate from ConvNet
	def arg_fn(spec):
		save_path = os.path.join(save_results_path, spec.dir(), lib.Spec.test_pickle_name)
		return clone_spec, load_clone_path, save_path, n_sysid_samples, infer_sysid
	lib.grid(specs, lib.test, arg_fn, n_procs)


def print_clone_expert_rews(testresult_dir: lib.MultiSpecDir, multispec: lib.Spec):

	results = lib.load_test_results(multispec, testresult_dir)
	specs, segsegs = zip(*results)
	all_segs = list(it.chain.from_iterable(segsegs))
	ep_rew = lib.ndarray_key(all_segs, "ep_rews").flatten()
	print(f"clone rew: mean = {ep_rew.mean()}, std = {ep_rew.std()}")


def main():

	rootdir, _ = os.path.split(__file__)
	spec_path = os.path.join(rootdir, "config.json")
	with open(spec_path) as f:
		multispec = lib.Spec(json.load(f))

	# n_seeds == n of environments tested
	n_seeds = 64
	multispec["seed"] = [1000 + int(a) for a in np.arange(n_seeds)]
	specs = lib.multispec_product(multispec)

	# number of processes to use for training & testing experts
	n_procs = os.cpu_count() // 2 - 1

	spec_updates = {
		"flavor": "embed",
		"n_hidden": 2,
		"hidden_sz": 256,
	}
	clone_spec = lib.Spec({**specs[0], **spec_updates, "seed": 0})

	# file system storage layout
	experts_dir = os.path.join(rootdir, "experts")
	clone_model_dir = os.path.join(rootdir, "clone_model")
	clone_test_dir = os.path.join(rootdir, "clone_tests_expert")
	finetune_model_dir = os.path.join(rootdir, "finetune_model")
	finetune_test_dir = os.path.join(rootdir, "finetune_tests")

	# train the experts
	for s in specs:
		s["exact_seed"] = s["seed"]
		s["seed"] = None
	#lib.train_multispec(multispec, experts_dir, n_procs, specs=specs)

	# test the experts
	n_procs = 32
	n_sysid_samples = 1
	for s in specs:
		s["n_batch"] = 32
	#lib.test_multispec(multispec, experts_dir, n_sysid_samples, n_procs, specs=specs)

	# train the clone
	epochs = 20
	#clone_experts(experts_dir, multispec, clone_spec, epochs, clone_model_dir)

	# test the clone on the training environments
	#test_clone_on_expert_envs(specs, clone_spec, clone_model_dir, clone_test_dir, n_procs)
	print("test results before finetune:")
	print_clone_expert_rews(clone_test_dir, multispec)

	# TODO: test the clone on some new environments

	# fine-tune the clone
	finetune_spec = lib.Spec({
		**clone_spec,
		"exact_seed": multispec["seed"],
		"seed": None,
		"train_iters": 100
	})

	lib.train(finetune_spec, finetune_model_dir, load_dir=clone_model_dir)

	# test fine-tuned clone
	test_clone_on_expert_envs(specs, finetune_spec, finetune_model_dir, finetune_test_dir, n_procs)
	print("test results after finetune:")
	print_clone_expert_rews(finetune_test_dir, multispec)


if __name__ == "__main__":
	main()
