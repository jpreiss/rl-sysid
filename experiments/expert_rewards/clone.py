import itertools as it
import json
import os
import typing

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
	keys = ("ob", "ac", "logp", "acmean", "aclogstd", "ep_rews")
	for seg in segs:
		seg_flatten_batches(seg, keys=keys)
	data = (lib.ndarray_key(segs, key).astype(np.float32) for key in keys)
	obs, acs, logps, acmeans, aclogstds, ep_rew = data
	nbytes = sum(a.nbytes for a in data)
	print(f"training data: {logps.size:,} timesteps, {nbytes:,} bytes")
	print(f"expert rew: mean = {ep_rew.mean()}, std = {ep_rew.std()}")

	# env only used to construct dim (not great, yeah)
	seed = clone_spec["seed"]
	env = lib.make_env(clone_spec)
	env.seed(seed)
	dim = lib.get_dim(clone_spec, env)

	# set up tf graph for training the clone
	dataset = tf.data.Dataset.from_tensor_slices((obs, acs, logps, acmeans, aclogstds))
	dataset = dataset.shuffle(logps.size).batch(clone_spec["minibatch"])
	iter = dataset.make_initializable_iterator()
	ob, expert_ac, expert_logp, expert_mu, expert_logstd = iter.get_next()

	ob_traj_ph = tf.placeholder(tf.float32, (None, dim.window, dim.ob), "ob_traj")
	ac_traj_ph = tf.placeholder(tf.float32, (None, dim.window, dim.ac), "ac_traj")

	np_random = np.random.RandomState(seed)
	with tf.variable_scope("pi"):
		pi = lib.make_batch_policy_fn(np_random, clone_spec, dim, test=False)(
			env.observation_space, env.action_space,
			ob, ob_traj_ph, ac_traj_ph)

	#expert_pdf = tf.distributions.Normal(expert_mu, tf.exp(expert_logstd))
	#kl_loss = tf.reduce_mean(expert_pdf.kl_divergence(pi.pdf.pdf))
	kl_loss = tf.reduce_mean(expert_logp - pi.logp(expert_ac))

	mu_loss = tf.reduce_mean((expert_ac - pi.ac_mean)**2)
	entropy = tf.reduce_mean(pi.entropy)
	#clone_loss = kl_loss# - 0.2 * entropy
	clone_loss = kl_loss

	rate = clone_spec["learning_rate"]
	print("clone lr:", rate)
	adam = tf.train.AdamOptimizer(rate)
	opt_op = adam.minimize(clone_loss, var_list=pi.policy_vars)

	# train the clone
	tf.set_random_seed(seed)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
			logger.record_tabular("epoch", epoch)
			sess.run(iter.initializer)
			while True:
				try:
					loss, losskl, lossmu, ent,  _ = sess.run(
						[clone_loss, kl_loss, mu_loss, entropy, opt_op])
				except tf.errors.OutOfRangeError:
					break
			logger.record_tabular("kl_loss", losskl)
			logger.record_tabular("mu_loss", lossmu)
			logger.record_tabular("entropy", ent)
			logger.record_tabular("clone_loss", loss)
			logger.dump_tabular()

		pi.save(sess, save_path)


def test_clone_on_expert_envs(rootdir: lib.MultiSpecDir, multispec: lib.Spec, spec_updates: dict,
	load_clone_path: lib.SaverDir, save_results_path: lib.TestPath, n_procs: int):

	specs = lib.multispec_product(multispec)
	os.makedirs(save_results_path, exist_ok=True)

	n_sysid_samples = 1  # only test on expert environments
	def arg_fn(spec):
		clone_spec = lib.Spec({**spec, **spec_updates})
		save_path = os.path.join(save_results_path, spec.dir(), lib.Spec.test_pickle_name)
		return clone_spec, load_clone_path, save_path, n_sysid_samples
	lib.grid(specs, lib.test, arg_fn, n_procs)


def print_clone_expert_rews(testresult_dir: lib.MultiSpecDir, multispec: lib.Spec):

	results = lib.load_test_results(multispec, testresult_dir)
	specs, segsegs = zip(*results)
	all_segs = list(it.chain.from_iterable(segsegs))
	ep_rew = lib.ndarray_key(all_segs, "ep_rews").flatten()

	print("clone performance on training envs:")
	print(f"clone rew: mean = {ep_rew.mean()}, std = {ep_rew.std()}")


def fine_tune_clone(
	load_path: lib.SaverDir, save_path: lib.SaverDir,
	multispec: lib.Spec, clone_spec: lib.Spec):

	assert len(multispec["seed"]) == 1
	n_procs = 1
	multispec = multispec.strip_singleton_lists()
	multispec.assert_is_scalar()
	lib.train(multispec, save_path, load_dir=load_path)



def main():

	rootdir, _ = os.path.split(__file__)
	spec_path = os.path.join(rootdir, "config.json")
	with open(spec_path) as f:
		multispec = lib.Spec(json.load(f))

	# n_seeds == n of environments tested
	n_seeds = 64
	multispec["seed"] = list(1000 + np.arange(n_seeds))
	specs = lib.multispec_product(multispec)

	# number of processes to use for training & testing experts
	n_procs = os.cpu_count() // 2 - 1

	spec_updates = {
		"flavor": "plain",
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
	#lib.train_multispec(spec, experts_dir, n_procs)

	# test the experts
	#n_sysid_samples = 1
	#multispec["n_batch"] = 256
	#lib.test_multispec(multispec, experts_dir, n_sysid_samples, n_procs)

	multispec["seed"] = multispec["seed"][:1]

	# train the clone
	#epochs = 2
	#clone_experts(experts_dir, multispec, clone_spec, epochs, clone_model_dir)

	# test the clone on the training environments
	#test_clone_on_expert_envs(rootdir, multispec, spec_updates,
		#clone_model_dir, clone_test_dir, n_procs)
	#print("test results before finetune:")
	#print_clone_expert_rews(clone_test_dir, multispec)

	# TODO: test the clone on some new environments

	# fine-tune the clone
	aaah = lib.Spec({**multispec, **spec_updates})
    aaah["train_iters"] = 20
	fine_tune_clone(clone_model_dir, finetune_model_dir, aaah, clone_spec)

	# test fine-tuned clone
	test_clone_on_expert_envs(rootdir, multispec, spec_updates,
		finetune_model_dir, finetune_test_dir, n_procs)
	print_clone_expert_rews(finetune_test_dir, multispec)


if __name__ == "__main__":
	main()
