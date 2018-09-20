import itertools as it
import json
import os

import numpy as np
import tensorflow as tf

import libexperiment as lib
import logger
from sysid_utils import *


# load the policy and test, saves pickled array of "seg" dictionaries
# TODO factor out some of the common init code w/ train()
def clone_experts():

	rootdir, _ = os.path.split(__file__)
	spec_path = os.path.join(rootdir, "config.json")
	with open(spec_path) as f:
		multispec = lib.Spec(json.load(f))
	
	# TODO extremely fragile, must match expert_rewards.py
	n_experts = 64
	multispec["seed"] = list(1000 + np.arange(n_experts))

	logger.configure(dir=rootdir, format_strs=["stdout", "csv"])

	n_procs = os.cpu_count() // 2 - 1
	n_sysid_samples = 1
	#lib.test_multispec(multispec, rootdir, n_sysid_samples, n_procs)

	results = lib.load_test_results(multispec, rootdir)
	specs, segsegs = zip(*results)
	all_segs = list(it.chain.from_iterable(segsegs))
	# we do not actually need these advantage computations,
	# but seg_flatten_batches is expecting the results to be there
	for seg in all_segs:
		seg_flatten_batches(seg, keys=["ob", "ac", "logp"])
	assert len(all_segs) == n_sysid_samples * n_experts

	updates = {
		"flavor": "plain",
		"seed" : 0,
		"n_hidden": 2,
		"hidden_sz": 256,
	}
	clone_spec = lib.Spec({**specs[0], **updates})

	seed = clone_spec["seed"]
	np_random = np.random.RandomState(seed)
	# only used for its gym.Spaces
	env = lib.make_env(clone_spec)
	env._seed(seed)

	ob = lib.ndarray_key(all_segs, "ob")
	ac = lib.ndarray_key(all_segs, "ac")
	logp = lib.ndarray_key(all_segs, "logp")

	dim = lib.get_dim(clone_spec, env)
	ob_ph = tf.placeholder(tf.float32, (None, dim.ob_concat), "ob")
	ob_traj_ph = tf.placeholder(tf.float32, (None, dim.window, dim.ob), "ob_traj")
	ac_traj_ph = tf.placeholder(tf.float32, (None, dim.window, dim.ac), "ac_traj")

	with tf.variable_scope("pi_clone"):
		pi = lib.make_batch_policy_fn(np_random, clone_spec, env, test=False)(
			env.observation_space, env.action_space,
			ob_ph, ob_traj_ph, ac_traj_ph)

	expert_ac_ph = tf.placeholder(tf.float32, (None, dim.ac), "expert_ac")
	expert_logp_ph = tf.placeholder(tf.float32, (None), "expert_logp")
	kl_loss = tf.reduce_mean(expert_logp_ph - pi.logp(expert_ac_ph))
	mu_loss = tf.reduce_mean((expert_ac_ph - pi.ac_mean)**2)
	clone_loss = kl_loss#1e-3 * kl_loss + mu_loss

	#rate = clone_spec["learning_rate"]
	rate = 1e-3
	adam = tf.train.AdamOptimizer(rate)
	opt_op = adam.minimize(clone_loss, var_list=pi.policy_vars)

	epochs = 400
	batch = clone_spec["minibatch"]
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
			for exob, exac, exlogp in minibatch_iter(
				batch, ob, ac, logp, np_random=np_random):
				loss, losskl, lossmu,  _ = sess.run(
					[clone_loss, kl_loss, mu_loss, opt_op], {
					ob_ph : exob,
					expert_ac_ph : exac,
					expert_logp_ph : exlogp,
				})
			loss, losskl, lossmu,  _ = sess.run(
				[clone_loss, kl_loss, mu_loss, opt_op], {
				ob_ph : ob[::8],
				expert_ac_ph : ac[::8],
				expert_logp_ph : logp[::8],
			})
			logger.record_tabular("epoch", epoch)
			logger.record_tabular("kl_loss", losskl)
			logger.record_tabular("mu_loss", lossmu)
			logger.record_tabular("clone_loss", loss)
			logger.dump_tabular()


if __name__ == "__main__":
	clone_experts()
