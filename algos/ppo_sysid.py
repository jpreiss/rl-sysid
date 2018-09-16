import itertools as it
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import logger
from sysid_utils import explained_variance, MLP, sysid_simple_generator, add_vtarg_and_adv, seg_flatten_batches, fmt_row


def learn(sess,
        np_random, env, dim, policy_func, *,
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation,
        max_iters,
        #max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        logdir
        ):

    #np.seterr(all="raise")

    # placeholders
    ob_ph = tf.placeholder(tf.float32, (None, dim.ob_concat), "ob")
    ac_ph = tf.placeholder(tf.float32, (None, dim.ac), "ac")
    ob_traj_ph = tf.placeholder(tf.float32, (None, dim.window, dim.ob), "ob_traj")
    ac_traj_ph = tf.placeholder(tf.float32, (None, dim.window, dim.ac), "ac_traj")
    adv_ph = tf.placeholder(tf.float32, (None,), "adv")
    ret_ph = tf.placeholder(tf.float32, (None,), "ret")

    # dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        ob_ph, ac_ph, adv_ph, ret_ph, ob_traj_ph, ac_traj_ph))
    batch = tf.cast(tf.shape(ob_ph)[0], tf.int64)
    batches = dataset.shuffle(batch).repeat(optim_epochs).batch(optim_batchsize)
    #batches = dataset.take(optim_batchsize).batch(optim_batchsize)
    batch_iter = batches.make_initializable_iterator()
    ob_b, ac_b, adv_b, ret_b, ob_traj_b, ac_traj_b = batch_iter.get_next()

    # construct policy computation graphs
    # old policy needed to define surrogate loss (see PPO paper)
    with tf.variable_scope("pi"):
        pi = policy_func(env.observation_space, env.action_space, ob_b, ob_traj_b, ac_traj_b)
    with tf.variable_scope("oldpi"):
        oldpi = policy_func(env.observation_space, env.action_space, ob_b, ob_traj_b, ac_traj_b)

    # learning rate and PPO loss clipping multiplier, updated with schedule
    #lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
    lrmult = tf.Variable(1.0)
    clip_param = clip_param * lrmult

    # KL divergence not actually part of PPO, only computed for logging
    meankl = tf.constant(0.0)#pi.policy_kl(oldpi)

    # policy entropy & regularization
    ent = pi.entropy
    meanent = tf.reduce_mean(ent)
    ent_rew = entcoeff * meanent

    # construct PPO's pessimistic surrogate loss (L^CLIP)
    #ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
    ratio = tf.exp(pi.logp(ac_b) - tf.stop_gradient(oldpi.logp(ac_b)))
    surr1 = ratio * adv_b
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_b
    pol_surr = tf.reduce_mean(tf.minimum(surr1, surr2))

    # value function
    vf_in = pi.vf_input
    vf_in = tf.stop_gradient(vf_in)
    vf = MLP("vf", vf_in, (256, 256, 256), 1, tf.nn.relu)
    vf_loss = tf.reduce_mean(tf.square(vf.out - ret_b))

    # total loss for reinforcement learning
    total_rew = pol_surr + ent_rew
    if len(pi.extra_rewards) > 0:
        total_rew += tf.add_n(pi.extra_rewards)
    total_loss = -total_rew

    # these losses are named so we can log them later
    losses = [pol_surr, ent_rew, vf_loss, meankl, meanent] + pi.extra_rewards
    loss_names = ["pol_surr", "ent_rew", "vf_loss", "kl", "ent"] + pi.extra_reward_names

    def n_params(vars):
        return np.sum([np.prod(v.shape) for v in vars])
    for list, name in (
        (pi.policy_vars, "pol"), (vf.vars, "vf"), (pi.estimator_vars, "estimator")):
        print("{}: {} params".format(name, n_params(list)))

    # gradient and Adam optimizer for policy
    adam_pi = tf.train.AdamOptimizer(lrmult * optim_stepsize, epsilon=adam_epsilon)
    opt_pi = adam_pi.minimize(total_loss, var_list=pi.policy_vars)

    # gradient and Adam optimizer for SysID network
    # they are separate so we can use different learning rate schedules, etc.
    if pi.estimator_vars:
        adam_sysid = tf.train.AdamOptimizer(lrmult * optim_stepsize, epsilon=adam_epsilon)
        opt_sysid = adam_sysid.minimize(pi.estimator_loss, var_list=pi.estimator_vars)
    else:
        opt_sysid = tf.no_op()

    adam_vf = tf.train.AdamOptimizer(lrmult * optim_stepsize, epsilon=adam_epsilon)
    opt_vf = adam_vf.minimize(vf_loss, var_list=vf.vars)

    assign_ops = [tf.assign(oldv, newv)
        for (oldv, newv) in zip(oldpi.all_vars, pi.all_vars)]

    writer = tf.summary.FileWriter('./board', sess.graph)
    sess.run(tf.global_variables_initializer())

    seg_gen = sysid_simple_generator(
        sess, pi, env, stochastic=True, test=False)

    logger.configure(dir=logdir, format_strs=['stdout', 'csv'])

    tstart = time.time()

    episodes_so_far = 0
    timesteps_so_far = 0

    for iter, seg in enumerate(it.islice(seg_gen, max_iters)):

        if callback: callback(locals(), globals())

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(iter) / max_iters, 0.05)
        else:
            raise NotImplementedError

        ob_flat = seg["ob"].reshape((-1, dim.ob_concat))
        vpred = sess.run(vf.out, { ob_b : ob_flat })
        seg["vpred"] = vpred.reshape(seg["task_rews"].shape)

        logger.log(f"********** Iteration {iter} ************")
        logger.record_tabular("__flavor__", pi.flavor) # underscore for sorting first

        add_vtarg_and_adv(seg, gamma, lam)
        # flatten leading (agent, rollout) dims to one big batch
        # (note: must happen AFTER add_vtarg_and_adv)
        sysid_vals = seg["ob"][0,:,dim.ob:]
        seg_flatten_batches(seg)
        ob, ac, adv, ret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        ob_traj, ac_traj = seg["ob_traj"], seg["ac_traj"]

        for k, v in seg.items():
            try:
                if np.any(np.isnan(v)):
                    print(f"nan in seg[{k}]")
            except TypeError:
                pass

        vpredbefore = seg["vpred"] # predicted value function before udpate
        adv = (adv - adv.mean()) / adv.std() # standardized advantage function estimate

        # copy current policy vars to "oldpi"
        ops = [assign_ops, batch_iter.initializer, tf.assign(lrmult, cur_lrmult)]

        sess.run(ops, {
            ob_ph : ob,
            ac_ph : ac,
            adv_ph : adv,
            ret_ph : ret,
            ob_traj_ph : ob_traj,
            ac_traj_ph : ac_traj,
        })

        while True:
            try:
                sess.run([opt_pi, opt_vf, opt_sysid])
            except tf.errors.OutOfRangeError:
                break

        # TODO eval per epoch
        logger.log(fmt_row(13, loss_names + ["SysID"]))
        lossvals = sess.run(losses, {
            ob_b : ob,
            ac_b : ac,
            adv_b : adv,
            ret_b : ret,
            ob_traj_b : ob_traj,
            ac_traj_b : ac_traj,
        })
        logger.log(fmt_row(13, lossvals))

        for (lossval, name) in zip(lossvals, loss_names + ["SysID"]):
            logger.record_tabular("loss_"+name, lossval)

        if False:
            embeddings_before_update = seg["est_true"]
            embeddings_after_update = sess.run(pi.est_target, {
                ob_ph : seg["ob"],
            })
            emb_delta = embeddings_after_update - embeddings_before_update
            printstats(abs(emb_delta), "abs(embeddings change after gradient step)")

        # log some further information about this iteration
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, ret))
        lens, rews = seg["ep_lens"], seg["ep_rews"]
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        logger.record_tabular("EpLenMean", np.mean(lens))
        logger.record_tabular("EpRewMean", np.mean(rews))
        logger.record_tabular("EpRewWorst", np.amin(rews))
        logger.record_tabular("EpThisIter", len(lens))
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        assert len(rews) == dim.agents
        for i in range(dim.agents):
            logger.record_tabular("Env{}Rew".format(i), rews[i])
        logger.dump_tabular()
