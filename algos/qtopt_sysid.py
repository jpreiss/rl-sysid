#from baselines.common import Dataset, explained_variance, fmt_row, zip
#import baselines.common.tf_util as U
#from baselines.common.mpi_adam import MpiAdam
#from baselines.common.mpi_moments import mpi_moments

from baselines.common.dataset import iterbatches
import baselines.common.batch_util2 as batch2
from baselines import logger

import tensorflow as tf
import numpy as np

import time


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def printstats(var, name):
    print("{}: mean={:3f}, std={:3f}, min={:3f}, max={:3f}".format(
        name, np.mean(var), np.std(var), np.min(var), np.max(var)))


class ReplayBuffer(object):
    def __init__(self, N, dims):
        N = int(N)
        self.bufs = tuple(np.zeros((N, d)) for d in dims)
        self.N = N
        self.size = 0
        self.cursor = 0

    def add(self, *args):
        if self.size < self.N:
            self.size += 1
        if self.cursor == 0:
            print("replay buffer roll over")
        for buf, item in zip(self.bufs, args):
            buf[self.cursor] = item
        self.cursor = (self.cursor + 1) % self.N

    def sample(self, np_random, batch_size):
        idx = np_random.randint(self.size, size=batch_size)
        returns = [buf[idx] for buf in self.bufs]
        return returns


def learn(np_random, env, q_func, *,
        learning_rate,
        target_update_iters,
        optim_epochs, optim_batchsize,# optimization hypers
        td_lambda, # TD backup (TODO: multi-step?)
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        logdir
        ):

    n_envs = env.N

    Q = q_func("Q", env.observation_space, env.action_space)
    Q_target = q_func("Q_target", env.observation_space, env.action_space)
    # TODO factor this out
    dim = Q.dim

    in_rew = tf.placeholder(tf.float32, (None,), name="reward")
    in_obnext = tf.placeholder(tf.float32, (None, dim.ob), name="ob_next")

    td_target = in_rew + td_lambda * Q_target.Q
    Qz = (Q.Q - Q.Q_rms.mean) / Q.Q_rms.std
    targetz = (td_target - Q.Q_rms.mean) / Q.Q_rms.std
    TD_err = tf.reduce_mean((Qz - tf.stop_gradient(targetz))**2)
    loss = tf.add_n([TD_err] + Q.extra_rewards)

    var_list = tf.trainable_variables()
    sysid_var_list = [v for v in var_list if v.name.split("/")[1].startswith("sysid")]

    """
    def n_params(vars):
        return np.sum([np.prod(v.shape) for v in vars])
    for list, name in (
        (var_list, "pol"),
        #(vf_var_list, "vf"),
        (sysid_var_list, "sysid")):
        print("{}: {} params".format(name, n_params(list)))
    """

    momentum = 0.9
    sgdm = tf.train.MomentumOptimizer(learning_rate, momentum)
    opt_op = sgdm.minimize(loss, var_list=var_list)

    # optimizer for SysID network
    # they are separate so we can use different learning rate schedules, etc.
    sgdm_sysid = tf.train.MomentumOptimizer(learning_rate, momentum)
    if len(sysid_var_list) > 0:
        opt_op_sysid = sgdm_sysid.minimize(Q.sysid_err_supervised, var_list=sysid_var_list)
    else:
        opt_op_sysid = tf.constant(0.0)

    assign_target_ops = [tf.assign(vtarget, v)
        for (vtarget, v) in zip(Q_target.get_variables(), Q.get_variables())]

    for vtarget, v in zip(Q_target.get_variables(), Q.get_variables()):
        print(f"{vtarget.name} <- {v.name}")

    # get ready
    sess = tf.Session()
    sess.__enter__() # reduce indent vs. with block
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./board', sess.graph)
    replay_buf = ReplayBuffer(1e6, (dim.ob_concat, dim.ac, 1, dim.ob_concat, dim.ac))

    seg_gen = batch2.sysid_simple_generator(Q, env, stochastic=True, test=False)

    episodes_so_far = 0
    timesteps_so_far = 0
    timesteps_since_last_episode_end = 0
    iters_so_far = 0
    tstart = time.time()

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0, max_seconds > 0]) == 1, \
        "Only one time constraint permitted"

    logger.configure(dir=logdir, format_strs=['stdout', 'csv'])

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            print("breaking due to max timesteps")
            break
        if max_episodes and episodes_so_far >= max_episodes:
            print("breaking due to max episodes")
            break
        if max_iters and iters_so_far >= max_iters:
            print("breaking due to max iters")
            break
        if max_seconds and time.time() - tstart >= max_seconds:
            print("breaking due to max seconds")
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(iters_so_far) / max_iters, 0.10)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        if iters_so_far % target_update_iters == 0:
            print("assigning to target network")
            sess.run(assign_target_ops)

        seg = seg_gen.__next__()

        # add to replay buffer
        horizon = seg["ob"].shape[0]
        for t in range(horizon - 1):
            for agent in range(n_envs):
                replay_buf.add(
                    seg["ob"][t,agent],
                    seg["ac"][t,agent],
                    seg["task_rews"][t,agent],
                    seg["ob"][t+1,agent],
                    seg["ac"][t+1,agent])

        # TODO should not need for TD Q learning
        batch2.add_vtarg_and_adv(seg, 0, 0)
        # flatten leading (agent, rollout) dims to one big batch
        # (note: must happen AFTER add_vtarg_and_adv)
        sysid_vals = seg["ob"][0,:,dim.ob:]
        batch2.seg_flatten_batches(seg)
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        ob_traj, ac_traj = seg["ob_traj"], seg["ac_traj"]
        rew = seg["task_rews"]
        embed_true = seg["embed_true"]

        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(Q, "ob_rms"): Q.ob_rms.update(ob) # update running mean/std for policy

        # do off-policy Q learning
        tot_TD_err = 0.0
        if replay_buf.size > 100000:
            print("optimizing Q")
        for _ in range(optim_epochs):
            ot, at, rt, ot1, at1 = replay_buf.sample(np_random, optim_batchsize)
            # do off-policy TD-update of Q
            _, tderr, Qvals, targets = sess.run([opt_op, TD_err, Q.Q, td_target], feed_dict={
                Q.in_ob : ot,
                Q.in_ac : at,
                in_rew : rt.flat,
                Q_target.in_ob : ot1,
                Q_target.in_ac : at1,
            })
            tot_TD_err += tderr
            mean_Q = np.mean(Qvals)
            print("mean Q =", mean_Q)
            mean_target = np.mean(targets)
            print("mean target =", mean_target)
            if hasattr(Q, "Q_rms"): Q.Q_rms.update(Qvals) # update running mean/std for policy

        # do on-policy (moving target) supervised learning of SysID network
        for obt, act, ob_ in iterbatches(
            (ob_traj, ac_traj, ob),
            batch_size=optim_batchsize):

            sess.run(opt_op_sysid, feed_dict= {
                Q.in_ob_traj : obt,
                Q.in_ac_traj : act,
                Q.in_ob : ob_
            })

        if False:
            meanlosses = [np.mean(loss) for loss in losses]
            logger.log(fmt_row(13, meanlosses))

            logger.log("Evaluating losses...")
            # compute and log the final losses after this round of optimization.
            # TODO figure out why we do this complicated thing
            # instead of passing in the whole dataset - 
            # maybe it's to avoid filling up GPU memory?
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses(
                    batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)

            meanlosses,_,_ = mpi_moments(losses, axis=0)
            logger.log(fmt_row(13, meanlosses))


            for (lossval, name) in zip(meanlosses, loss_names + ["SysID"]):
                logger.record_tabular("loss_"+name, lossval)


        # log some further information about this iteration
        #logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        episodes_so_far += n_envs
        rews = seg["ep_rews"]
        timesteps_so_far += sum(seg["ep_lens"])
        iters_so_far += 1
        logger.record_tabular("EpLenMean", np.mean(seg["ep_lens"]))
        logger.record_tabular("EpRewMean", np.mean(rews))
        logger.record_tabular("EpRewWorst", np.amin(rews))
        logger.record_tabular("EpThisIter", n_envs)
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("TotTDError", tot_TD_err)
        #print(rews.shape)
        assert len(rews) == n_envs
        for i in range(n_envs):
            logger.record_tabular("Env{}Rew".format(i), rews[i])
        logger.dump_tabular()

    return Q
