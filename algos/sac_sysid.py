import itertools as it
import time

import numpy as np
import tensorflow as tf

import logger
from sysid_utils import MLP, ReplayBuffer, sysid_simple_generator, minibatch_iter


# TODO:
# scale by entropy coefficient?


class UniformPolicy(object):
    def __init__(self, low, high, seed):
        self.low = low
        self.high = high
        self.flavor = "plain"
        self.ob = tf.placeholder(tf.float32, (None, None))
        self.est_target = tf.constant(np.zeros((1,0)))
        dist = tf.distributions.Uniform(low, high)
        N = tf.shape(self.ob)[0]
        self.ac_stochastic = dist.sample(N, seed=seed)

        # TODO this is not the proper log prob of multi-var uniform dist!
        self.log_prob = tf.reduce_mean(dist.log_prob(self.ac_stochastic), axis=1)
        self.ac_mean = dist.mean()
        self.ac_logstd = tf.log(dist.stddev())


    def estimate_sysid(self, *args):
        raise NotImplementedError


def learn(
    sess,              # TF Session
    np_random,         # numpy.random.RandomState for determinism
    env,               # environment w/ OpenAI Gym API
    dim,               # Dimension namedtuple
    policy_func,       # func takes (ob space, ac space, ob input placeholder)
    vf_hidden,         # array of sizes of hidden layers in MLP learned value functions
    max_iters,         # stop after this many iters (defined by env.ep_len)
    logdir,            # directory to write csv logs
    tboard_dir,        # directory to write tensorboard summaries & graph

    learning_rate,     # ...
    init_explore_steps,# explore this many steps with random policy at start

    is_finetune,

    n_train_repeat,    # do this many gradient steps on replay buf each step
    buf_len,           # size of replay buf
    minibatch,         # size of per-step optimization minibatch
    TD_discount,       # ...
    reward_scale,      # SAC uses reward scaling instead of entropy weighting
    tau,               # exp. moving avg. speed for value fn estimator
    vf_grad_thru_embed,# whether to allow the value function's gradient to pass through the embedder or not

    adam_epsilon=1e-8,
    ):

    # set up so we do tensorboard and csv logging
    logvars = []
    def log_scalar(name, tfvar):
        tf.summary.scalar(name, tfvar)
        logvars.append((name, tfvar))

    # get dims
    N = env.N

    # placeholders
    ob_ph = tf.placeholder(tf.float32, (None, dim.ob_concat), "ob")
    ac_ph = tf.placeholder(tf.float32, (None, dim.ac), "ac")
    ob_traj_ph = tf.placeholder(tf.float32, (None, dim.window, dim.ob), "ob_traj")
    ac_traj_ph = tf.placeholder(tf.float32, (None, dim.window, dim.ac), "ac_traj")
    rew_ph = tf.placeholder(tf.float32, (None, ), "rew")
    nextob_ph = tf.placeholder(tf.float32, (None, dim.ob_concat), "next_ob")

    # construct policy, twice so we can reuse the embedder (if applicable)
    with tf.variable_scope("pi"):
        pi = policy_func(env.observation_space, env.action_space, ob_ph, ob_traj_ph, ac_traj_ph)
    with tf.variable_scope("pi", reuse=True):
        pi_nextob = policy_func(env.observation_space, env.action_space, nextob_ph, ob_traj_ph, ac_traj_ph)

    pi.sess_init(sess)
    pi_nextob.sess_init(sess)

    # policy's probability of own stochastic action
    with tf.variable_scope("log_prob"):
        log_pi = pi.log_prob

    meanent = tf.reduce_mean(pi.entropy)
    log_scalar("mean_entropy", meanent)

    # value function
    vf_in = pi.vf_input
    if not vf_grad_thru_embed:
        vf_in = tf.stop_gradient(vf_in)

    vf = MLP("myvf", vf_in, vf_hidden, 1, tf.nn.relu)

    # double q functions - these ones are used "on-policy" in the vf loss
    q_in = tf.concat([vf_in, pi.ac_stochastic], axis=1, name="q_in")
    qf1 = MLP("qf1", q_in, vf_hidden, 1, tf.nn.relu)
    qf2 = MLP("qf2", q_in, vf_hidden, 1, tf.nn.relu)
    qf_min = tf.minimum(qf1.out, qf2.out, name="qf_min")

    if len(pi.extra_rewards) > 0:
        extra_losses = -tf.add_n(pi.extra_rewards)
    else:
        extra_losses = 0.0

    # policy loss
    # TODO impose L2 regularization externally?
    with tf.variable_scope("policy_loss"):
        policy_kl_loss = tf.reduce_mean(log_pi - qf_min)
        pol_vars = set(pi.policy_vars)
        pi_reg_losses = tf.reduce_sum([v for v in
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if v in pol_vars])
        policy_loss = policy_kl_loss + pi_reg_losses + extra_losses
        log_scalar("policy_kl_loss", policy_kl_loss)

    for t, name in zip(pi.extra_rewards, pi.extra_reward_names):
        log_scalar(name, t)

    # value function loss
    with tf.variable_scope("vf_loss"):
        vf_loss = 0.5 * tf.reduce_mean((vf.out - tf.stop_gradient(qf_min - log_pi))**2)
        vf_loss += extra_losses
        log_scalar("vf_loss", vf_loss)

    # same q functions, but for the off-policy TD training
    qtrain_in = tf.concat([vf_in, ac_ph], axis=1)
    qf1_t = MLP("qf1", qtrain_in, vf_hidden, 1, tf.nn.relu, reuse=True)
    qf2_t = MLP("qf2", qtrain_in, vf_hidden, 1, tf.nn.relu, reuse=True)

    # target (slow-moving) vf, used to update Q functions
    vf_TDtarget = MLP("vf_target", pi_nextob.vf_input, vf_hidden, 1, tf.nn.relu)

    # q fn TD-target & losses
    with tf.variable_scope("TD_target"):
        rew_thisstep = reward_scale * rew_ph - pi.alpha_sysid * tf.stop_gradient(pi.estimator_loss)
        TD_target = tf.stop_gradient(rew_thisstep + TD_discount * vf_TDtarget.out)

    with tf.variable_scope("TD_loss1"):
        TD_loss1 = 0.5 * tf.reduce_mean((TD_target - qf1_t.out)**2)
        TD_loss1 += extra_losses
        log_scalar("TD_loss1", TD_loss1)

    with tf.variable_scope("TD_loss2"):
        TD_loss2 = 0.5 * tf.reduce_mean((TD_target - qf2_t.out)**2)
        TD_loss2 += extra_losses
        log_scalar("TD_loss2", TD_loss2)


    # training ops
    def make_opt(loss, vars):
        prefix, *_ = vars[0].name.split("/")
        name = prefix + "_adam"
        if False:
            print(name, "vars:")
            for v in vars:
                print(v.name)
        adam = tf.train.AdamOptimizer(learning_rate, epsilon=adam_epsilon, name=name)
        return adam.minimize(loss, var_list=vars)

    policy_opt_op = make_opt(policy_loss, pi.policy_vars)
    vf_opt_op = make_opt(vf_loss, vf.vars)
    qf1_opt_op = make_opt(TD_loss1, qf1.vars)
    qf2_opt_op = make_opt(TD_loss2, qf2.vars)

    train_ops = [policy_opt_op, vf_opt_op, qf1_opt_op, qf2_opt_op]
    vf_train_ops = train_ops[1:]

    # SysID estimator - does not use replay buffer
    if len(pi.estimator_vars) > 0:
        estimator_adam = tf.train.AdamOptimizer(learning_rate, epsilon=adam_epsilon, name="estimator_adam")
        estimator_opt_op = estimator_adam.minimize(pi.estimator_loss, var_list=pi.estimator_vars)
        log_scalar("estimator_loss_sqrt", tf.sqrt(pi.estimator_loss))
        train_ops += [estimator_opt_op]
    else:
        estimator_opt_op = None

    # ops to update slow-moving target vf
    with tf.variable_scope("vf_target_assign"):
        vf_target_moving_avg_ops = [
            tf.assign(target, (1 - tau) * target + tau * source)
            for target, source in zip(vf_TDtarget.vars, vf.vars)
        ]


    buf_dims = (dim.ob_concat, dim.ac, 1, dim.ob_concat,
        (dim.window, dim.ob), (dim.window, dim.ac))
    replay_buf = ReplayBuffer(buf_len, buf_dims)


    # init tf
    writer = None
    if tboard_dir is not None:
        writer = tf.summary.FileWriter(tboard_dir, sess.graph)
        summaries = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    sess.run(vf_target_moving_avg_ops)
    #sess.run([tf.assign(vtarg, v) for vtarg, v in zip(V_target.vars, V.vars)])

    TRAIN_NONE = 0
    TRAIN_VF = 1
    TRAIN_ALL = 2
    do_train = TRAIN_NONE

    n_trains = [0] # get ref semantics in callback closure. lol @python

    # update the policy every time step for high sample efficiency
    # note: closure around do_train
    def per_step_callback(locals, globals):

        ob = locals["ob"]
        ac = locals["ac"]
        rew = locals["rew"][:,None]
        ob_next = locals["ob_next"]
        ob_traj = locals["ob_traj"]
        ac_traj = locals["ac_traj"]
        # add all agents' steps to replay buffer
        replay_buf.add_batch(ob, ac, rew, ob_next, ob_traj, ac_traj)

        if do_train == TRAIN_NONE:
            return

        ops = vf_train_ops if do_train == TRAIN_VF else train_ops

        # gradient step
        for i in range(n_train_repeat):
            ot, at, rt, ot1, obtj, actj = replay_buf.sample(np_random, minibatch)
            feed_dict = {
                ob_ph: ot,
                ac_ph: at,
                rew_ph: rt[:,0],
                nextob_ph: ot1,
                ob_traj_ph: obtj,
                ac_traj_ph: actj,
            }
            # TODO get diagnostics
            sess.run(ops, feed_dict)
            sess.run(vf_target_moving_avg_ops)

            if (writer is not None) and (i == 0) and (n_trains[0] % 1e3 == 0):
                summary = sess.run(summaries, feed_dict)
                writer.add_summary(summary, i)

        n_trains[0] += 1


    explore_epochs = int(np.ceil(float(init_explore_steps) / (N * env.ep_len)))
    #print(f"random exploration stage: {explore_epochs} epochs...")

    if is_finetune:
        explore_policy = pi
        do_train = TRAIN_VF
    else:
        explore_policy = UniformPolicy(
            -np.ones(dim.ac), np.ones(dim.ac), seed=np_random.randint(100))
        explore_policy.dim = pi.dim
        do_train = TRAIN_NONE

    exploration_gen = sysid_simple_generator(sess,
        explore_policy, env, stochastic=True, test=False,
        callback=per_step_callback)

    for i, seg in enumerate(it.islice(exploration_gen, explore_epochs)):
        # callback does almost all the work
        do_train = replay_buf.size > 2000
        #print(f"exploration epoch {i} complete")

    do_train = TRAIN_ALL

    #print("begin policy rollouts")

    iters_so_far = 0
    tstart = time.time()

    logger.configure(dir=logdir, format_strs=['stdout', 'csv'])

    seg_gen = sysid_simple_generator(sess, pi, env,
        stochastic=True, test=False, callback=per_step_callback)

    tstart = time.time()
    for seg in seg_gen:
        tend = time.time()
        logger.record_tabular("IterSeconds", (tend - tstart))
        tstart = tend
        if max_iters and iters_so_far >= max_iters:
            #print("breaking due to max iters")
            break

        logger.log("********** Iteration %i ************"%iters_so_far)
        logger.record_tabular("__flavor__", pi.flavor) # underscore for sorting first
        logger.record_tabular("__alpha__", pi.alpha_sysid)
        logger.record_tabular("__algorithm", "SAC")

        def seg_flatten(seg):
            shape = [seg.shape[0] * seg.shape[1]] + list(seg.shape[2:])
            return seg.reshape(shape)

        ob, ac, rew = seg["ob"], seg["ac"], seg["task_rews"]
        ob_flat = seg_flatten(ob)

        # TODO: uniform exploration policy
        if replay_buf.size < init_explore_steps:
            continue

        # the estimator must be trained "on-policy" because we reward the policy 
        # for behaving in a way that makes estimation easier.
        # therefore, we don't use the replay buffer to train the estimator.
        if False and estimator_opt_op is not None:
            ob_traj = seg_flatten(seg["ob_traj"])
            ac_traj = seg_flatten(seg["ac_traj"])
            assert ob_traj.shape[0] == ob_flat.shape[0]
            #est_target = seg["est_true"]
            sum_mserr = 0.0
            count = 0
            for ob, obt, act in minibatch_iter(minibatch,
                ob_flat, ob_traj, ac_traj, np_random=np_random):
                estimator_loss, _ = sess.run([pi.estimator_loss, estimator_opt_op], {
                    pi.ob : ob,
                    pi.ob_traj : obt,
                    pi.ac_traj : act,
                })
                sum_mserr += estimator_loss
                count += 1
            logger.record_tabular("estimator_rmse", np.sqrt(sum_mserr / count))

        # get all the summary variables
        ot, at, rt, ot1, obtj, actj = replay_buf.sample(np_random, minibatch)
        feed_dict = {
            ob_ph: ot,
            ac_ph: at,
            rew_ph: rt[:,0],
            nextob_ph: ot1,
            ob_traj_ph: obtj,
            ac_traj_ph: actj,
        }
        logvals = sess.run([v for name, v in logvars], feed_dict)
        for (name, var), val in zip(logvars, logvals):
            logger.record_tabular(name, val)

        #logger.record_tabular("V_loss", V_loss_b)
        #logger.record_tabular("V_mean", np.mean(V_b.flat))
        #logger.record_tabular("Q1_rmse", Q1_rmse)
        #logger.record_tabular("Q2_rmse", Q2_rmse)
        #logger.record_tabular("Q_target_mean", np.mean(Q_target_b.flat))

        iters_so_far += 1
        ep_rew = np.sum(rew, axis=0)
        logger.record_tabular("EpRewMean", np.mean(ep_rew.flat))
        logger.record_tabular("EpRewBest", np.amax(ep_rew.flat))
        logger.record_tabular("EpRewWorst", np.amin(ep_rew.flat))
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        for i in range(N):
            logger.record_tabular("Env{}Rew".format(i), ep_rew[i])
        logger.dump_tabular()

    return pi
