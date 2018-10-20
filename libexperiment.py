import collections
import contextlib
from copy import deepcopy
import hashlib
import itertools as it
import json
import multiprocessing
import os
import pdb
import pickle
import shutil
import subprocess
import sys
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

import numpy as np
import scipy as sp
import scipy.stats
import gym

import algos
from sysid_batch_policy import SysIDPolicy
# from sysid_batch_Q import SysIDQ # TODO: bring up to date
from sysid_utils import Dim, sysid_simple_generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf  # noqa 402


# JSON schema for fully defining experiments
experiment_params = {
    "env": "HalfCheetah-Batch-v1",
    "ep_len": 500,
    "n_batch": 8,
    "n_total": 256,
    "randomness": 1.75,
    "test_mode": "resample",  # one of "resample" or "tweak"
    "test_tweak": 1.1,
    "flavor": ["blind", "plain", "embed"],
    # "flavor": "blind",
    "alpha_sysid": [0.0, 0.01],
    "seed": [0, 1, 2],
    "algorithm": "sac",
    "lr_schedule": "constant",
    "vf_grad_thru_embed": False,
    "TD_discount": 0.99,
}

policy_params = {
    "embed_dim": 32,
    "window": 16,
    "embed_KL_weight": 0.1,
    "n_hidden": 2,
    "hidden_sz": 128,
    "activation": "relu",
}

ppo_params = {
    "learning_rate": 1e-3,
    "opt_iters": 4,
    "minibatch": 256,
    "entropy_coeff": 0.010,
    "train_iters": 400,
}

sac_params = {
    "learning_rate": 1e-3,
    "reward_scale": 5.0,
    "tau": 0.005,
    "init_explore_steps": int(1e3),
    "n_train_repeat": 2,
    "buf_len": int(1e5),
    "minibatch": 256,
    "train_iters": 200,
}


# represents exactly one experiment - one seed, policy type, etc.
# mainly exists to centralize all decisions about how to store results on disk
class Spec(dict):
    def __init__(self, specdict={}, **kwargs):
        super().__init__(**{**specdict, **kwargs})

    def hash(self) -> str:
        blob = json.dumps(self, sort_keys=True).encode("utf-8")
        return str(hashlib.md5(blob).hexdigest()[:8])

    def copy(self):
        return type(self)(specdict=self)

    def assert_is_scalar(self):
        assert len(self.multi_items()) == 0

    def multi_items(self):
        list_keys = ["exact_seed"]
        def is_multi(k, v):
            return type(v) == list and k not in list_keys
        return {k: v for k, v in self.items() if is_multi(k, v)}

    def strip_singleton_lists(self):
        s = self.copy()
        for k, v in s.items():
            if type(v) == list and len(v) == 1:
                s[k] = v[0]
        return Spec(s)

    def canonical_seed(self):
        self.assert_is_scalar()
        if "exact_seed" in self:
            assert self["seed"] == None
            s = self["exact_seed"]
            if type(s) is int:
                return s
            return s[0]
        else:
            return self["seed"]

    def dir(self):
        return self["directory"]

    csv_name = "progress.csv"  # TODO pass as arg to logging
    json_name = "config.json"
    saver_name = "trained_model.ckpt"
    test_pickle_name = "test_results.pickle"


MultiSpecDir = typing.NewType("MultiSpecDir", str)
SaverDir = typing.NewType("SaverDir", str)
TestPath = typing.NewType("TestPath", str)


# default multispec
_algo = typing.cast(str, experiment_params["algorithm"])
multispec = Spec(kwargs={
    **experiment_params,
    **policy_params,
    **({
        "sac": sac_params,
        "ppo": ppo_params,
    }[_algo])
})


# takes dict where some fields may be arrays,
# returns cartesian product of all array values as list of scalar specs
# adds "directory" field to scalar specs giving each a unique subpath
def multispec_product(spec: Spec) -> List[Spec]:
    def rec(spec: Spec, prefix: str):
        for k, v in sorted(spec.items()):
            # recursive case
            if type(v) == list:
                for val in v:
                    spec2 = deepcopy(spec)
                    spec2[k] = val
                    kvname = f"{k}_{val}"
                    yield from rec(spec2, os.path.join(prefix, kvname))
                return
        # base case
        spec.assert_is_scalar()
        spec["directory"] = prefix
        yield spec
    return list(rec(spec, ""))


# returns Spec with list values for any key with varying values across input Specs
def multispec_union(specs: List[Spec]) -> Spec:
    for spec in specs:
        spec.assert_is_scalar()
    keys: Set[str] = set()
    keys = keys.union(*it.chain(list(s.keys()) for s in specs))
    s = {}
    for k in keys:
        vals = set(s[k] for s in specs if k in s)
        if len(vals) == 1:
            s[k] = vals.pop()
        else:
            s[k] = sorted(list(vals))
    return Spec(s)


# our wrapper allowing to pass extra args to the batch environment ctors.
def make_env(spec, **kwargs) -> gym.Env:

    if "exact_seed" in spec:
        assert spec["seed"] is None
        seeds = deepcopy(spec["exact_seed"])
        if type(seeds) is int:
            seeds = [seeds]
    else:
        npr = np.random.RandomState(spec["seed"])
        seeds = [npr.randint(100000) for _ in range(spec["n_total"])]

    kwargs = deepcopy(kwargs)
    kwargs["seeds"] = seeds
    for key in ["n_batch", "randomness", "ep_len"]:
        kwargs[key] = spec[key]
    gym_spec = gym.envs.registry.env_specs[spec["env"]]
    return gym_spec.make(**kwargs)


def get_dim(spec: Spec, env: gym.Env) -> Dim:
    ob_space = env.observation_space
    ac_space = env.action_space
    sysid_dim = int(env.sysid_dim)
    dim = Dim(
        ob=ob_space.shape[0] - sysid_dim,
        sysid=sysid_dim,
        ob_concat=ob_space.shape[0],
        ac=ac_space.shape[0],
        embed=spec["embed_dim"],
        agents=env.N,
        window=spec["window"],
    )
    return dim


# for compatibility with OpenAI Baselines learning algorithms
def make_batch_policy_fn(np_random: np.random.RandomState,
    spec: Spec, dim: Dim, test: bool, load_dir: Optional[SaverDir]=None):

    spec.assert_is_scalar()
    flavor = spec["flavor"]
    alpha_sysid = spec["alpha_sysid"]
    logstd_is_fn = spec["logstd_is_fn"] and spec["algorithm"] != "ppo"

    activation = {"relu": tf.nn.relu, "selu": tf.nn.selu}[spec["activation"]]

    def f(ob_space, ac_space, ob_input, ob_traj_input, ac_traj_input):
        for space in (ob_space, ac_space):
            assert isinstance(space, gym.spaces.Box)
            assert len(space.shape) == 1
        if spec["algorithm"] in ["qtopt"]:
            raise NotImplementedError("adapt to not passing in name anymore.")
            return SysIDQ(
                name=name, flavor=flavor, dim=dim,
                hid_size=spec["hidden_sz"], n_hid=spec["n_hidden"],
                alpha_sysid=alpha_sysid)
        else:
            hid_sizes = [spec["hidden_sz"]] * spec["n_hidden"]
            embed_middle_size = 2 * int(np.sqrt(dim.sysid * dim.embed))
            embed_hid_sizes = [embed_middle_size]
            #embed_hid_sizes = []
            return SysIDPolicy(
                ob_input, ob_traj_input, ac_traj_input, flavor=flavor, dim=dim,
                hid_sizes=hid_sizes, embed_hid_sizes=embed_hid_sizes, activation=activation,
                alpha_sysid=alpha_sysid, logstd_is_fn=logstd_is_fn,
                embed_KL_weight=spec["embed_KL_weight"],
                squash=True, embed_tanh=spec["embed_tanh"],
                embed_stochastic=spec["embed_stochastic"],
                seed=np_random.randint(100),
                test=test,
                load_dir=load_dir,
            )
    return f


def train(spec: Spec, save_dir: SaverDir, load_dir: Optional[SaverDir]=None):

    spec.assert_is_scalar()

    env = make_env(spec)
    env.seed()
    seed = spec.canonical_seed()
    var_init_npr = np.random.RandomState(seed)

    dim = get_dim(spec, env)

    policy_fn = make_batch_policy_fn(var_init_npr, spec, dim, test=False, load_dir=load_dir)

    mydir, _ = os.path.split(save_dir)
    os.makedirs(mydir, exist_ok=True)

    g = tf.Graph()
    g.seed = seed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    opt = spec["optimizer"].lower()
    if opt == "adam":
        opt_fn = lambda lr: tf.train.AdamOptimizer(lr, epsilon=1e-4)
    elif opt == "rmsprop":
        opt_fn = lambda lr: tf.train.RMSPropOptimizer(lr)
    elif opt == "momentum":
        opt_fn = lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.9)
    else:
        raise ValueError("unknown optimizer type in Spec")

    with tf.Session(config=config, graph=g) as sess:
        algo = spec["algorithm"]
        if algo == "trpo":
            raise NotImplementedError("update to baselines-free, new policies")
            trained_policy = algos.trpo_sysid.learn(
                env, policy_fn,
                max_iters=spec["train_iters"],
                max_kl=0.003, cg_iters=10, cg_damping=0.1,
                gamma=spec["TD_discount"], lam=0.98,
                vf_iters=spec["opt_iters"], vf_stepsize=spec["learning_rate"],
                entcoeff=spec["entropy_coeff"],
                logdir=mydir)
        elif algo == "ppo":
            raise NotImplementedError("opt_fn")
            trained_policy = algos.ppo_sysid.learn(
                sess, env.np_random, env, dim, policy_fn,
                max_iters=spec["train_iters"],
                clip_param=0.2, entcoeff=spec["entropy_coeff"],
                optim_epochs=spec["opt_iters"], optim_batchsize=spec["minibatch"],
                optim_stepsize=spec["learning_rate"],
                gamma=spec["TD_discount"], schedule=spec["lr_schedule"],
                # lam=spec["tdlambda"],
                lam=0.96,
                logdir=mydir
            )
        elif algo == "qtopt":
            raise NotImplementedError("update to baselines-free, new policies")
            trained_policy = algos.qtopt_sysid.learn(
                env.np_random, env, policy_fn,
                learning_rate=spec["learning_rate"],
                target_update_iters=spec["q_target_assign"],
                max_iters=spec["train_iters"],
                optim_epochs=spec["opt_iters"], optim_batchsize=spec["minibatch"],
                td_lambda=0.99,
                schedule=spec["lr_schedule"],
                logdir=mydir
            )
        elif algo == "sac":

            is_finetune = load_dir is not None
            explore_steps = int(1e5) if is_finetune else spec["init_explore_steps"]

            hid_sizes = [spec["hidden_sz"]] * spec["n_hidden"]

            trained_policy = algos.sac_sysid.learn(
                sess, env.np_random, env, dim, policy_fn, opt_fn,
                vf_hidden=hid_sizes,
                learning_rate=spec["learning_rate"],
                max_iters=spec["train_iters"],
                logdir=mydir,
                tboard_dir=mydir,
                init_explore_steps=explore_steps,
                is_finetune=is_finetune,
                n_train_repeat=spec["n_train_repeat"],
                buf_len=spec["buf_len"],
                minibatch=spec["minibatch"],
                TD_discount=spec["TD_discount"],
                reward_scale=spec["reward_scale"],
                tau=spec["tau"],
                vf_grad_thru_embed=spec["vf_grad_thru_embed"],
            )
        else:
            assert False, "invalid choice of RL algorithm: " + algo

        trained_policy.save(sess, save_dir)

    env.close()


def test(spec: Spec, load_dir: SaverDir, save_path: TestPath,
    n_sysid_samples: int, infer_sysid: bool=True):

    spec.assert_is_scalar()

    tweak = spec["test_tweak"] if spec["test_mode"] == "tweak" else 0.0
    env = make_env(spec, tweak=tweak)
    env.seed()
    seed = spec.canonical_seed()
    var_init_npr = np.random.RandomState(seed)

    g = tf.Graph()
    g.seed = seed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config, graph=g) as sess:

        dim = get_dim(spec, env)
        ob_ph = tf.placeholder(tf.float32, (None, dim.ob_concat), "ob")
        ob_traj_ph = tf.placeholder(tf.float32, (None, dim.window, dim.ob), "ob_traj")
        ac_traj_ph = tf.placeholder(tf.float32, (None, dim.window, dim.ac), "ac_traj")

        with tf.variable_scope("pi"):
            pi = make_batch_policy_fn(var_init_npr, spec, dim, test=infer_sysid)(
                env.observation_space, env.action_space,
                ob_ph, ob_traj_ph, ac_traj_ph)

        pi.restore(sess, load_dir)

        # while in some cases, you might set stochastic=False at test time
        # to get the "best" actions, for SysID stochasticity could be
        # an important / desired part of the policy
        seg_gen = sysid_simple_generator(sess, pi, env, stochastic=True, test=infer_sysid)
        segs = list(it.islice(seg_gen, n_sysid_samples))

    mydir, _ = os.path.split(save_path)
    os.makedirs(mydir, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(segs, f, protocol=4)


# helper fn for parallelization over specs.
# arg_fn should take spec and return args for fn(*args).
# returns list of fn result.
# TODO: somehow separate stdouts (tiles w/ e.g. ncurses? open terminals?)
def grid(specs: List[Spec], fn, arg_fn, n_procs: int, always_spawn=False):
    if not always_spawn and (n_procs == 1 or len(specs) == 1):
        return [fn(*arg_fn(spec)) for spec in specs]
    else:
        asyncs = []
        pool = multiprocessing.Pool(processes=n_procs)
        for spec in specs:
            args = arg_fn(spec)
            res = pool.apply_async(fn, args)
            asyncs.append(res)
        results = [res.get() for res in asyncs]
        return results


def train_multispec(spec: Spec, rootdir: MultiSpecDir, n_procs: int,
    specs=None, load_dir: Optional[SaverDir]=None):

    if not specs:
        specs = multispec_product(spec)
    tboard_process = subprocess.Popen(["tensorboard", "--logdir", rootdir])
    print("Child TensorBoard process:", tboard_process.pid)
    try:
        def arg_fn(spec):
            save_dir = os.path.join(rootdir, spec.dir(), Spec.saver_name)
            return spec, save_dir, load_dir
        grid(specs, train, arg_fn, n_procs)
    finally:
        tboard_process.kill()
        pass


def test_multispec(multispec, rootdir: MultiSpecDir, n_sysid_samples: int, n_procs: int,
    specs=None, load_override: Optional[SaverDir]=None):

    if not specs:
        specs = multispec_product(multispec)

    def arg_fn(spec):
        if load_override is not None:
            load_dir = load_override
        else:
            load_dir = os.path.join(rootdir, spec.dir(), Spec.saver_name)
        save_path = os.path.join(rootdir, spec.dir(), Spec.test_pickle_name)
        return spec, load_dir, save_path, n_sysid_samples

    grid(specs, test, arg_fn, n_procs)


def load_test_results(multispec, rootdir: MultiSpecDir) -> List[List[dict]]:
    def gen():
        for spec in multispec_product(multispec):
            path = os.path.join(rootdir, spec.dir(), Spec.test_pickle_name)
            with open(path, "rb") as f:
                yield spec, pickle.load(f)
    return list(gen())


def iter_key(dicts, key):
    for d in dicts:
        yield d[key]


def ndarray_key(segs: List[dict], key: str) -> np.ndarray:
    rews = np.concatenate(list(iter_key(segs, key)), axis=0)
    return rews


def flatfirst(x):
    s = (-1,) + x.shape[2:]
    return np.reshape(x, s)


def flatlast(x):
    s = x.shape[:-2] + (-1,)
    return np.reshape(x, s)


def mean_sysid_err(segs):
    err2s = [np.mean((seg["est_true"] - seg["est"]) ** 2, axis=-1) for seg in segs]
    return np.mean(np.concatenate(err2s, axis=0))


def dict_with(d, **kwargs):
    d2 = deepcopy(d)
    d2.update(**kwargs)
    return d2


def group_by(specs: List[Spec], attached: list, key: str) -> List[Tuple[Any, Tuple[List[Spec], List[Any]]]]:
    vals = sorted(set(iter_key(specs, key)))
    def get(v):
        filtered = [(s, a) for s, a in zip(specs, attached) if s[key] == v]
        return tuple(zip(*filtered))
    return [(v, get(v)) for v in vals]


# returns:
#   List[List] outer dim is key, inner dim is all values of that key
#   np.ndarray with first len(keys) dimensions for key-vals, rest are dim(attached[0])
def tabulate(specs: List[Spec], attached: list, keys) -> Tuple[List[list], np.ndarray]:
    assert len(specs) == len(attached)
    if len(keys) == 0:
        return [], np.stack(attached)
    k = keys[0]
    vals, tups = zip(*group_by(specs, attached, keys[0]))

    def gen():
        for val, (gspecs, gattached) in group_by(specs, attached, keys[0]):
            yield tabulate(gspecs, gattached, keys[1:])

    nextvals, arrs = [list(z) for z in zip(*gen())]
    for nv in nextvals[1:]:
        assert nv == nextvals[0]
    return [vals] + nextvals[0], np.stack(arrs)


def group_seeds(specs: List[Spec], attached: List[Any]) -> List[Tuple[Spec, List[Tuple[int, Any]]]]:

    if len(specs) == 0:
        specs = multispec_product(specs[0])

    cores: List[Spec] = []
    for s in specs:
        s2 = dict_with(s, seed=None, directory=None)
        if s2 in cores:
            continue
        cores.append(s2)

    def matches(core, spec):
        return core == dict_with(spec, seed=None, directory=None)

    def find_seeds(core):
        for spec, att in zip(specs, attached):
            if matches(core, spec):
                seed = spec["seed"]
                #print(f"yielding ({seed}, {att})")
                yield seed, att

    return [(core, list(find_seeds(core))) for core in cores]


def group_reduce(specs: List[Spec], attached: List[Any],
    reducer: Callable[[List[Any]], Any]) -> List[Tuple[Spec, Any]]:

    groups = group_seeds(specs, attached)
    def gen():
        for spec, seeditems in groups:
            items = [item for seed, item in seeditems]
            yield spec, reducer(items)
    return list(gen())


def print_test_results(multispec, results):
    print("Printing results for Spec:")
    print(json.dumps(multispec, indent=4, sort_keys=True).replace('"', ''))
    print_keys = [k for k, v in multispec.items() if type(v) == list and k != "seed"]

    grouped = group_seeds(*zip(*results))

    for core, seed_results in grouped.items():

        def gen_lines():
            yield ", ".join(f"{k} = {core[k]}" for k in print_keys)
            rews = []
            rmses = []
            for seed, segs in seed_results:
                r = ndarray_key(segs, "ep_rews")
                rmse = np.sqrt(np.mean(ndarray_key(segs, "est_mserr"), axis=-1))
                rews.append(r)
                rmses.append(rmse)
                yield f"seed {seed:2} rew: {stats_str(r)}, est_rmse: {stats_str(rmse)}"

            yield f"overall rew: {stats_str(rews)}, est_rmse: {stats_str(rmses)}"
            # TODO std over episode, etc. instead of all timesteps?

        boxprint(list(gen_lines()))


def print_anova(flat_rews):
    raise NotImplementedError("multispec")
    # for ANOVA
    each_trial_rews = np.reshape(flat_rews, (flat_rews.shape[0], -1))
    f, p = sp.stats.f_oneway(*each_trial_rews)
    print("ANOVA each trial results: f = {}, p = {}".format(f, p))

    each_seed_rews = np.mean(flat_rews, axis=2).reshape((flat_rews.shape[0], -1))
    f, p = sp.stats.f_oneway(*each_seed_rews)
    print("ANOVA individual seed results: f = {}, p = {}".format(f, p))


def load_learning_curve(spec, rootdir):
    path = os.path.join(rootdir, spec.dir(), Spec.csv_name)
    data = np.genfromtxt(path, names=True, delimiter=",", dtype=np.float64)
    col_names = ["Env{}Rew".format(i) for i in range(spec["n_batch"])]
    rews = np.column_stack([data[c] for c in col_names])
    return rews


def embed_scatter_data(segs):
    trues = np.stack(iter_key(segs, "est_true"))
    estimates = np.stack(iter_key(segs, "est"))
    return trues, estimates


# compute the policy mean action at a fixed task state
# conditioned on the SysID params
# TODO: rewrite to use specs
def action_conditional(sess, env_id, test_state, flavor, seed, mydir):

    raise NotImplementedError("multispec")
    assert env_id == "PointMass-Batch-v0"
    env = gym.make(env_id)

    alpha_sysid = 0  # doesn't matter at test time
    pi = make_batch_policy_fn(env, flavor, alpha_sysid, seed=seed)(
        "pi", env.observation_space, env.action_space)

    seeddir = os.path.join(mydir, str(seed))
    ckpt_path = os.path.join(seeddir, 'trained_model.ckpt')
    pi.restore(sess, ckpt_path)

    assert env.sysid_dim == len(env.sysid_names)
    N = 1000
    x = np.linspace(-4, 4, N)

    test_state = np.array([0.9, 0.9, 0, 0])  # env starting state - upper right corner

    def gen():
        for i, name in enumerate(env.sysid_names):
            sysid = np.zeros((N, env.sysid_dim))
            sysid[:, i] = x
            pi_input = np.hstack([np.tile(test_state, (N, 1)), sysid])
            actions, values = pi.act(False, pi_input)
            actions[actions < -1] = -1
            actions[actions > 1] = 1
            yield name, x, actions

    return list(gen())








# generate data for plotting the mapping from SysID dimension to embedding.
def sysids_to_embeddings(sess, env_id, seed, mydir):

    raise NotImplementedError("multispec")
    env = gym.make_env(env_id)
    assert env.sysid_dim == len(env.sysid_names)

    alpha_sysid = 0  # doesn't matter at test time
    pi = make_batch_policy_fn(env, sysid_batch_policy.EMBED, alpha_sysid)(
        "pi", env.observation_space, env.action_space)

    seeddir = os.path.join(mydir, str(seed))
    ckpt_path = os.path.join(seeddir, 'trained_model.ckpt')
    pi.restore(sess, ckpt_path)

    if env.sysid_dim == 1:
        N = 1000
        x = np.linspace(*env.sysid_ranges[0], num=N)
        y = pi.sysid_to_embedded(x[:, None])
        return name, x, y

    elif env.sysid_dim == 2:
        N = 256
        x0, x1 = np.meshgrid(
            *(np.linspace(r[0], r[1], N) for r in env.sysid_ranges))
        input = np.concatenate([x0[:, :, None], x1[:, :, None]], axis=2)
        input = input.reshape((-1, 2))
        y = pi.sysid_to_embedded(input)
        y = y.reshape((N, N, 2))
        return env.sysid_ranges, env.sysid_names, y

    else:
        raise NotImplementedError


def stats_str(x, axis=None):
    if np.any(np.isnan(x)):
        mu = std = np.nan()
    else:
        mu = np.mean(x, axis=axis)
        std = np.std(x, axis=axis)
    return f"mean = {mu:.2f}, std = {std:.2f}"


# print an array of strings surrounded by a box
def boxprint(lines: List[str]):
    maxlen = max(len(s) for s in lines)

    def frame(around, middle):
        return around + middle + around[::-1]

    def rpad(s):
        return s + " " * (maxlen - len(s))

    bar = frame("-" * (maxlen + 2), "  *")
    print(bar)
    for line in lines:
        print(frame(rpad(line), '  | '))
    print(bar)


def set_xterm_title(title: str):
    sys.stdout.write("\33]0;" + title + "\a")


# if directory already exists, ask user what to do.
# raises RuntimeError if user chooses to abort.
# >>> user can ask to rm -rf, this function does it! <<<
def check_spec_dir(spec: Spec, rootdir: str) -> str:

    path = os.path.join(rootdir, spec.hash())
    json_path = os.path.join(path, Spec.json_name)

    if not os.path.exists(path):
        os.makedirs(path)
        return path

    if os.path.exists(json_path):
        with open(json_path) as f:
            dirspec = json.load(f)
            if dirspec != spec:
                raise ValueError("Hash collision!!")
    else:
        with open(json_path, "w") as f:
            json.dump(spec, f)

    while True:
        response = input(
            f"directory {path} already exists. Abort / Keep / Delete? (a/k/d)\n")
        char = response[0].lower()
        if char == "a":
            raise RuntimeError("user aborted due to pre-existing results")
        if char == "k":
            break
        if char == "d":
            shutil.rmtree(path)
            os.mkdir(path)
            break

    return path


class Progress(object):
    def __init__(self, total_work, print_every_percent=10):

        self.total = float(total_work)
        self.n = 0
        self.last_emitted = -1
        self.chunk = print_every_percent

    def update(self, completed_work):
        self.n = completed_work
        percent = 100.0 * (self.n / self.total)
        print_pct = self.chunk * (int(percent) // self.chunk)

        if print_pct != self.last_emitted:
            self.last_emitted = print_pct
            return True, print_pct
        else:
            return False, int(percent)

    def add(self, amount):
        return self.update(self.n + amount)

