import numpy as np
import matplotlib.pyplot as plt

def flat_rew_hist(segs, flat_rewards):
    import pdb; pdb.set_trace()
    # TODO don't average over seeds?
    hist_datas = [flat_rews.flatten() for flat_rews in flat_rewards]
    labels = ["{}, {}".format(f, a) for (f, a, _) in segs]
    fig = plt.figure()
    plt.hist(hist_datas, label=labels)
    plt.legend()
    plt.xlabel("reward")
    return fig
