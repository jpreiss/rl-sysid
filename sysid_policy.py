import tensorflow as tf
import numpy as np
import baselines as bl
import baselines.common.tf_util as U


# obs_dim: dimension of the observation
# act_dim: dimension of the action
# latent_dim: dimension of the latent variable (user choice, not specified by task)
# hidden_dim: dimension of the known task specification
#             (e.g. one-hot for tasks, physical params for SysID)
# n_history: number of state-action pairs input to SysID
#
class SysID(object):
	def __init__(self, name,
		obs_dim, act_dim, latent_dim, sysid_dim,
		n_history,
		alpha_sysid,
		):

		# TODO match variable prefixes "vf" and "pol"

		self.obs_dim = obs_dim
		self.act_dim = obs_dim
		self.latent_dim = obs_dim
		self.sysid_dim = obs_dim
		self.n_history = n_history
		self.alpha_sysid = alpha_sysid

		self.recurrent = False

		class MLPModule(object):
			def __init__(self, input, output_dim, name):
				self.name = name
				with tf.variable_scope(name + "_fc1"):
					self.fc1 = tf.layers.dense(input, units=128, activation=tf.nn.relu)
				with tf.variable_scope(name + "_fc2"):
					self.fc2 = tf.layers.dense(self.fc1, units=64, activation=tf.nn.relu)
				with tf.variable_scope(name + "_output"):
					self.output = tf.layers.dense(self.fc2, units=output_dim)

		# ------------------ initialize inputs and network ----------------- #
		with tf.variable_scope(name):

			self.scope = tf.get_variable_scope().name

			# split gym environment's input into real observation and sysID info

			# sigh... baselines stupid bullshit creeping into my code...
			#self.input_concatenated = tf.placeholder(
				#tf.float32, [None, obs_dim + sysid_dim], name="ob")
			self.input_concatenated = U.get_placeholder(
				name="ob", dtype=tf.float32, shape=[None, obs_dim + sysid_dim])
			self.input_obs, self.input_sysid = tf.split(
				self.input_concatenated, [obs_dim, sysid_dim], 1)

			with tf.variable_scope("pol"):
				# embedding network
				self.embed = MLPModule(self.input_sysid, latent_dim, "embedding")

				# policy network
				self.policy_input = tf.concat([self.input_obs, self.embed.output], 1)
				self.policy = MLPModule(self.policy_input, 2 * act_dim, "policy")
				self.policy_mu, self.policy_sigma = tf.split(
					self.policy.output, [act_dim, act_dim], 1)

				# sysid network
				#self.input_traj_obs = tf.placeholder(
					#tf.float32, [None, n_history, obs_dim], name="traj_obs")
				self.input_traj_obs = U.get_placeholder(
					name="traj_ob", dtype=tf.float32, shape=[None, n_history, obs_dim])
				#self.input_traj_act = tf.placeholder(
					#tf.float32, [None, n_history, act_dim], name="traj_act")
				self.input_traj_act = U.get_placeholder(
					name="traj_ac", dtype=tf.float32, shape=[None, n_history, act_dim])
				self.input_traj = tf.concat([self.input_traj_obs, self.input_traj_act], 2)
				self.sysid = MLPModule(
					tf.layers.flatten(self.input_traj), latent_dim, "sysID")

			with tf.variable_scope("vf"):
				# value function network
				# (TODO: why do the OpenAI baselines use value func instead of q func?)
				self.value = MLPModule(self.input_obs, 1, "value")

			# for OpenAI Baselines interface
			self.vpred = self.value.output
			#self.input_stochastic = tf.placeholder(tf.bool, (), name="stochastic")
			self.input_stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
			self.pdtype = bl.common.distributions.DiagGaussianPdType(act_dim)
			self.pd = self.pdtype.pdfromflat(self.policy.output)
			self.ac = tf.cond(self.input_stochastic,
				lambda: self.pd.sample(),
				lambda: self.pd.mode())

			self.sysid_err_supervised = tf.losses.mean_squared_error(
				self.embed.output,
				self.sysid.output)

			# set up ADAM optimizer for supervised learning of sysid network
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.lr_decay = tf.train.exponential_decay(
				5e-4, # learning_rate
				self.global_step, # global_step
				500, # decay_steps
				0.999 # decay_rate
			)
			self.sysid_supervised_optimizer = tf.train.AdamOptimizer(
				learning_rate=self.lr_decay
			)
			self.opt_step_sysid = self.sysid_supervised_optimizer.minimize(
				loss=self.sysid_err_supervised,
				global_step=self.global_step)

	def extra_losses(self):
		return [self.sysid_err_supervised], ["System_ID_Supervised"]

	# do a supervised learning step on the system id network
	def learn_sysid(self, traj_obs, traj_act, sysid):
		feed_dict = {
			self.input_traj_obs: traj_obs,
			self.input_traj_act: traj_act,
			self.input_sysid: sysid,
		}
		fetches = [self.opt_step_sysid, self.sysid_err]
		_, loss = sess.run(fetches, feed_dict=feed_dict)
		return loss

	# OpenAI Baselines TRPO interface
	def act(self, stochastic, obs):
		sess = tf.get_default_session()
		feed_dict = {
			self.input_concatenated: obs[None],
			self.input_stochastic: stochastic,
		}
		fetches = [self.ac, self.value.output]
		actions, values = sess.run(fetches, feed_dict=feed_dict)
		action, value = actions[0], values[0]
		# TODO get actual bounds
		action[action > 1] = 1
		action[action < -1] = -1
		return action, value
	def get_variables(self):
		vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
		#print("get_variables:")
		#for v in vars:
			#print(v)
		return vars
	def get_trainable_variables(self):
		vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
		#print("get_trainable_variables:")
		#for v in vars:
			#print(v)
		return vars
	def get_initial_state(self):
		return []
