# DRLinter

`DRLinter` is a toolset for verifying Deep Reinforcement Learning (DRL) models using meta-modeling and graph transformations.
This tool set performs verification of DRL models that are specified using graph transformations by the Groove toolset.
First, the DRL program is parsed to extract relevant information according to the meta-model. The model of the program is a graph that conforms to the type graph (meta-model). Then, the graph is verified by `Groove` as a model checker. The output graph of Groove is used to extract relevant Information for the final report.

`groove-x_x_x-bin` and `DRL-metamodel.gps` folders are the Groove tollset and type graph respectively which are needed for running `NeuraLint`.

`graphs` folder is also used for intermediate calculation and required to run `DRLinter` tool.

There are also two parsers to parse the DRL programs. The first one is usable for the DRL programs follow the sample template. The second one can be used for the DRL programs annotated with predefined annotations. 

The tool is written in `Python` and it can be easily run in the command line. To use the tool set, please enter following options with running command:
```
$ python endToEnd.py [name of DRL programs (.py)] [parser_type] [name of the output file]
```
- `[name of DRL programs (.py)]` should be entered with `.py`
 
- `[parser type]` should be `annot` or `synth`

- `[name of the output file]` should be entered without file type

## Examples
### Cartpole environment with TensorFlow Example

Following code is a sample script for `cartpole` using `tensorflow`.
```python
import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from statistics import mean
from gym import wrappers

class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear', kernel_initializer='RandomNormal') 

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next) 

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon: 
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0]) 

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset() 
    losses = list()
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action) 
        rewards += reward
        if done: 
            reward = -200
            env.reset() 

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp) 
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet) 
    return rewards, mean(losses)

def main():
    gamma = 0.99 
    copy_step = 25 
    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n
    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32 
    lr = 1e-2 
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    N = 50000 
    total_rewards = np.empty(N)
    epsilon = 0.99 
    decay = 0.9999 
    min_epsilon = 0.1
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay) 
        total_reward, losses = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
            tf.summary.scalar('average loss)', losses, step=n)
        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
                  "episode loss: ", losses)
    print("avg reward for last 100 episodes:", avg_rewards)
    env.close()

if __name__ == '__main__':
    for i in range(3):
        main()
```

User can use following command to pars mentioned code using `DRLinter`. 

```
$ python EndToEnd.py cartpole_tensorflow.py result
```

The result of `DRLinter` is

```
Wrong initialization.
```

## The Paper
You can find the paper [here](https://link.springer.com/article/10.1007/s10515-021-00313-x) and the citation is as follows:

    @article{nikanjam2022faults,
    title={Faults in deep reinforcement learning programs: a taxonomy and a detection approach},
    author={Nikanjam, Amin and Morovati, Mohammad Mehdi and Khomh, Foutse and Ben Braiek, Houssem},
    journal={Automated Software Engineering},
    volume={29},
    number={1},
    pages={1--32},
    year={2022},
    publisher={Springer},
    url = {https://link.springer.com/article/10.1007/s10515-021-00313-x},
    doi = {10.1007/s10515-021-00313-x}
    }
