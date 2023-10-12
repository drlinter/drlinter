from collections import deque
import tensorflow as tf
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import pickle
from time import time
t = int(time())
class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 2000)
        self.gamma = 0.95 #@DRLinter-->gamma
        #self.epsilon = 1.0
        #self.epsilon_min = 0.01
        #self.epsilon_decay = 0.995
        self.learning_rate = 0.001 #@DRLinter-->learning_rate
        self.model = self._build_model()

    def _build_model(self):
        graph = tf.Graph()
        with graph.as_default():
            inp = tf.placeholder(tf.float32, [None, self.state_size])
            out = tf.placeholder(tf.float32, [None, self.action_size])
            w1 = tf.Variable(tf.truncated_normal([self.state_size, 24]))
            b1 = tf.Variable(tf.zeros([24]))

            hidden = tf.nn.tanh(tf.matmul(inp, w1) + b1)

            w2 = tf.Variable(tf.truncated_normal([24, 24]))
            b2 = tf.Variable(tf.zeros([24]))

            hidden1 = tf.nn.tanh(tf.matmul(hidden, w2) + b2)

            w3 = tf.Variable(tf.truncated_normal([24, 24]))
            b3 = tf.Variable(tf.zeros([24]))

            hidden2 = tf.nn.tanh(tf.matmul(hidden1, w3) + b3)

            wo = tf.Variable(tf.truncated_normal([24, self.action_size]))
            bo = tf.Variable(tf.zeros([self.action_size]))

            prediction = tf.matmul(hidden2, wo) + bo

            loss = tf.losses.mean_squared_error(out, prediction)
            train = tf.train.AdamOptimizer().minimize(loss)
            init = tf.global_variables_initializer()

        return graph, inp, out, prediction, train, init

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, sess):
        act_values = sess.run(self.model[3], feed_dict = { self.model[1]: state})
        return np.argmax(act_values[0])

    def replay(self, batch_size, sess):
        try:
            minibatch = random.sample(self.memory, batch_size)
        except ValueError:
            minibatch = self.memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(sess.run(self.model[3], feed_dict = { self.model[1]: next_state})) #@DRLinter-->update_eq
            target_f = sess.run(self.model[3], feed_dict = { self.model[1]: state})
            target_f[0][action] = target
            #print(target_f)
            sess.run(self.model[4], feed_dict = { self.model[1]: state, self.model[2]: target_f})

if __name__ == "__main__":
    environment = 'CartPole-v0' #@DRLinter-->initialize_env
    env = gym.make(environment)
    avgs = deque(maxlen = 50)
    rewardLA = []
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    sess = tf.Session(graph = agent.model[0])
    sess.run(agent.model[5])
    episodes = 10000 #@DRLinter-->epoch_count
    rewardL = []
    for e in range(episodes):
        state = env.reset() #@DRLinter-->initialize_env_correct,env_close
        state = np.reshape(state, [1, 4])
        for time_t in range(500):
            #env.render()
            action = agent.act(state, sess)
            next_state, reward, done, _ = env.step(action) #@DRLinter-->in_correct_step
            next_state = np.reshape(next_state, [1, 4])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break #@DRLinter-->terminal_state,terminate_isCorrect
        avgs.append(time_t)
        rewardLA.append(sum(avgs)/len(avgs))
        print("episode: ", e, "score: ", time_t)
        rewardL.append(time_t)
        agent.replay(32, sess) #@DRLinter-->batch_size,_
    #pickle.dump(rewardL, open(environment + "_" + str(t) + "_rewardL.pickle", "wb"))
    plt.plot(rewardLA)
    plt.show()
