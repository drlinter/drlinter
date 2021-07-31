import gym
import numpy as np
import tensorflow as tf
import math
import keras
import random

class cartpole:
    def __init__(self, sess, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.sess = sess
        self.epsilon = 1.0 #@DRLinter-->epsilon_decay
        self.return_loss = 0.0
        self.memory = []
        self.gamma = .95 #@DRLinter-->gamma

        self.q_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    def q_model(self):
        self.state_input = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
        self.reward_labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.hiddenlayer1_weights = tf.Variable(tf.random_normal([self.state_size, 32]))
        self.hiddenlayer1_bias = tf.Variable(tf.random_normal([32]))
        self.hiddenlayer1_output = tf.matmul(self.state_input, self.hiddenlayer1_weights) + self.hiddenlayer1_bias
        self.hiddenlayer1_output = tf.nn.relu(self.hiddenlayer1_output)

        self.hiddenlayer2_weights = tf.Variable(tf.random_normal([32, 16]))
        self.hiddenlayer2_bias = tf.Variable(tf.random_normal([16]))
        self.hiddenlayer2_output = tf.matmul(self.hiddenlayer1_output, self.hiddenlayer2_weights) + self.hiddenlayer2_bias
        self.hiddenlayer2_output = tf.nn.relu(self.hiddenlayer2_output)


        self.q_weights = tf.Variable(tf.random_normal([16, self.num_actions]))
        self.q_bias = tf.Variable(tf.random_normal([self.num_actions]))
        self.q_output = tf.matmul(self.hiddenlayer2_output, self.q_weights) + self.q_bias
        self.q_output = keras.activations.linear(self.q_output)
        
        
        self.max_q_value = tf.reshape(tf.reduce_max(self.q_output), (1,1))
        self.best_action = tf.squeeze(tf.argmax(self.q_output, axis=1))

        self.loss = tf.losses.mean_squared_error(self.max_q_value, self.reward_labels)
        self.train_model = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
    
    def predict_action(self, state):
        self.epsilon *= .995 + .01
        if (np.random.random() < self.epsilon):
            action = env.action_space.sample() #@DRLinter-->exploration_check
        else:
            action = self.sess.run(self.best_action, feed_dict={self.state_input: state})
        return action

    def predict_value(self, state):
        state = np.array(state).reshape((1, 4))
        max_q_value = self.sess.run(self.max_q_value, feed_dict={self.state_input: state})[0][0]
        return max_q_value

    
    def train_q_model(self, state, reward):
        q_values, _, loss = self.sess.run([self.q_output, self.train_model, self.loss], feed_dict={self.state_input: state, self.reward_labels: reward})
        self.return_loss = loss

    def get_loss(self):
        return self.return_loss

    def experience_replay(self):
        if len(self.memory) < 33:
            return
        del self.memory[0]
        batch = random.sample(self.memory, 32)
        for state, action, reward, new_state, done in self.memory:
            reward = reward if not done else - reward
            new_state = np.array(new_state).reshape((1, 4))
            if not done:
                reward = reward + (self.gamma * self.predict_value(new_state)) #@DRLinter-->update_eq
            reward = np.array(reward).reshape((1, 1))
            
            self.train_q_model(state, reward)



env = gym.make("CartPole-v0") #@DRLinter-->create_env
sess = tf.Session()
A2C = cartpole(sess, env)

episodes = 2000
reward_history = []
for i in range(episodes):
    state = env.reset() #@DRLinter-->initialize_env_correct,terminate_isCorrect
    reward_total = 0
    while True:
        state = np.array(state).reshape((1, 4))
        average_best_reward = sum(reward_history[-100:]) / 100.0
        if (average_best_reward) > 195:
            env.render()

        action = A2C.predict_action(state)
        new_state, reward, done, _ = env.step(action) #@DRLinter-->in_correct_step,env_close
        reward_total += reward
        A2C.memory.append([state, action, reward, new_state, done])
        A2C.experience_replay()
        state = new_state


        if done:
            if (average_best_reward >= 195):
                print("Finished! Episodes taken: ", i, "average reward: ", average_best_reward)
            print("average reward  = ", average_best_reward, "reward total = ", reward_total, "loss = ", A2C.get_loss())
            reward_history.append(reward_total)
            break #@DRLinter-->terminal_state
