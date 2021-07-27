import gym
import universe
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation
from keras.models import load_model

import random

env = gym.make('gym-core.Pong-v0') #@DRLinter-->create_env
env.configure(remotes=1)


def num2str(number, obs):
    number = np.argmax(number)
    if number == 0:
        action = [[('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'ArrowLeft', True)] for ob in obs]
    elif number == 1:
        action = [[('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)] for ob in obs]

    return action


def preprocess(original_obs):
    obs = original_obs
    obs = np.array(obs)[0]['vision']
    obs = np.delete(obs, np.s_[195:769], axis=0)
    obs = np.delete(obs, np.s_[0:35], axis=0)
    obs = np.delete(obs, np.s_[160:1025], axis=1)
    obs = np.mean(obs, axis=2)
    obs = obs[::2,::2]
    obs = np.reshape(obs, (80, 80, 1))
    return obs



model = Sequential()
model.add(Conv2D(32, kernel_size = (8, 8), strides = (4, 4), border_mode='same', activation='relu', init='uniform', input_shape = (80, 80, 4)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, kernel_size = (2, 2), strides = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), strides = (1, 1)))

model.add(Flatten())
model.add(Dense(256, init='uniform', activation='relu'))
model.add(Dense(2, init='uniform', activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


init_observe_time = 500

D = []

e = 1.0
e_threshold = 0.05
e_decay = 0.01 #@DRLinter-->epsilon_decay

gamma = 0.99 #@DRLinter-->gamma

batch_size = 15 #@DRLinter-->batch_size
frequency = 10 #@DRLinter-->update_freq

Q_values = np.array([0, 0])

obs = env.reset() #@DRLinter-->initialize_env_correct

while True:
    obs = env.step(num2str(np.array([random.randint(0, 1) for i in range(0, 2)]), obs))[0]
    if obs != [None]:
        break

x_t1 = preprocess(obs)
s_t1 = np.stack((x_t1, x_t1, x_t1, x_t1), axis = 2)
s_t1 = np.reshape(s_t1, (80, 80, 4))

t = 0
while True:

    print("Time since last start: ", t)

    a_t = np.zeros(2)

    if random.random() < e:
        a_index = random.randint(0, 1) #@DRLinter-->exploration_check
        a_t[a_index] = 1
    else:
        Q_values = model.predict(np.array([s_t1]))[0]
        a_index = np.argmax(Q_values)
        a_t[a_index] = 1

    print("Q Values: ", Q_values)
    print("action taken: ", np.argmax(a_t))
    print("epsilon: ", e)

    if e > e_threshold:
        e -= e_decay

    obs, r_t, done, info = env.step(num2str(a_t, obs)) #@DRLinter-->in_correct_step

    if obs == [None]:
        continue #@DRLinter-->terminal_state

    x_t2 = preprocess(obs)
    print(x_t2.shape, s_t1[:,:,0:3].shape)
    s_t2 = np.append(x_t2, s_t1[:,:,0:3], axis = 2)

    D.append((s_t1, a_t, r_t, s_t2, done))

    if t > init_observe_time and t%frequency == 0:
        minibatch = random.sample(D, batch_size)

        s1_batch = [i[0] for i in minibatch]
        a_batch = [i[1] for i in minibatch]
        r_batch = [i[2] for i in minibatch]
        s2_batch = [i[3] for i in minibatch]

        q_batch = model.predict(np.array(s2_batch))
        y_batch = np.zeros((batch_size, 2))
        y_batch = model.predict(np.array(s1_batch))
        print("Q batch: ",  q_batch)
        print("y batch: ",  y_batch)
        for i in range(0, batch_size):
            if (minibatch[i][4]):
                y_batch[i][np.argmax(a_batch[i])] = r_batch[i][0]
            else:
                y_batch[i][np.argmax(a_batch[i])] = r_batch[i][0] + gamma * np.max(q_batch[i]) #@DRLinter-->update_equation


        model.train_on_batch(np.array(s1_batch), y_batch)
    s_t1 = s_t2

    t += 1
    env.render()
