import tensorflow as tf
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from memory import Memory



class BaseModel():

    def __init__() -> None:
        pass


    def _create_model(self) -> tf.keras.Model:
        return None


    def predict(self, state : np.ndarray) -> np.ndarray:
        return None


    def learn(self, mem : Memory, n_samples : int) -> None:
        pass


    def save(self) -> None:
        pass


    def load(self) -> None:
        pass


    def refresh_weights(self) -> None:
        pass



class ModelNaive(BaseModel):

    def __init__(self, n_input : int, n_output : int, n_neurons : list,
                 learning_rate : float, gamma : float, save_file : str, verbose : int, device : str) -> None:
        self.n_input = n_input
        self.n_output = n_output
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.verbose : Model = verbose
        self.model : Model = self._create_model()
        self.device =  device
        self.save_file = save_file + '_' + self.__class__.__name__ + ' ' + self.device


    def _create_model(self) -> Model:
        model = Sequential([
            Input(shape=(self.n_input)),
            Dense(self.n_neurons[0], activation='relu'),
            Dense(self.n_neurons[1], activation='relu'),
            Dense(self.n_output, activation='linear')
        ])

        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mse',
                      metrics=['mse', 'mae'])

        model.summary()

        return model


    def predict(self, state : np.ndarray) -> np.ndarray:
        x_data = np.reshape(state, (1, self.n_input))
        return self.model.predict(x=x_data, verbose=self.verbose)


    def learn(self, mem : Memory, n_samples : int) -> None:
        if n_samples > mem.count:
            return

        states, actions, rewards, new_states, terminates = mem.sample(n_samples=n_samples)

        q_table = self.model.predict(x=states, verbose=self.verbose)
        q_table_next = self.model.predict(x=new_states, verbose=self.verbose)

        batch_index = np.arange(n_samples, dtype=np.int32)
        action_values = np.array(list(range(self.n_output)), dtype=np.int8)
        action_indices = np.dot(actions, action_values)
        max_actions = np.argmax(q_table_next, axis=1)
        q_table[batch_index, action_indices] = rewards[batch_index]
        q_table[batch_index, action_indices] += self.gamma * q_table_next[batch_index, max_actions] * terminates[batch_index]

        self.model.fit(x=states, y=q_table, verbose=self.verbose, batch_size=n_samples, epochs=1)


    def save(self) -> None:
        self.model.save(f'weights_{self.save_file}.h5')


    def load(self) -> None:
        try:
            self.model = load_model(f'weights_{self.save_file}.h5')
            print(f'using weights weights_{self.save_file}.h5')
        except:
            print(f'cannot load weights weights_{self.save_file}.h5')



class ModelActorCritic(BaseModel):

    def __init__(self, n_input : int, n_output : int, n_neurons : list,
                 learning_rate : float, gamma : float, save_file : str, verbose : int, device : str) -> None:
        self.n_input = n_input
        self.n_output = n_output
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.verbose = verbose
        self.actor : Model = self._create_model()
        self.target : Model = self._create_model()
        self.device = device
        self.save_file = save_file + '_' + self.__class__.__name__ + '_' + self.device


    def _create_model(self) -> Model:
        model = Sequential([
            Input(shape=(self.n_input)),
            Dense(self.n_neurons[0], activation='relu'),
            Dense(self.n_neurons[1], activation='relu'),
            Dense(self.n_output, activation='linear')
        ])

        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mse',
                      metrics=['mse', 'mae'],
                      run_eagerly=False,
                      jit_compile=True)

        #model.summary()

        return model


    def predict(self, state : np.ndarray) -> np.ndarray:
        x_data = np.reshape(state, (1, self.n_input))
        return self.actor.predict(x=x_data, verbose=self.verbose)


    def learn(self, mem : Memory, n_samples : int) -> None:
        if n_samples > mem.count:
            return
        states, actions, rewards, new_states, terminates = mem.sample(n_samples=n_samples)

        q_table = self.actor.predict(x=states, verbose=self.verbose)
        q_table_next_action = self.actor.predict(x=new_states, verbose=self.verbose)
        q_table_next = self.target.predict(x=new_states, verbose=self.verbose)


        batch_index = np.arange(n_samples, dtype=np.int32)
        action_values = np.array(list(range(self.n_output)), dtype=np.int8)
        action_indices = np.dot(actions, action_values)
        max_actions = np.argmax(q_table_next_action, axis=1)
        q_table[batch_index, action_indices] = rewards[batch_index]
        q_table[batch_index, action_indices] += self.gamma * q_table_next[batch_index, max_actions] * terminates[batch_index]

        self.actor.fit(x=states, y=q_table, verbose=self.verbose, batch_size=n_samples, epochs=1)


    def save(self) -> None:
        self.actor.save(f'weights_{self.save_file}.h5')


    def load(self) -> None:
        try:
            self.actor = load_model(f'weights_{self.save_file}.h5')
            self.target = load_model(f'weights_{self.save_file}.h5')
            print(f'using weights weights_{self.save_file}.h5')
        except:
            print(f'cannot load weights weights_{self.save_file}.h5')


    def refresh_weights(self) -> None:
        self.target.set_weights(self.actor.get_weights())
