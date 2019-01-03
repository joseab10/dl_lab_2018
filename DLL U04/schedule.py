import numpy as np

class Schedule:

    def __init__(self, y_0, y_1, t, decay_function="exponential",
                 # Cosine Annealing Parameters
                 cosine_annealing=True, annealing_cycles=3,
                 # Stepping Parameters
                 steps=0):

        self.y_0 = y_0
        self.y_1 = y_1
        self.t   = t

        decay_function = decay_function.lower()

        if decay_function == "constant":
            self._decay_function = self._constant
            self.epsilon = self._constant
            self.y_0 = y_1

        elif decay_function == "linear":
            self._decay_function = self._linear

        elif decay_function == "exponential":
            self._decay_function = self._exponential
            # Due to the exponential aproaches asymptotically the final value,
            # this factor is to lower the asymptote a little bit to allow for an exact minimum value of epsilon after t epochs,
            # otherwise, it would just aproximate e_min, but never actually reach it
            self.exponential_k = 0.98
            sign = 1
            if y_1 > y_0:
                self.exponential_k = 2 - self.exponential_k
                sign = -1

            # Exponential multiplicative factor (e ^ (alpha * x)) to have a nice continuous function (almost tangent to e_min)
            # within the whole training interval while also reaching e_min at the end.
            self.exponential_factor = np.log(
                (sign * (1 - self.exponential_k) * y_1) / (self.exponential_k * y_1 + y_0)) / self.t

        else:
            raise NotImplemented("Error: Decay Function not implemented.")


        if steps > 0 and decay_function is not 'constant':

            self.step_interval = self.t / steps
            tmp_x = np.arange(steps) * self.step_interval

            if decay_function == "linear":
                self.y_1 = self._linear(self.t + self.step_interval)
                self.step_table = self._linear(tmp_x)

            if decay_function == "exponential":
                self.y_1 = self._exponential(self.t + self.step_interval)
                self.step_table = self._exponential(tmp_x)

            self.y_1 = y_1

            self._decay_function = self._step


        # Cosine Annealing Parameters
        #----------------------------
        if cosine_annealing and decay_function is not 'constant':
            self.epsilon = self._cos
        else:
            self.epsilon = self._decay_function

        # Number of full Peaks the cosine will make within the training interval
        self.annealing_cycles = annealing_cycles


    def _constant(self, x):
        if isinstance(x, np.ndarray):
            return self.y_1 * np.ones_like(x)

        else:
            return self.y_1

    def _linear(self, x):
        return (self.y_0 - ((self.y_0 - self.y_1) / self.t) * x)

    def _exponential(self, x):
        #exp = self.e_0 * np.power(self.exponential_base, x)
        power = x * self.exponential_factor
        exp = (self.y_0 - (self.exponential_k * self.y_1)) * np.exp(power) + (self.exponential_k * self.y_1)

        if isinstance(exp, np.ndarray):
            if self.y_0 > self.y_1:
                exp[exp < self.y_1] = self.y_1
            else:
                exp[exp > self.y_1] = self.y_1
            decay = exp

        else:
            decay = exp
            if self.y_0 > self.y_1 and decay < self.y_1:
              decay = self.y_1

        return decay

    def _step(self, x):

        index = x // self.step_interval

        if isinstance(index, np.ndarray):
            index = index.astype(int)
            index[index < 0] = 0
            index[index >= len(self.step_table)] = len(self.step_table) - 1

            return np.array([self.step_table[i] for i in index])
        else:
            index = int(index)
            return self.step_table[index]

    def _cos(self, x):

        decay = self._decay_function(x)
        # Cosine Amplitude (Height)
        a = (decay - self.y_1) / 2

        # Cosine Function Centerline Offset (around which the cosine oscillates)
        b = (decay + self.y_1) / 2

        return (a * np.cos((2 * self.annealing_cycles + 1) * (np.pi / self.t) * x)) + b

    def _piecewise(self, x):

        y = self.epsilon(x)

        if isinstance(x, np.ndarray):
            indexes = np.where(x < 0)
            y[indexes] = self.y_0
            indexes = np.where(x > self.t)
            y[indexes] = self.y_1

        else:
            if x < 0:
                y = self.y_0
            elif x > self.t:
                y = self.y_1

        return y

    def __call__(self, x):
        return self._piecewise(x)



