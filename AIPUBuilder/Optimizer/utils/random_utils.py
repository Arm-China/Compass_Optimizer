# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.
from abc import abstractmethod
import torch
import numpy as np


mt19937_params = {
    "W": 32,
    "N": 624,
    "M": 397,
    "R": 31,
    "A": 0x9908b0df,
    "F": 1812433253,
    "U": 11,
    "D": 0xFFFFFFFF,
    "S": 7,
    "B": 0x9D2C5680,
    "T": 15,
    "C": 0xEFC60000,
    "L": 18
}


class random_generate_engine(object):
    def __init__(self, params_dict):
        super(random_generate_engine, self).__init__()
        self.params_dict = params_dict

    @abstractmethod
    def min(self):
        return 0

    @abstractmethod
    def max(self):
        return 0

    @abstractmethod
    def set_seed(self, seed):
        return seed

    @abstractmethod
    def __call__(self):
        return 0


# Consistent with the implementation of the c++ open source library mt19937
class mersenne_twister_engine(random_generate_engine):
    def __init__(self, params_dict):
        super(mersenne_twister_engine, self).__init__(params_dict)
        self.W = self.params_dict['W']
        self.N = self.params_dict['N']
        self.M = self.params_dict['M']
        self.R = self.params_dict['R']
        self.A = self.params_dict['A']
        self.F = np.array(self.params_dict['F']).astype(np.uint64)
        self.U = self.params_dict['U']
        self.D = self.params_dict['D']
        self.S = self.params_dict['S']
        self.B = self.params_dict['B']
        self.T = self.params_dict['T']
        self.C = self.params_dict['C']
        self.L = self.params_dict['L']
        self.mt = np.zeros([self.N], dtype=np.uint32)
        self.index = 0
        self.MASK_LOWER = (np.power(2, self.R)-1).astype(np.uint64)
        self.MASK_UPPER = ~self.MASK_LOWER

    def max(self):
        return np.power(2, self.W) - 1

    def min(self):
        return 0

    def set_seed(self, seed):
        self.mt[0] = int(seed)
        for __i in range(1, self.N):
            __x = (self.mt[__i - 1]).astype(np.uint64)
            __x ^= (__x // np.power(2, self.W - 2)).astype(np.uint64)
            __x *= (self.F)
            __x += np.array(__i % self.N).astype(np.uint64)
            self.mt[__i] = ((__x % np.array(np.power(2, self.W)).astype(np.uint64))).astype(np.uint32)

        self.index = self.N

    def Twist(self):
        for i in range(0, self.N - self.M):
            __y = ((self.mt[i] & self.MASK_UPPER) | (self.mt[i+1] & self.MASK_LOWER)).astype(np.uint32)
            self.mt[i] = (self.mt[i + self.M].astype(np.uint32) ^
                          (__y // 2).astype(np.uint32) ^ (self.A if (__y & 0x01) else 0))

        for __k in range(self.N - self.M, self.N-1):
            __y = ((self.mt[__k] & self.MASK_UPPER) | (self.mt[__k + 1] & self.MASK_LOWER)).astype(np.uint32)
            self.mt[__k] = (self.mt[__k + (self.M - self.N)] ^ (__y // 2) ^ (self.A if (__y & 0x01) else 0))

        __y = ((self.mt[self.N - 1] & self.MASK_UPPER) | (self.mt[0] & self.MASK_LOWER)).astype(np.uint32)
        self.mt[self.N - 1] = (self.mt[self.M - 1] ^ (__y // 2) ^ (self.A if (__y & 0x01) else 0))

        self.index = 0

    def __call__(self):
        i = self.index
        if self.index >= self.N:
            self.Twist()
            i = self.index

        y = self.mt[i]
        self.index = i + 1

        y ^= (self.mt[i] >> self.U)
        y ^= (y << self.S) & self.B
        y ^= (y << self.T) & self.C
        y ^= (y >> self.L)
        return y


class distribution(object):
    def __init__(self, low, high, Generater: random_generate_engine):
        super(distribution, self).__init__()
        self.low = low
        self.high = high
        self.g = Generater

    @abstractmethod
    def getrand(self):
        return self.g()


class uniform_real_distribution(distribution):
    def __init__(self, low, high, Generater):
        super(uniform_real_distribution, self).__init__(low, high, Generater)

    def getrand(self):
        __b = 53
        urng_max = self.g.max()
        urng_min = self.g.min()
        __r = urng_max - urng_min + 1.0
        __log2r = np.log2(__r) / np.log2(2.0)
        __k = max(1, (__b + __log2r - 1) // __log2r)
        __sum = 0
        __tmp = 1
        __ret = 1.0
        for i in range(int(__k)):
            __sum += (float(self.g()) - urng_min) * __tmp
            __tmp *= __r
        __ret = __sum / __tmp
        __ret = __ret * (self.high - self.low) + self.low
        return __ret


class uniform_int_distribution(distribution):
    def __init__(self, low, high, Generater):
        super(uniform_int_distribution, self).__init__(low, high, Generater)

    def getrand(self):
        __urngmin = self.g.min()
        __urngmax = self.g.max()
        __urngrange = __urngmax - __urngmin
        __urange = self.high - self.low

        # downscaling
        __uerange = __urange + 1
        __scaling = int(__urngrange / __uerange)
        __past = int(__uerange * __scaling)
        __ret = self.g() - __urngmin
        while (__ret >= __past):
            __ret = self.g() - __urngmin
        __ret /= __scaling
        return int(__ret) + self.low


def shuffle(d, g, low, high):
    d = d if isinstance(d, torch.Tensor) else torch.tensor(d)
    data = d.clone()

    def swap(a, first, second):
        tmp = a[first].item()
        a[first] = a[second].item()
        a[second] = tmp
    if g is None:
        for i in range(low, high):
            first = i
            second = torch.randint(0, i, [1]).item()
            swap(data, first, second)
    else:
        __urngrange = g.max() - g.min()
        __urange = high - low
        if (__urngrange / __urange >= __urange):
            __i = low + 1
            if ((__urange % 2) == 0):
                tmp_first = __i
                tmp_high = low + uniform_int_distribution(0, 1, g).getrand()
                swap(data, tmp_first, tmp_high)
                __i += 1

            while (__i != high):
                __swap_range = (__i - low) + 1
                __x = uniform_int_distribution(0, __swap_range*(__swap_range+1)-1, g).getrand()
                tmp_first, tmp_second = int(__x / (__swap_range + 1)), int(__x % (__swap_range + 1))
                swap(data, __i, low+tmp_first)
                __i += 1
                swap(data, __i, low+tmp_second)
                __i += 1
            return data
        else:
            # currently not supported
            return data
