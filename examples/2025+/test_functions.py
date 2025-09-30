#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:13:23 2025

@author: surero
"""
import numpy as np
import scipy

class sinfunc:
    def __init__(self):
        self.data_name = "sinfunc"
        self.zlim = np.array([[0, 1], [0, 1]])
        self.theta_true = np.array([0.5]) # np.array([np.pi / 5])
        self.real_data = None
        self.out = [("f", float)]
        self.d = 1
        self.p = 2
        self.dx = 1
        self.dt = 1
        self.x = None
        self.sigma2 = 0.1

    def function(self, x, theta):
        f = np.sin(10 * x - 5 * theta)
        return f

    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1])
        var_noise = self.noise(thetas[0], thetas[1])
        noise = persis_info["rand_stream"].normal(0, np.sqrt(var_noise), 1)
        f += noise
        return f

    def realdata(self, x, seed):
        self.x = x
        self.d = len(x)
        self.obsvar = np.diag(np.repeat(self.sigma2, self.d))

        M = np.array([self.function(x, self.theta_true) for x in self.x], dtype=float).reshape(1, self.d)
        if seed is not None:
            rand_stream = np.random.default_rng(seed)
            R = rand_stream.normal(0, np.sqrt(self.sigma2), size=(1, self.d))
            self.real_data = M + R
        else:
            self.real_data = M

    def noise(self, x, theta):
        alpha = 10
        min_value = 0.01#0.05
        max_value = 0.1#0.3

        weight = 1 / (1 + np.exp(-alpha * (x - 0.5)))  # Sigmoid transition
        value = min_value + (max_value - min_value) * weight  # Scale between 0.1 and 10
        return value


class unimodalx:
    def __init__(self):
        self.data_name = "unimodalx"
        self.zlim = np.array([[0, 1], [0, 1], [0, 1]])
        self.real_data = None
        self.out = [("f", float)]
        self.d = 1
        self.p = 3
        self.x = np.arange(0, self.d)[:, None]
        self.theta_true = np.array([0.5, 0.5])
        self.sigma2 = 0.1
        self.dx = 1
        self.dt = 2

    def function(self, x, t1, t2):
        t1 = -10 + t1 * 20
        t2 = -10 + t2 * 20
        f = 0.26 * (t1**2 + t2**2) - 0.48 * t1 * t2 + (2 * x - 1)
        return f

    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1], thetas[2])
        V = self.noise(thetas[0], thetas[1], thetas[2])
        R = persis_info["rand_stream"].normal(0, np.sqrt(V), 1)
        f += R
        return f

    def noise(self, x, t1, t2):
        V = 0.01 + 1.2 * (t1**2 + t2**2) * 2
        V = V
        return V

    def realdata(self, x, seed):
        self.x = x
        self.d = len(x)
        self.obsvar = np.diag(np.repeat(self.sigma2, self.d))
        M = np.array([self.function(x, *self.theta_true) for x in self.x], dtype=float)
        if seed is not None:
            rand_stream = np.random.default_rng(seed)
            R = rand_stream.normal(0, np.sqrt(self.sigma2), size=(1, self.d))
            self.real_data = M + R
        else:
            self.real_data = M


class bimodalx:
    def __init__(self):
        self.data_name = "bimodalx"
        self.zlim = np.array([[0, 1], [0, 1], [0, 1]])
        self.theta_true = np.array([0.35, 0.35])
        self.real_data = None
        self.out = [("f", float)]
        self.p = 3
        self.d = 1
        self.x = np.arange(0, self.d)[:, None]
        self.sigma2 = 0.05
        self.dx = 1
        self.dt = 2

    def function(self, x, t1, t2):
        mu1 = (0.35, 0.35)
        mu2 = (0.65, 0.65)
        sigma = 0.15
        term1 = np.exp(-((t1 - mu1[0]) ** 2 + (t2 - mu1[1]) ** 2) / sigma**2)
        term2 = np.exp(-((t1 - mu2[0]) ** 2 + (t2 - mu2[1]) ** 2) / sigma**2)
        f = (term1 + term2) + (2 * x - 1)
        return f

    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1], thetas[2])
        V = self.noise(thetas[0], thetas[1], thetas[2])
        R = persis_info["rand_stream"].normal(0, np.sqrt(V), 1)
        f += R
        return f

    def noise(self, x, t1, t2):
        cov = np.array([[0.05, 0], [0, 0.05]])
        var = scipy.stats.multivariate_normal(mean=[0.85, 0.85], cov=cov)
        return 0.1*var.pdf(np.array([t1, t2]))

    def realdata(self, x, seed):
        self.x = x
        self.d = len(x)
        self.obsvar = np.diag(np.repeat(self.sigma2, self.d))
        M = np.array([self.function(x, *self.theta_true) for x in self.x], dtype=float)
        if seed is not None:
            rand_stream = np.random.default_rng(seed)
            R = rand_stream.normal(0, np.sqrt(self.sigma2), size=(1, self.d))
            self.real_data = M + R
        else:
            self.real_data = M

class braninx:
    def __init__(self):
        self.data_name = "braninx"
        self.zlim = np.array([[0, 1], [0, 1], [0, 1]])
        self.real_data = None
        self.out = [("f", float)]
        self.d = 1
        self.p = 3
        self.x = np.arange(0, self.d)[:, None]
        self.theta_true = np.array([0.9613333333333334, 0.16466666666666668])
        self.sigma2 = 5
        self.dx = 1
        self.dt = 2

    def function(self, x, t1, t2):
        t1 = -5 + 15 * t1
        t2 = 15 * t2
        f = (
            (t2 - (5.1 / (4 * np.pi**2)) * (t1**2) + (5 / np.pi) * t1 - 6) ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(t1)
            + 10
            + (2 * x - 1)
        )
        return f

    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1], thetas[2])
        V = self.noise(thetas[0], thetas[1], thetas[2])
        R = persis_info["rand_stream"].normal(0, np.sqrt(V), 1)
        f += R
        return f

    def noise(self, x, t1, t2):
        alpha = 10
        min_value = 0.1
        max_value = 15

        weight = 1 / (1 + np.exp(-alpha * (t2 - 0.5)))
        value = min_value + (max_value - min_value) * weight

        return value

    def realdata(self, x, seed):
        self.x = x
        self.d = len(x)
        self.obsvar = np.diag(np.repeat(self.sigma2, self.d))
        M = np.array([self.function(x, *self.theta_true) for x in self.x], dtype=float)
        if seed is not None:
            rand_stream = np.random.default_rng(seed)
            R = rand_stream.normal(0, np.sqrt(self.sigma2), size=(1, self.d))
            self.real_data = M + R
        else:
            self.real_data = M


class pritam:
    def __init__(self):
        self.data_name = "pritam"
        self.zlim = np.array([[0, 1], [0, 1], [0, 1]])
        self.theta_true = np.array([0.5])
        self.out = [("f", float)]
        self.d = 1
        self.p = 3
        self.real_data = None
        self.dx = 2
        self.sigma2 = 10 #0.5**2
        self.dx = 2
        self.dt = 1

    def function(self, x1, x2, theta1):
        f = (30 + 5 * x1 * np.sin(5 * x1)) * (6 * theta1 + 1 + np.exp(-5 * x2))
        return f

    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1], thetas[2])
        V = self.noise(thetas[0], thetas[1], thetas[2])
        R = persis_info["rand_stream"].normal(0, np.sqrt(V), 1)
        f += R
        return f

    def realdata(self, x, seed):
        self.x = x
        self.d = len(x)
        self.obsvar = np.diag(np.repeat(self.sigma2, self.d))
        M = np.array([self.function(x[0], x[1], self.theta_true) for x in self.x], dtype=float).reshape(1, self.d)
        if seed is not None:
            rand_stream = np.random.default_rng(seed)
            R = rand_stream.normal(0, np.sqrt(self.sigma2), size=(1, self.d))
            self.real_data = M + R
        else:
            self.real_data = M
    
    def noise(self, x1, x2, t1):
        # return 5

        cov = np.array([[0.1, 0], [0, 0.1]])
        var = scipy.stats.multivariate_normal(mean=[0.5, 0.5], cov=cov)
        return (15*t1)*var.pdf(np.array([x1, x2]))

