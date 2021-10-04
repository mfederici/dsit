import math
import numpy as np


# Schedulers for beta
class Scheduler:
    def __call__(self, **kwargs):
        raise NotImplemented()


class ConstantScheduler(Scheduler):
    def __init__(self, value):
        self.value = value

    def __call__(self, iteration):
        return self.value


class LinearScheduler(Scheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0):
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.m = (end_value - start_value) / float(n_iterations)

    def __call__(self, iteration):
        if iteration > self.start_iteration + self.n_iterations:
            return self.end_value
        elif iteration <= self.start_iteration:
            return self.start_value
        else:
            return (iteration - self.start_iteration) * self.m + self.start_value


class LinearIncrementScheduler(Scheduler):
    def __init__(self, start_value, increment_by, start_iteration=0):
        self.start_value = start_value
        self.start_iteration = start_iteration
        self.increment_by = increment_by

    def __call__(self, iteration):
        if iteration <= self.start_iteration:
            return self.start_value
        else:
            return (iteration - self.start_iteration) * self.increment_by + self.start_value
