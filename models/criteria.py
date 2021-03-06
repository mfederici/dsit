# class for a general training criterion: -I(y;z) + reg*r(y,e,x,z)
class Criterion:
    def __init__(self, reg):
        self.reg = reg

    def compute_loss(self, dist):
        # the loss is composed of two terms
        #   1) the amount of predictive information 'I(y;z)' that needs to be maximized
        #   2) a regularization term 'r' with strength defined by 'reg'
        return -dist.mi('y', 'z') + self.reg * self.r(dist)

    def r(self, dist):
        raise NotImplemented


# Information Bottleneck Criterion: -I(y;z) + reg*I(x;z)
class InformationBottleneckCriterion(Criterion):
    def r(self, dist):
        return dist.mi('x', 'z')


# Independence Criterion: -I(y;z) + reg*I(e;z)
class IndependenceCriterion(Criterion):
    def r(self, dist):
        return dist.mi('e', 'z')


# Sufficiency Criterion: -I(y;z) + reg*I(e;y|z)
class SufficiencyCriterion(Criterion):
    def r(self, dist):
        return dist.mi('e', 'y', 'z')


# Separation Criterion: -I(y;z) + reg*I(e;z|y)
class SeparationCriterion(Criterion):
    def r(self, dist):
        return dist.mi('e', 'z', 'y')
