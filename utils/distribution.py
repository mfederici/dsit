import torch
from torch.distributions import Categorical
import string


# Utility function
def expand(dist, indices):
    for idx in dist.indices:
        assert idx in indices
    p = dist.p
    dim_to_add = [indices.index(e) for e in indices if e not in dist.indices]
    for dim in sorted(dim_to_add):
        p = p.unsqueeze(dim)
    return p


class DiscreteDistribution:
    def __init__(self, p, indices, condition=None, set_conditions=None):
        '''
        Create a discrete distribution object
        :param p: n-dimentional pytorch tensor specifying a conditional or joint distribution
        :param indices: labels used to identify the dimensions of the tensors (must be a list of n strings)
        :param condition: labels that are on the conditioning side of the distribution
        '''
        self.p = p
        self.indices = indices
        if condition is None:
            condition = []
        self.condition = set(condition)

        if set_conditions == None:
            set_conditions = dict()
        self.set_conditions = set_conditions

    def compose(self, cond_dist):
        '''
        Compose conditional and marginal distribuitons if possible
        :param cond_dist: distribution object to combine with self
        :return: the combined distribution
        '''
        new_indices = self.indices + [e for e in cond_dist.indices if e not in self.indices]

        s = '%s,%s->%s' % (
            ''.join([string.ascii_lowercase[i] for i in range(len(self.indices))]),
            ''.join([string.ascii_lowercase[new_indices.index(e)] for e in cond_dist.indices]),
            ''.join([string.ascii_lowercase[i] for i in range(len(new_indices))]),
        )

        p = torch.einsum(s, self.p, cond_dist.p)

        condition = [e for e in cond_dist.condition if e in self.condition]

        return DiscreteDistribution(p, new_indices, condition=condition, set_conditions=self.set_conditions)

    def marginal(self, a):
        '''
        Marginalize the distribution leaving only the specified components
        :param a: the labels for the variables to keep
        :return: the marginalized distribution
        '''
        return self._marginal(a, {})

    def _marginal(self, a, b):
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        a = set(a)
        b = set(b)

        if b != self.condition:
            raise Exception('Cannot compute p(%s) since p(%s|%s) is unknown' %
                            (','.join(a) + ('|%s' % ','.join(b)),
                             ','.join(self.condition - b),
                             ','.join(set(self.indices) - self.condition)))

        if len(a - set(self.indices)) > 0:
            raise Exception('Cannot compute p(%s) since p(%s|%s) is unknown' %
                            (','.join(a),
                             ','.join(a - set(self.indices)),
                             ','.join(self.indices)))

        p = self.p
        indices = []
        for dim in range(len(self.indices))[::-1]:
            e = self.indices[dim]
            if not (e in a.union(b)):
                p = p.sum(dim)
            else:
                indices = [e] + indices
        dist = DiscreteDistribution(p, indices, condition=b, set_conditions=self.set_conditions)

        return dist

    def conditional(self, a, b):  # p(a|b)
        if b is None or len(b) == 0:
            return self.marginal(a)

        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]

        b = set(b)
        a = set(a)

        c = self.condition.intersection(b)

        assert len(a.intersection(b)) == 0

        if c == b:
            return self._marginal(a, b)
        else:
            p_ab = self.conditional(a.union(b) - c, c)
            p_b = self.conditional(b - c, c)

        p_b_prob = expand(p_b, p_ab.indices)
        p_a_b_prob = torch.zeros(p_ab.p.shape).to(p_b_prob.device)
        p_a_b_prob[p_ab.p > 0] = p_ab.p[p_ab.p > 0] / p_b_prob.expand(p_ab.p.shape)[p_ab.p > 0]

        dist = DiscreteDistribution(p_a_b_prob, p_ab.indices, b, set_conditions=self.set_conditions)

        return dist

    def condition_on(self, a, val):
        '''
        Produce the conditional distribution obtained by selecting the specified values of the variables
        :param a: the variable name to condition on
        :param val: the value to condition on for a
        :return: the conditioned distribution
        '''
        new_indices = [index for index in self.indices if index != a]
        new_cond_indices = [index for index in self.condition if index != a]
        if a in self.condition:
            p_cond = self
        else:
            p_cond = self.conditional(new_indices, a)

        a_dim = p_cond.indices.index(a)
        assert p_cond.p.shape[a_dim] > val and val >= 0

        set_conditions = {k: v for k, v in self.set_conditions.items()}
        set_conditions[a] = val

        return DiscreteDistribution(indices=new_indices,
                                    p=p_cond.p.index_select(a_dim, torch.LongTensor([val])).squeeze(),
                                    condition=new_cond_indices, set_conditions=set_conditions)

    def mi(self, a, b, c=None):  # I(a;b|c)
        '''
        Compute (conditional) mutual information I(a;b|c)
        :param a: the name(s) of the first argument of the mutual information
        :param b: the name(s) of the second argument of the mutual information
        :param c: (optional) the name of the variables used for conditioning
        :return: the value of mutual information in nats
        '''
        if a == b:
            return self.h(a, c)

        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        if isinstance(c, str):
            c = [c]

        if c is None:
            p_abc = self.marginal(a + b)
            p_a_bc = self.conditional(a, b)
            p_a_c = self.marginal(a)
        else:
            p_abc = self.marginal(a + b + c)
            p_a_bc = self.conditional(a, b + c)
            p_a_c = self.conditional(a, c)

        p_a_c_prob = expand(p_a_c, p_a_bc.indices)

        log_ratio = torch.log(p_a_bc.p[p_a_bc.p > 0] / (p_a_c_prob).expand(p_a_bc.p.shape)[p_a_bc.p > 0])

        return (p_abc.p[p_a_bc.p > 0] * log_ratio).sum()

    def compute(self, quantity):
        '''
        Compute the quantity expressed as a string. the formats I(a;b|c), H(a,b|c) for mutual information and entropy
        respectively
        :param quantity: the string indicating which quantity needs to be computed
        :return: the required quantity in nats
        '''
        if quantity[0] == 'H':
            args = quantity.split('(')[1].split(')')[0]
            if '|' in args:
                b = args.split('|')[1].split(',')
                args = args.split('|')[0]
            else:
                b = None
            a = args.split(',')
            value = self.h(a, b)
        elif quantity[0] == 'I':
            args = quantity.split('(')[1].split(')')[0]
            if '|' in args:
                c = args.split('|')[1].split(',')
                args = args.split('|')[0]
            else:
                c = None
            a = args.split(';')[0].split(',')
            b = args.split(';')[1].split(',')
            value = self.mi(a, b, c)
        return value

    def h(self, a, b=None):  # H(a|b)
        '''
        Compute (conditional) entropy
        :param a: the name(s) of the variable(s) for the entropy conputation
        :param b: (optional) the name(s) of the variable(s) used for conditioning
        :return: the required value in nats
        '''
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]

        if b is None:
            p_ab = self.marginal(a)
            p_a_b = p_ab
        else:
            p_ab = self.marginal(a + b)
            p_a_b = self.conditional(a, b)

        return -(p_ab.p * torch.log(p_a_b.p)).sum()

    def sample(self, n=None):
        assert len(self.condition) == 0
        flat_p = self.p.view(-1)

        c_sample = Categorical(probs=flat_p).sample(torch.Size([n]) if n else [])

        sample = {}
        for i, v in enumerate(reversed(self.indices)):
            sample[v] = c_sample % self.p.shape[-(i+1)]
            c_sample = c_sample // self.p.shape[-(i+1)]

        return sample

    def __repr__(self):
        return "p(" + ",".join([n for n in self.indices if n not in self.condition]) + (
            "|%s" % (",".join(self.condition)) + (',' if len(self.condition)>0 else '') + (
                ','.join(['%s=%d' % (k, v) for k, v in self.set_conditions.items()])
            ) if len(self.condition) + len(self.set_conditions) > 0 else "") + ")"



def compute_kl(dist_1, dist_2, cond_1=None):
    '''
    Compute KL (dist_1(support|cond_1)||dist_2(support|cond_2))
    :param dist_1: the distribution for the first argument of the kl
    :param dist_2: the distribution for the second argument of the kl
    :param support: the name(s) of the support variables on which the Kl-divergence has to be computed
    :param cond_1: name(s) of the conditioning for the first distribution
    :param cond_2: name(s) of the conditioning for the second distribution
    :return: the value of the required kl-divergence in nats
    '''
    if cond_1 is None:
        cond_1 = dist_2.condition

    support = set(dist_2.indices)-dist_2.condition
    variables = set(support).union(set(cond_1)).union(set(dist_2.condition))
    dist_1 = dist_1.marginal(list(variables))

    dims_list = dist_1.indices

    cond_1 = dist_1.conditional(support, cond_1)
    cond_1_p = cond_1.p

    cond_2 = dist_2
    cond_2_p = cond_2.p

    p_cond = cond_1_p.permute([cond_1.indices.index(index) for index in dims_list if index in cond_1.indices])
    q_cond = cond_2_p.permute([cond_2.indices.index(index) for index in dims_list if index in cond_2.indices])

    i = 0
    for dim in dims_list:
        if not (dim in cond_1.indices):
            p_cond = p_cond.unsqueeze(i)
        if not (dim in cond_2.indices):
            q_cond = q_cond.unsqueeze(i)
        i += 1

    p = dist_1.p.permute([dims_list.index(index) for index in dist_1.indices])

    log_ratio = p_cond.log() - q_cond.log()
    kl = (p[p > 0] * log_ratio[p > 0]).sum()

    return kl


def compute_ce(dist_1, dist_2):
    '''
    Compute the cross-entropy error of dist_2 according to the distribution defined by dist_1
    :param dist_1: the distribution used for the expectation
    :param dist_2: the distribution inside the expectation
    :return: the cross-entropy in nats
    '''

    variables = set(dist_2.indices)
    dist_1 = dist_1.marginal(list(variables))

    dist_1_joint = dist_1.p
    dims_list = dist_1.indices

    cond_2_p = dist_2.p

    q_cond = cond_2_p.permute([dist_2.indices.index(index) for index in dims_list if index in dist_2.indices])

    i = 0
    for dim in dims_list:
        if not (dim in dist_2.indices):
            q_cond = q_cond.unsqueeze(i)
        i += 1

    p = dist_1_joint.permute([dims_list.index(index) for index in dist_1.indices])

    ce = -(p[p > 0] * q_cond.log()[p > 0]).sum()

    return ce
