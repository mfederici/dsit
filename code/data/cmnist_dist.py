import torch
from utils.distribution import DiscreteDistribution

CMNIST_NAME = 'CMNIST'
D_CMNIST_NAME = 'd-CMNIST'
Y_CMNIST_NAME = 'y-CMNIST'
CMNIST_VERSIONS = [CMNIST_NAME, D_CMNIST_NAME, Y_CMNIST_NAME]

# function to make the CMNIST, d-CMNIST, and y-CMNIST distributions
def make_joint_distribution(name):
    assert name in ['CMNIST', 'd-CMNIST', 'y-CMNIST']

    # p(t|e)
    p_t_e = torch.zeros(2, 3)

    # p(t=1|e=0)=p(t=1|e=1)=1
    p_t_e[1, 0] = p_t_e[1, 1] = 1
    # p(t=0|e=2)=1
    p_t_e[0, 2] = 1

    p_t_e = DiscreteDistribution(p_t_e, ['t', 'e'], condition=['e'])

    # p(c|y,e)
    p_c_ye = torch.zeros(2, 2, 3)

    # p(c=1|y=1,e=0)=p(c=0|y=10e=0)0.9
    p_c_ye[1, 1, 0] = p_c_ye[0, 0, 0] = 0.9
    p_c_ye[0, 1, 0] = p_c_ye[1, 0, 0] = 0.1

    # p(c=1|y=1,e=1)=p(c=0|y=0,e=1)=0.8
    p_c_ye[1, 1, 1] = p_c_ye[0, 0, 1] = 0.8
    p_c_ye[0, 1, 1] = p_c_ye[1, 0, 1] = 0.2

    # p(c=1|y=1,e=2)=p(c=0|y=0,e=2)=0.1
    p_c_ye[1, 1, 2] = p_c_ye[0, 0, 2] = 0.1
    p_c_ye[0, 1, 2] = p_c_ye[1, 0, 2] = 0.9

    p_c_ye = DiscreteDistribution(p_c_ye, ['c', 'y', 'e'], condition=['y', 'e'])

    # add x as a combination of digit and color
    p_x_cd = torch.zeros(20, 2, 10)

    for c in range(2):
        for d in range(10):
            p_x_cd[c * 10 + d, c, d] = 1
    p_x_cd = DiscreteDistribution(p_x_cd, ['x', 'c', 'd'], condition=['c', 'd'])

    #######################
    # CMNIST distribution #
    #######################
    if name == 'CMNIST':
        # p(d)
        p_d = torch.zeros(10)

        # p(d=1)=..=p(d=9)=0.1
        p_d[:] = 0.1

        p_d = DiscreteDistribution(p_d, ['d'])

        # p(y|d)
        p_y_d = torch.zeros(2, 10)

        # p(y=0|d<5)= p(y=1|d>4) = 0.75
        p_y_d[0, :5] = p_y_d[1, 5:] = 0.75
        p_y_d[1, :5] = p_y_d[0, 5:] = 0.25

        p_y_d = DiscreteDistribution(p_y_d, ['y', 'd'], condition=['d'])

        # p(e)
        p_e = torch.zeros(3)

        # p(e=0)=p(e=1)=p(e=2)=1/3
        p_e[:] = 1./3

        p_e = DiscreteDistribution(p_e, ['e'])

        p_yde = p_d.compose(p_y_d).compose(p_e)

    #########################
    # d-CMNIST distribution #
    #########################
    if name == 'd-CMNIST':
        # p(d|e)
        p_d_e = torch.zeros(10, 3)

        # p(d<5|e=0)=0.6
        p_d_e[:5, 0] = 0.6 / 5
        p_d_e[5:, 0] = 0.4 / 5

        # p(d<5|e=1)=0.2/5
        p_d_e[:5, 1] = 0.2 / 5
        p_d_e[5:, 1] = 0.8 / 5

        # p(d=1|e=2)=..=p(d=9|e=2)=0.1
        p_d_e[:, 2] = 0.1

        p_d_e = DiscreteDistribution(p_d_e, ['d', 'e'], condition=['e'])

        # p(y|d)
        p_y_d = torch.zeros(2, 10)

        # p(y=0|d<5)= p(y=1|d>4) = 0.75
        p_y_d[0, :5] = p_y_d[1, 5:] = 0.75
        p_y_d[1, :5] = p_y_d[0, 5:] = 0.25

        p_y_d = DiscreteDistribution(p_y_d, ['y', 'd'], condition=['d'])

        # p(e)
        p_e = torch.zeros(3)

        # p(e=0)=1/2
        p_e[0] = 0.5
        # p(e=1)=1/6
        p_e[1] = 1./6
        # p(e=2)=1/3
        p_e[2] = 1./3

        p_e = DiscreteDistribution(p_e, ['e'])

        p_yde = p_d_e.compose(p_e).compose(p_y_d)

    #########################
    # y-CMNIST distribution #
    #########################
    if name == 'y-CMNIST':
        # p(d|y)
        p_d_y = torch.zeros(10, 2)

        # p(d<5|y=0)=0.75/5
        p_d_y[:5, 0] = 0.75 / 5
        p_d_y[5:, 0] = 0.25 / 5

        # p(d<5|y=1)=0.25/5
        p_d_y[:5, 1] = 0.25 / 5
        p_d_y[5:, 1] = 0.75 / 5

        p_d_y = DiscreteDistribution(p_d_y, ['d', 'y'], condition=['y'])

        # p(y|e)
        p_y_e = torch.zeros(2, 3)

        # p(y=0|e=0) = 0.6
        p_y_e[0, 0] = 0.6
        p_y_e[1, 0] = 0.4

        # p(y=0|e=1) = 0.2
        p_y_e[0, 1] = 0.2
        p_y_e[1, 1] = 0.8

        # p(y=0|e=2) = 0.5
        p_y_e[0, 2] = p_y_e[1, 2] = 0.5

        p_y_e = DiscreteDistribution(p_y_e, ['y', 'e'], condition=['e'])

        # p(e)
        p_e = torch.zeros(3)

        # p(e=0)=1/2
        p_e[0] = 0.5
        # p(e=1)=1/6
        p_e[1] = 1. / 6
        # p(e=2)=1/3
        p_e[2] = 1. / 3

        p_e = DiscreteDistribution(p_e, ['e'])

        p_yde = p_y_e.compose(p_e).compose(p_d_y)

    return p_yde.compose(p_x_cd).compose(p_c_ye).compose(p_t_e)

