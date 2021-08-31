import torch
import numpy as np

from tqdm.auto import tqdm
from torch.optim import Adam

from code.discrete_distributions import compute_ce


def train(encoder, criterion, train_dist, test_dist,
          step_iterations=1000, initial_iterations=5000, log_every=100, tollerance=1e-4, lr=1e-3, verbose=False):
    logs = []

    # Initialize the ADAM optimizer =
    opt = Adam(encoder.parameters(), lr=lr)

    iterations = initial_iterations
    iteration = 0
    last_train_ce = None
    last_test_ce = None
    while True:
        for i in tqdm(range(iterations)) if verbose else range(iterations):
            loss = criterion.compute_loss(encoder(train_dist))

            if i % log_every == 0:
                with torch.no_grad():
                    logs.append({
                        'Test Cross-entropy': compute_ce(encoder(test_dist).marginal(['y', 'z']),
                                                         encoder(train_dist).conditional('y', 'z')).item(),
                        'Train Cross-entropy': encoder(train_dist).h('y', 'z').item(),
                        'iteration': iteration
                    })

            opt.zero_grad()
            loss.backward()
            opt.step()
            iteration += 1

        train_ce = encoder(train_dist).h('y', 'z').item()
        test_ce = compute_ce(encoder(test_dist).marginal(['y', 'z']),
                             encoder(train_dist).conditional('y', 'z')).item()

        if not (last_train_ce == None):
            distance = np.sqrt((train_ce - last_train_ce) ** 2 + (test_ce - last_test_ce) ** 2)
            if distance <= tollerance:
                if verbose:
                    tqdm.write('Done')
                break
            elif verbose:
                tqdm.write('Distance: %f>%f' % (distance, tollerance))

        last_train_ce = train_ce
        last_test_ce = test_ce
        iterations = step_iterations

    return logs
