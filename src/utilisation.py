import numpy as np


def constant_utilisation(allocation, utilisation_rate, **kwargs):
    """
    Returns a vector of constant utilisations
    """
    return np.array([utilisation_rate for _ in allocation])


def given_utilisations(allocation, given_utilisations, **kwargs):
    """
    Returns a vector of given utilisations
    """
    return np.array(given_utilisations)


def proportional_utilisations(allocation, demand_rates, service_rate, **kwargs):
    """
    Returns a vector the proportions the demand equally across the allocation
    """
    total_demand = demand_rates.sum()
    return np.array(
        [0 if z == 0 else total_demand / (service_rate * z) for z in allocation]
    )
