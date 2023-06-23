import numpy as np


def constant_utilisation_primary(
    allocation_primary, utilisation_rate_primary, **kwargs
):
    """
    Returns a vector of constant utilisations for primary vehicles
    """
    return np.array([utilisation_rate_primary for _ in allocation_primary])


def constant_utilisation_secondary(
    allocation_secondary, utilisation_rate_secondary, **kwargs
):
    """
    Returns a vector of constant utilisations for secondary vehicles
    """
    return np.array([utilisation_rate_secondary for _ in allocation_secondary])


def given_utilisations_primary(given_utilisations_primary, **kwargs):
    """
    Returns a vector of given primary utilisations
    """
    return np.array(given_utilisations_primary)


def given_utilisations_secondary(given_utilisations_secondary, **kwargs):
    """
    Returns a vector of given secondary utilisations
    """
    return np.array(given_utilisations_secondary)


def proportional_utilisations_primary(
    allocation_primary, demand_rates, service_rate_primary, **kwargs
):
    """
    Returns a vector the proportions the primary demand equally across the allocation
    """
    total_demand = demand_rates.sum()
    return np.array(
        [
            0 if z == 0 else total_demand / (service_rate_primary * z)
            for z in allocation_primary
        ]
    )


def proportional_utilisations_secondary(
    allocation_secondary, demand_rates, service_rate_secondary, **kwargs
):
    """
    Returns a vector the proportions the secondary demand equally across the allocation
    """
    total_demand = demand_rates[:-1].sum()
    return np.array(
        [
            0 if z == 0 else total_demand / (service_rate_secondary * z)
            for z in allocation_secondary
        ]
    )
