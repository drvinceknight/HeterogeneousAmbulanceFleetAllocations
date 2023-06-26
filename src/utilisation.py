import numpy as np


def constant_utilisation(
    allocation_primary,
    allocation_secondary,
    utilisation_rate_primary,
    utilisation_rate_secondary,
    **kwargs
):
    """
    Returns two vectors:
     + a vector of constant utilisations for primary vehicles
     + a vector of constant utilisations for secondary vehicles
    """
    primary_utilisations = np.array(
        [utilisation_rate_primary for _ in allocation_primary]
    )
    secondary_utilisations = np.array(
        [utilisation_rate_secondary for _ in allocation_secondary]
    )
    return primary_utilisations, secondary_utilisations


def given_utilisations(
    given_utilisations_primary, given_utilisations_secondary, **kwargs
):
    """
    Returns two vectors:
     + a vector of given primary utilisations
     + a vector of given secondary utilisations
    """
    primary_utilisations = np.array(given_utilisations_primary)
    secondary_utilisations = np.array(given_utilisations_secondary)
    return primary_utilisations, secondary_utilisations


def proportional_utilisations(
    allocation_primary,
    allocation_secondary,
    demand_rates,
    service_rate_primary,
    service_rate_secondary,
    **kwargs
):
    """
    Returns two vectors:
     + a vector the proportions the primary demand equally across the allocation
     + a vector the proportions the secondary demand equally across the allocation
    """
    total_primary_demand = demand_rates.sum()
    total_secondary_demand = demand_rates[:-1].sum()
    primary_utilisations = np.array(
        [
            0 if z == 0 else total_primary_demand / (service_rate_primary * z)
            for z in allocation_primary
        ]
    )
    secondary_utilisations = np.array(
        [
            0 if z == 0 else total_secondary_demand / (service_rate_secondary * z)
            for z in allocation_secondary
        ]
    )
    return primary_utilisations, secondary_utilisations
