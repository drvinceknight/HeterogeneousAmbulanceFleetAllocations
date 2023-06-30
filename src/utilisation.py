import numpy as np
import objective
import scipy.optimize  # type: ignore
import warnings

warnings.simplefilter("ignore")


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


def get_lambda_differences_primary(
    lhs, service_rate_primary, allocation_primary, beta, demand_rates
):
    """
    Returns the difference between the LHS and RHS of the primary demand rates relationship equation
    """
    utilisations = np.divide(
        lhs / service_rate_primary,
        allocation_primary,
        out=np.zeros_like(lhs),
        where=allocation_primary != 0,
    )
    all_closer = objective.get_all_same_closer_busy_vector(
        utilisations, allocation_primary, beta
    )
    not_busy = objective.get_is_not_busy_vector(utilisations, allocation_primary)
    rhs = (demand_rates.sum(axis=0) * (not_busy * all_closer.T).T).sum(axis=1)
    return rhs - lhs


def get_lambda_differences_secondary(
    lhs,
    service_rate_secondary,
    allocation_secondary,
    allocation_primary,
    utilisations_primary,
    beta,
    R,
    demand_rates,
):
    """
    Returns the difference between the LHS and RHS of the secondary demand rates relationship equation
    """
    utilisations = np.divide(
        lhs / service_rate_secondary,
        allocation_secondary,
        out=np.zeros_like(lhs),
        where=allocation_secondary != 0,
    )
    all_closer = objective.get_all_same_closer_busy_vector(
        utilisations, allocation_secondary, beta
    )
    not_busy = objective.get_is_not_busy_vector(utilisations, allocation_secondary)
    all_primary_closer = objective.get_all_primary_closer_busy_vector(
        utilisations_primary, allocation_primary, R
    )
    rhs = (
        demand_rates[:-1].sum(axis=0) * (not_busy * all_closer.T * all_primary_closer).T
    ).sum(axis=1)
    return rhs - lhs


def solve_utilisations_primary(
    allocation_primary,
    beta,
    demand_rates,
    service_rate_primary,
    overall_utilisation_limit=0.99,
    **kwargs
):
    """
    Finds the utilisations by solving the equations relating demans to each ambulance location.
    """
    total_demand = demand_rates.sum()
    if (
        total_demand / (service_rate_primary * sum(allocation_primary))
        > overall_utilisation_limit
    ):
        return np.array([overall_utilisation_limit for _ in allocation_primary])

    starting_lambdas = np.array(
        [total_demand / len(allocation_primary) for _ in allocation_primary]
    )
    final_lambdas = scipy.optimize.fsolve(
        get_lambda_differences_primary,
        starting_lambdas,
        args=(service_rate_primary, allocation_primary, beta, demand_rates),
    )
    utilisations = np.divide(
        final_lambdas,
        allocation_primary * service_rate_primary,
        out=np.zeros_like(final_lambdas),
        where=allocation_primary != 0,
    )
    return utilisations


def solve_utilisations_secondary(
    allocation_secondary,
    allocation_primary,
    utilisations_primary,
    beta,
    R,
    demand_rates,
    service_rate_secondary,
    overall_utilisation_limit=0.99,
    **kwargs
):
    """
    Finds the utilisations by solving the equations relating demans to each ambulance location.
    """
    total_demand = demand_rates[:-1].sum()
    if (
        total_demand / (service_rate_secondary * sum(allocation_secondary))
        > overall_utilisation_limit
    ):
        return np.array([overall_utilisation_limit for _ in allocation_primary])

    starting_lambdas = np.array(
        [total_demand / len(allocation_secondary) for _ in allocation_secondary]
    )
    final_lambdas = scipy.optimize.fsolve(
        get_lambda_differences_secondary,
        starting_lambdas,
        args=(
            service_rate_secondary,
            allocation_secondary,
            allocation_primary,
            utilisations_primary,
            beta,
            R,
            demand_rates,
        ),
    )
    utilisations = np.divide(
        final_lambdas,
        allocation_secondary * service_rate_secondary,
        out=np.zeros_like(final_lambdas),
        where=allocation_secondary != 0,
    )
    return utilisations


def solve_utilisations(
    allocation_primary,
    allocation_secondary,
    beta,
    R,
    demand_rates,
    service_rate_primary,
    service_rate_secondary,
    **kwargs
):
    """
    Utilises scipy.optimize.fsolve to find utilisations by finding roots of the demand-utilisation relationships.
    Returns two vectors:
     + a vector the solved utilisations for primary vehicles
     + a vector the solved utilisations for secondary vehicles
    """
    primary_utilisations = solve_utilisations_primary(
        allocation_primary=allocation_primary,
        beta=beta,
        demand_rates=demand_rates,
        service_rate_primary=service_rate_primary,
        **kwargs
    )
    secondary_utilisations = solve_utilisations_secondary(
        allocation_secondary=allocation_secondary,
        allocation_primary=allocation_primary,
        utilisations_primary=primary_utilisations,
        beta=beta,
        R=R,
        demand_rates=demand_rates,
        service_rate_secondary=service_rate_secondary,
        **kwargs
    )
    return primary_utilisations, secondary_utilisations
