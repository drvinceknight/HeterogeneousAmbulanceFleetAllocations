import types
import numpy as np
import objective
import utilisation

## Time units in minutes
raw_travel_times = np.genfromtxt("./test_data/travel_times_matrix.csv", delimiter=",")
beta = objective.get_beta(travel_times=raw_travel_times)
primary_vehicle_travel_times = raw_travel_times / 0.75
secondary_vehicle_travel_times = raw_travel_times / 1.215

R = objective.get_R(
    primary_vehicle_travel_times=primary_vehicle_travel_times,
    secondary_vehicle_travel_times=secondary_vehicle_travel_times,
)
survival_functions = (
    lambda t: 1 / (1 + np.exp(0.26 + 0.139 * t)),
    lambda t: np.heaviside(15 - t, 1),
    lambda t: np.heaviside(60 - t, 1),
)
demand_rates = np.genfromtxt("./test_data/demand.csv", delimiter=",") / 1440
vehicle_locations, pickup_locations = tuple(map(range, raw_travel_times.shape))

weights_single_vehicle = np.array([0, 0, 1])
weights_multiple_vehicles = np.array([1, 1, 0])

primary_survivals, secondary_survivals = objective.get_survival_time_vectors(
    survival_functions, primary_vehicle_travel_times, secondary_vehicle_travel_times
)


# Utilisations and allocations for resource level 61
given_utilisations_primary_61 = np.genfromtxt(
    "./test_data/primary_utilisations_61.csv", delimiter=","
)
given_utilisations_secondary_61 = np.genfromtxt(
    "./test_data/secondary_utilisations_61.csv", delimiter=","
)
allocation_61 = np.genfromtxt("./test_data/allocation_61.csv", delimiter=",").astype(
    np.int64
)


# Utilisations and allocations for resource level 68
given_utilisations_primary_68 = np.genfromtxt(
    "./test_data/primary_utilisations_68.csv", delimiter=","
)
given_utilisations_secondary_68 = np.genfromtxt(
    "./test_data/secondary_utilisations_68.csv", delimiter=","
)
allocation_68 = np.genfromtxt("./test_data/allocation_68.csv", delimiter=",").astype(
    np.int64
)


# Utilisations and allocations for resource level 75
given_utilisations_primary_75 = np.genfromtxt(
    "./test_data/primary_utilisations_75.csv", delimiter=","
)
given_utilisations_secondary_75 = np.genfromtxt(
    "./test_data/secondary_utilisations_75.csv", delimiter=","
)
allocation_75 = np.genfromtxt("./test_data/allocation_75.csv", delimiter=",").astype(
    np.int64
)


# Utilisations and allocations for resource level 82
# Directly from simulation
given_utilisations_primary_82 = np.genfromtxt(
    "./test_data/primary_utilisations_82.csv", delimiter=","
)
given_utilisations_secondary_82 = np.genfromtxt(
    "./test_data/secondary_utilisations_82.csv", delimiter=","
)
allocation_82 = np.genfromtxt("./test_data/allocation_82.csv", delimiter=",").astype(
    np.int64
)


# Utilisations and allocations for resource level 89
given_utilisations_primary_89 = np.genfromtxt(
    "./test_data/primary_utilisations_89.csv", delimiter=","
)
given_utilisations_secondary_89 = np.genfromtxt(
    "./test_data/secondary_utilisations_89.csv", delimiter=","
)
allocation_89 = np.genfromtxt("./test_data/allocation_89.csv", delimiter=",").astype(
    np.int64
)


# Utilisations and allocations for resource level 96
given_utilisations_primary_96 = np.genfromtxt(
    "./test_data/primary_utilisations_96.csv", delimiter=","
)
given_utilisations_secondary_96 = np.genfromtxt(
    "./test_data/secondary_utilisations_96.csv", delimiter=","
)
allocation_96 = np.genfromtxt("./test_data/allocation_96.csv", delimiter=",").astype(
    np.int64
)


def test_paramaters():
    assert raw_travel_times.shape == (67, 261)
    assert beta.shape == (261, 67, 67)
    assert R.shape == (261, 67, 67)
    assert survival_functions[0](0) == 1 / (1 + np.exp(0.26))
    assert survival_functions[1](16) == 0.0
    assert survival_functions[2](16) == 1.0
    assert demand_rates.shape == (3, 261)
    assert len(given_utilisations_primary_61) == 67
    assert len(given_utilisations_primary_68) == 67
    assert len(given_utilisations_primary_75) == 67
    assert len(given_utilisations_primary_82) == 67
    assert len(given_utilisations_primary_89) == 67
    assert len(given_utilisations_primary_96) == 67
    assert len(given_utilisations_secondary_61) == 67
    assert len(given_utilisations_secondary_68) == 67
    assert len(given_utilisations_secondary_75) == 67
    assert len(given_utilisations_secondary_82) == 67
    assert len(given_utilisations_secondary_89) == 67
    assert len(given_utilisations_secondary_96) == 67
    assert sum(allocation_61[:67]) + 1 / 3 * sum(allocation_61[-67:]) == 61
    assert sum(allocation_68[:67]) + 1 / 3 * sum(allocation_68[-67:]) == 68
    assert sum(allocation_75[:67]) + 1 / 3 * sum(allocation_75[-67:]) == 75
    assert sum(allocation_82[:67]) + 1 / 3 * sum(allocation_82[-67:]) == 82
    assert sum(allocation_89[:67]) + 1 / 3 * sum(allocation_89[-67:]) == 89
    assert sum(allocation_96[:67]) + 1 / 3 * sum(allocation_96[-67:]) == 96


def test_objective_function_61():
    """
    Tests the value of the objective function when using an allocation with a resource level of 61
     - When using a simulation, this will achieve a value of 169.9796
     - This objective gives a value of 232.2921043699148
    """
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=allocation_61[:67],
        allocation_secondary=allocation_61[67:],
        given_utilisations_primary=given_utilisations_primary_61,
        given_utilisations_secondary=given_utilisations_secondary_61,
    )
    objective_in_days = g * 1440
    assert np.isclose(objective_in_days, 232.2921043699148)


def test_objective_function_68():
    """
    Tests the value of the objective function when using an allocation with a resource level of 68
     - When using a simulation, this will achieve a value of 208.6921
     - This objective gives a value of 254.8692893372367
    """
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=allocation_68[:67],
        allocation_secondary=allocation_68[67:],
        given_utilisations_primary=given_utilisations_primary_68,
        given_utilisations_secondary=given_utilisations_secondary_68,
    )
    objective_in_days = g * 1440
    assert np.isclose(objective_in_days, 254.8692893372367)


def test_objective_function_75():
    """
    Tests the value of the objective function when using an allocation with a resource level of 75
     - When using a simulation, this will achieve a value of 209.5593
     - This objective gives a value of 254.11456549042802
    """
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=allocation_75[:67],
        allocation_secondary=allocation_75[67:],
        given_utilisations_primary=given_utilisations_primary_75,
        given_utilisations_secondary=given_utilisations_secondary_75,
    )
    objective_in_days = g * 1440
    assert np.isclose(objective_in_days, 254.11456549042802)


def test_objective_function_82():
    """
    Tests the value of the objective function when using an allocation with a resource level of 82
     - When using a simulation, this will achieve a value of 233.7559
     - This objective gives a value of 257.44519956766254
    """
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=allocation_82[:67],
        allocation_secondary=allocation_82[67:],
        given_utilisations_primary=given_utilisations_primary_82,
        given_utilisations_secondary=given_utilisations_secondary_82,
    )
    objective_in_days = g * 1440
    assert np.isclose(objective_in_days, 257.44519956766254)


def test_objective_function_89():
    """
    Tests the value of the objective function when using an allocation with a resource level of 89
     - When using a simulation, this will achieve a value of 242.9359
     - This objective gives a value of 260.2749321977294
    """
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=allocation_89[:67],
        allocation_secondary=allocation_89[67:],
        given_utilisations_primary=given_utilisations_primary_89,
        given_utilisations_secondary=given_utilisations_secondary_89,
    )
    objective_in_days = g * 1440
    assert np.isclose(objective_in_days, 260.2749321977294)


def test_objective_function_96():
    """
    Tests the value of the objective function when using an allocation with a resource level of 96
     - When using a simulation, this will achieve a value of 246.4273
     - This objective gives a value of 260.2722627389342
    """
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=allocation_96[:67],
        allocation_secondary=allocation_96[67:],
        given_utilisations_primary=given_utilisations_primary_96,
        given_utilisations_secondary=given_utilisations_secondary_96,
    )
    objective_in_days = g * 1440
    assert np.isclose(objective_in_days, 260.2722627389342)


def test_objective_function_with_allocation_dependent_utilisation():
    """
    Tests the value of the objective function when using a utilisation function
    that uses the allocation.
    """
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.constant_utilisation,
        allocation_primary=allocation_96[:67],
        allocation_secondary=allocation_96[67:],
        utilisation_rate_primary=0.4,
        utilisation_rate_secondary=0.7,
    )
    objective_in_days = g * 1440
    assert np.isclose(objective_in_days, 255.08170500308506)
