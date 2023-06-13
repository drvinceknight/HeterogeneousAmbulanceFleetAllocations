import types

import numpy as np

import objective

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
    lambda t: t <= 15,
    lambda t: t <= 60,
)
demand_rates = np.genfromtxt("./test_data/demand.csv", delimiter=",") / 1440
vehicle_locations, pickup_locations = tuple(map(range, raw_travel_times.shape))
patient_type_partitions = ((1, 2), (0,))


# Utilisations and allocations for resource level 68
primary_vehicle_station_utilisation_68 = np.genfromtxt(
    "./test_data/primary_utilisations_68.csv", delimiter=","
)  # Directly from simulation
secondary_vehicle_station_utilisation_68 = np.genfromtxt(
    "./test_data/secondary_utilisations_68.csv", delimiter=","
)  # Directly from simulation
allocation_68 = np.genfromtxt("./test_data/allocation_68.csv", delimiter=",")


# Utilisations and allocations for resource level 75
primary_vehicle_station_utilisation_75 = np.genfromtxt(
    "./test_data/primary_utilisations_75.csv", delimiter=","
)  # Directly from simulation
secondary_vehicle_station_utilisation_75 = np.genfromtxt(
    "./test_data/secondary_utilisations_75.csv", delimiter=","
)  # Directly from simulation
allocation_75 = np.genfromtxt("./test_data/allocation_75.csv", delimiter=",")


# Utilisations and allocations for resource level 82
primary_vehicle_station_utilisation_82 = np.genfromtxt(
    "./test_data/primary_utilisations_82.csv", delimiter=","
)  # Directly from simulation
secondary_vehicle_station_utilisation_82 = np.genfromtxt(
    "./test_data/secondary_utilisations_82.csv", delimiter=","
)  # Directly from simulation
allocation_82 = np.genfromtxt("./test_data/allocation_82.csv", delimiter=",")


# Utilisations and allocations for resource level 89
primary_vehicle_station_utilisation_89 = np.genfromtxt(
    "./test_data/primary_utilisations_89.csv", delimiter=","
)  # Directly from simulation
secondary_vehicle_station_utilisation_89 = np.genfromtxt(
    "./test_data/secondary_utilisations_89.csv", delimiter=","
)  # Directly from simulation
allocation_89 = np.genfromtxt("./test_data/allocation_89.csv", delimiter=",")


# Utilisations and allocations for resource level 96
primary_vehicle_station_utilisation_96 = np.genfromtxt(
    "./test_data/primary_utilisations_96.csv", delimiter=","
)  # Directly from simulation
secondary_vehicle_station_utilisation_96 = np.genfromtxt(
    "./test_data/secondary_utilisations_96.csv", delimiter=","
)  # Directly from simulation
allocation_96 = np.genfromtxt("./test_data/allocation_96.csv", delimiter=",")


def test_paramaters():
    assert raw_travel_times.shape == (67, 261)
    assert type(beta) is dict
    assert len(beta) == 67**2 * 261
    assert type(R) is dict
    assert len(R) == 67**2 * 261
    assert survival_functions[0](0) == 1 / (1 + np.exp(0.26))
    assert survival_functions[1](16) is False
    assert survival_functions[2](16) is True
    assert demand_rates.shape == (3, 261)
    assert len(primary_vehicle_station_utilisation_68) == 67
    assert len(primary_vehicle_station_utilisation_75) == 67
    assert len(primary_vehicle_station_utilisation_82) == 67
    assert len(primary_vehicle_station_utilisation_89) == 67
    assert len(primary_vehicle_station_utilisation_96) == 67
    assert len(secondary_vehicle_station_utilisation_68) == 67
    assert len(secondary_vehicle_station_utilisation_75) == 67
    assert len(secondary_vehicle_station_utilisation_82) == 67
    assert len(secondary_vehicle_station_utilisation_89) == 67
    assert len(secondary_vehicle_station_utilisation_96) == 67
    assert sum(allocation_68[:67]) + 1 / 3 * sum(allocation_68[-67:]) == 68
    assert sum(allocation_75[:67]) + 1 / 3 * sum(allocation_75[-67:]) == 75
    assert sum(allocation_82[:67]) + 1 / 3 * sum(allocation_82[-67:]) == 82
    assert sum(allocation_89[:67]) + 1 / 3 * sum(allocation_89[-67:]) == 89
    assert sum(allocation_96[:67]) + 1 / 3 * sum(allocation_96[-67:]) == 96


def test_objective_function_68():
    """
    Tests the value of the objective function when using an allocation with a resource level of 68
     - When using a simulation, this will achieve a value of 144.5987.
     - This objective gives a value of 124.69164131282206
    """
    g = objective.get_objective_function(
        pickup_locations=pickup_locations,
        patient_type_partitions=patient_type_partitions,
        survival_functions=survival_functions,
        primary_travel_times=primary_vehicle_travel_times,
        secondary_travel_times=secondary_vehicle_travel_times,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation_68,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation_68,
        vehicle_locations=vehicle_locations,
        is_station_closer_to_pickup_location=beta,
        is_vehicle_type_closer_to_pickup_location=R,
        weights=(1, 1, 1),
        demands=demand_rates,
    )
    assert isinstance(g, types.FunctionType)
    assert g([0 for _ in range(67 * 2)]) == 0
    objective_in_days = g(allocation_68) * 1440
    assert np.isclose(
        objective_in_days, 124.69164131282206
    )

def test_objective_function_75():
    """
    Tests the value of the objective function when using an allocation with a resource level of 75
     - When using a simulation, this will achieve a value of 145.8549.
     - This objective gives a value of 138.87802089723246
    """
    g = objective.get_objective_function(
        pickup_locations=pickup_locations,
        patient_type_partitions=patient_type_partitions,
        survival_functions=survival_functions,
        primary_travel_times=primary_vehicle_travel_times,
        secondary_travel_times=secondary_vehicle_travel_times,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation_75,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation_75,
        vehicle_locations=vehicle_locations,
        is_station_closer_to_pickup_location=beta,
        is_vehicle_type_closer_to_pickup_location=R,
        weights=(1, 1, 1),
        demands=demand_rates,
    )
    assert isinstance(g, types.FunctionType)
    assert g([0 for _ in range(67 * 2)]) == 0
    objective_in_days = g(allocation_75) * 1440
    assert np.isclose(
        objective_in_days, 138.87802089723246
    )

def test_objective_function_82():
    """
    Tests the value of the objective function when using an allocation with a resource level of 82
     - When using a simulation, this will achieve a value of 145.6902
     - This objective gives a value of 131.23026163719774
    """
    g = objective.get_objective_function(
        pickup_locations=pickup_locations,
        patient_type_partitions=patient_type_partitions,
        survival_functions=survival_functions,
        primary_travel_times=primary_vehicle_travel_times,
        secondary_travel_times=secondary_vehicle_travel_times,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation_82,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation_82,
        vehicle_locations=vehicle_locations,
        is_station_closer_to_pickup_location=beta,
        is_vehicle_type_closer_to_pickup_location=R,
        weights=(1, 1, 1),
        demands=demand_rates,
    )
    assert isinstance(g, types.FunctionType)
    assert g([0 for _ in range(67 * 2)]) == 0
    objective_in_days = g(allocation_82) * 1440
    assert np.isclose(
        objective_in_days, 131.23026163719774
    )

def test_objective_function_89():
    """
    Tests the value of the objective function when using an allocation with a resource level of 89
     - When using a simulation, this will achieve a value of 148.0597
     - This objective gives a value of 136.32357249461737
    """
    g = objective.get_objective_function(
        pickup_locations=pickup_locations,
        patient_type_partitions=patient_type_partitions,
        survival_functions=survival_functions,
        primary_travel_times=primary_vehicle_travel_times,
        secondary_travel_times=secondary_vehicle_travel_times,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation_89,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation_89,
        vehicle_locations=vehicle_locations,
        is_station_closer_to_pickup_location=beta,
        is_vehicle_type_closer_to_pickup_location=R,
        weights=(1, 1, 1),
        demands=demand_rates,
    )
    assert isinstance(g, types.FunctionType)
    assert g([0 for _ in range(67 * 2)]) == 0
    objective_in_days = g(allocation_89) * 1440
    assert np.isclose(
        objective_in_days, 136.32357249461737
    )

def test_objective_function_96():
    """
    Tests the value of the objective function when using an allocation with a resource level of 96
     - When using a simulation, this will achieve a value of 147.4496
     - This objective gives a value of 141.0246340566464
    """
    g = objective.get_objective_function(
        pickup_locations=pickup_locations,
        patient_type_partitions=patient_type_partitions,
        survival_functions=survival_functions,
        primary_travel_times=primary_vehicle_travel_times,
        secondary_travel_times=secondary_vehicle_travel_times,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation_96,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation_96,
        vehicle_locations=vehicle_locations,
        is_station_closer_to_pickup_location=beta,
        is_vehicle_type_closer_to_pickup_location=R,
        weights=(1, 1, 1),
        demands=demand_rates,
    )
    assert isinstance(g, types.FunctionType)
    assert g([0 for _ in range(67 * 2)]) == 0
    objective_in_days = g(allocation_96) * 1440
    assert np.isclose(
        objective_in_days, 141.0246340566464
    )