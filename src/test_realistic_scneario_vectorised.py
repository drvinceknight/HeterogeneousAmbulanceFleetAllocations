import types
import numpy as np
import objective_vectorised as objective

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
    assert beta.shape == (261, 67, 67)
    assert R.shape == (261, 67, 67)
    assert survival_functions[0](0) == 1 / (1 + np.exp(0.26))
    assert survival_functions[1](16) == 0.0
    assert survival_functions[2](16) == 1.0
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
     - This objective gives a value of 151.9535873771784
    """
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation_68,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation_68,
        allocation_primary=allocation_68[:67],
        allocation_secondary=allocation_68[67:],
    )
    objective_in_days = g * 1440
    assert np.isclose(objective_in_days, 151.9535873771784)


def test_objective_function_75():
    """
    Tests the value of the objective function when using an allocation with a resource level of 75
     - When using a simulation, this will achieve a value of 145.8549.
     - This objective gives a value of 151.92056329691513
    """
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation_75,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation_75,
        allocation_primary=allocation_75[:67],
        allocation_secondary=allocation_75[67:],
    )
    objective_in_days = g * 1440
    assert np.isclose(objective_in_days, 151.92056329691513)


def test_objective_function_82():
    """
    Tests the value of the objective function when using an allocation with a resource level of 82
     - When using a simulation, this will achieve a value of 145.6902
     - This objective gives a value of 152.0974945594322
    """
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation_82,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation_82,
        allocation_primary=allocation_82[:67],
        allocation_secondary=allocation_82[67:],
    )
    objective_in_days = g * 1440
    assert np.isclose(objective_in_days, 152.0974945594322)


def test_objective_function_89():
    """
    Tests the value of the objective function when using an allocation with a resource level of 89
     - When using a simulation, this will achieve a value of 148.0597
     - This objective gives a value of 151.95283744990988
    """
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation_89,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation_89,
        allocation_primary=allocation_89[:67],
        allocation_secondary=allocation_89[67:],
    )
    objective_in_days = g * 1440
    assert np.isclose(objective_in_days, 151.95283744990988)


def test_objective_function_96():
    """
    Tests the value of the objective function when using an allocation with a resource level of 96
     - When using a simulation, this will achieve a value of 147.4496
     - This objective gives a value of 151.9626105857951
    """
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation_96,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation_96,
        allocation_primary=allocation_96[:67],
        allocation_secondary=allocation_96[67:],
    )
    objective_in_days = g * 1440
    assert np.isclose(objective_in_days, 151.9626105857951)
