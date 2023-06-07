import types

import numpy as np

import objective

raw_travel_times = np.genfromtxt("./travel_times_matrix.csv", delimiter=",")
beta = objective.get_beta(travel_times=raw_travel_times)
primary_vehicle_travel_times = raw_travel_times / 0.75
secondary_vehicle_travel_times = raw_travel_times / 125
R = objective.get_R(primary_vehicle_travel_times=primary_vehicle_travel_times,
        secondary_vehicle_travel_times=secondary_vehicle_travel_times)
survival_functions = (
        lambda t:  1 / (1 + np.exp(0.26 + 0.139 * t)),
        lambda t: t <= 15,
        lambda t: t <= 60,
        )
demand_rates = np.genfromtxt("./demand.csv", delimiter=",")
vehicle_locations, pickup_locations = tuple(map(range, raw_travel_times.shape))
patient_type_partitions = ((0,), (1, 2))
primary_vehicle_station_utilisation = [.5 for _ in raw_travel_times]
secondary_vehicle_station_utilisation = [.5 for _ in raw_travel_times]
allocation = np.genfromtxt("./allocation_75.csv", delimiter=",")

def test_paramaters():
    assert raw_travel_times.shape == (67, 261)
    assert type(beta) is dict
    assert len(beta) == 67 ** 2 * 261
    assert type(R) is dict
    assert len(R) == 67 ** 2 * 261
    assert survival_functions[0](0) == 1 / (1 + np.exp(0.26))
    assert survival_functions[1](16) is False
    assert survival_functions[2](16) is True
    assert demand_rates.shape == (3, 261)
    assert len(primary_vehicle_station_utilisation) == 67
    assert len(secondary_vehicle_station_utilisation) == 67
    assert sum(allocation[:67]) + 1 / 3 * sum(allocation[-67:]) == 75


def test_objective_function():
    g = objective.get_objective_function(
        pickup_locations=pickup_locations,
        patient_type_partitions=patient_type_partitions,
        survival_functions=survival_functions,
        primary_travel_times=primary_vehicle_travel_times,
        secondary_travel_times=secondary_vehicle_travel_times,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation,
        vehicle_locations=vehicle_locations,
        is_station_closer_to_pickup_location=beta,
        is_vehicle_type_closer_to_pickup_location=R,
        weights=(1, 1, 1),
        demands=demand_rates,
    )
    assert isinstance(g, types.FunctionType)
    assert g([0 for _ in range(67 * 2)]) == 0
    assert np.isclose(g(allocation), 131.21336710313716)
