import numpy as np
import types

import objective


def test_get_single_vehicle_patient_survival():
    patient_type = 0
    pickup_location = 1
    ambulance_station = 2
    survival_functions = [lambda x: np.exp(-x)]
    travel_times = np.array(((0, 3.3, 4.2, 3), (1, 0, 1, 2), (2, 3, 0, 2)))
    station_utilisation = np.array((0.2, 0.4, 0.5))
    vehicle_locations = (0, 1, 2)
    vehicle_allocation = (2, 3, 1)
    is_station_closer_to_pickup_location = {
        (0, 0, 0): False,
        (0, 0, 1): False,
        (0, 0, 2): False,
        (0, 0, 3): False,
        (0, 1, 0): True,
        (0, 1, 1): False,
        (0, 1, 2): False,
        (0, 1, 3): False,
        (0, 2, 0): True,
        (0, 2, 1): False,
        (0, 2, 2): True,
        (0, 2, 3): False,
        (1, 0, 0): True,
        (1, 0, 1): True,
        (1, 0, 2): False,
        (1, 0, 3): False,
        (1, 1, 0): False,
        (1, 1, 1): False,
        (1, 1, 2): False,
        (1, 1, 3): False,
        (1, 2, 0): True,
        (1, 2, 1): False,
        (1, 2, 2): True,
        (1, 2, 3): False,
        (2, 0, 0): True,
        (2, 0, 1): True,
        (2, 0, 2): False,
        (2, 0, 3): False,
        (2, 1, 0): False,
        (2, 1, 1): False,
        (2, 1, 2): False,
        (2, 1, 3): False,
        (2, 2, 0): False,
        (2, 2, 1): False,
        (2, 2, 2): False,
        (2, 2, 3): False,
    }
    output = objective.get_single_vehicle_patient_survival(
        patient_type=patient_type,
        pickup_location=pickup_location,
        ambulance_station=ambulance_station,
        survival_functions=survival_functions,
        travel_times=travel_times,
        station_utilisation=station_utilisation,
        vehicle_locations=vehicle_locations,
        vehicle_allocation=vehicle_allocation,
        is_station_closer_to_pickup_location=is_station_closer_to_pickup_location,
    )
    assert np.isclose(output, 0.01244676767)


def test_get_multiple_vehicle_patient_survival():
    patient_type = 0
    pickup_location = 1
    ambulance_station = 2
    survival_functions = [lambda x: np.exp(-x)]
    primary_travel_times = np.array(((0, 3.3, 4.2, 3), (1, 0, 1, 2), (2, 3, 0, 2)))
    secondary_travel_times = np.array(
        ((0.1, 3.2, 4.1, 2), (1, 0, 1, 2), (2, 3.1, 0, 1.9))
    )
    primary_vehicle_station_utilisation = np.array((0.2, 0.4, 0.5))
    secondary_vehicle_station_utilisation = np.array((0.3, 0.2, 0.55))
    vehicle_locations = (0, 1, 2)
    primary_vehicle_allocation = (2, 3, 1)
    secondary_vehicle_allocation = (2, 3, 1)
    is_station_closer_to_pickup_location = {
        (0, 0, 0): False,
        (0, 0, 1): False,
        (0, 0, 2): False,
        (0, 0, 3): False,
        (0, 1, 0): True,
        (0, 1, 1): False,
        (0, 1, 2): False,
        (0, 1, 3): False,
        (0, 2, 0): True,
        (0, 2, 1): False,
        (0, 2, 2): True,
        (0, 2, 3): False,
        (1, 0, 0): True,
        (1, 0, 1): True,
        (1, 0, 2): False,
        (1, 0, 3): False,
        (1, 1, 0): False,
        (1, 1, 1): False,
        (1, 1, 2): False,
        (1, 1, 3): False,
        (1, 2, 0): True,
        (1, 2, 1): False,
        (1, 2, 2): True,
        (1, 2, 3): False,
        (2, 0, 0): True,
        (2, 0, 1): True,
        (2, 0, 2): False,
        (2, 0, 3): False,
        (2, 1, 0): False,
        (2, 1, 1): False,
        (2, 1, 2): False,
        (2, 1, 3): False,
        (2, 2, 0): False,
        (2, 2, 1): False,
        (2, 2, 2): False,
        (2, 2, 3): False,
    }
    is_vehicle_type_closer_to_pickup_location = {
        (0, 0, 0): False,
        (0, 0, 1): False,
        (0, 0, 2): False,
        (0, 0, 3): False,
        (0, 1, 0): True,
        (0, 1, 1): False,
        (0, 1, 2): True,
        (0, 1, 3): False,
        (0, 2, 0): True,
        (0, 2, 1): False,
        (0, 2, 2): True,
        (0, 2, 3): False,
        (1, 0, 0): True,
        (1, 0, 1): True,
        (1, 0, 2): False,
        (1, 0, 3): False,
        (1, 1, 0): False,
        (1, 1, 1): False,
        (1, 1, 2): False,
        (1, 1, 3): False,
        (1, 2, 0): True,
        (1, 2, 1): False,
        (1, 2, 2): True,
        (1, 2, 3): False,
        (2, 0, 0): True,
        (2, 0, 1): True,
        (2, 0, 2): False,
        (2, 0, 3): False,
        (2, 1, 0): True,
        (2, 1, 1): False,
        (2, 1, 2): False,
        (2, 1, 3): False,
        (2, 2, 0): False,
        (2, 2, 1): False,
        (2, 2, 2): False,
        (2, 2, 3): False,
    }
    output = objective.get_multiple_vehicle_patient_survival(
        patient_type=patient_type,
        pickup_location=pickup_location,
        ambulance_station=ambulance_station,
        survival_functions=survival_functions,
        primary_travel_times=primary_travel_times,
        secondary_travel_times=secondary_travel_times,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation,
        vehicle_locations=vehicle_locations,
        primary_vehicle_allocation=primary_vehicle_allocation,
        secondary_vehicle_allocation=secondary_vehicle_allocation,
        is_station_closer_to_pickup_location=is_station_closer_to_pickup_location,
        is_vehicle_type_closer_to_pickup_location=is_vehicle_type_closer_to_pickup_location,
    )
    assert np.isclose(output, 0.00557483879)


def test_get_objective_function():
    pickup_locations = (0, 1, 2, 3)
    patient_type_partitions = ((0,), (1, 2))
    survival_functions = (
            lambda x: np.exp(-x), 
            lambda x: 1 if x < 8 else 0,
            lambda x: 1 if x < 16 else 0,
            )
    primary_travel_times = np.array(((0, 3.3, 4.2, 3), (1, 0, 1, 2), (2, 3, 0, 2)))
    secondary_travel_times = np.array(
        ((0.1, 3.2, 4.1, 2), (1, 0, 1, 2), (2, 3.1, 0, 1.9))
    )
    primary_vehicle_station_utilisation = np.array((0.2, 0.4, 0.5))
    secondary_vehicle_station_utilisation = np.array((0.3, 0.2, 0.55))
    vehicle_locations = (0, 1, 2)
    is_station_closer_to_pickup_location = {
        (0, 0, 0): False,
        (0, 0, 1): False,
        (0, 0, 2): False,
        (0, 0, 3): False,
        (0, 1, 0): True,
        (0, 1, 1): False,
        (0, 1, 2): False,
        (0, 1, 3): False,
        (0, 2, 0): True,
        (0, 2, 1): False,
        (0, 2, 2): True,
        (0, 2, 3): False,
        (1, 0, 0): True,
        (1, 0, 1): True,
        (1, 0, 2): False,
        (1, 0, 3): False,
        (1, 1, 0): False,
        (1, 1, 1): False,
        (1, 1, 2): False,
        (1, 1, 3): False,
        (1, 2, 0): True,
        (1, 2, 1): False,
        (1, 2, 2): True,
        (1, 2, 3): False,
        (2, 0, 0): True,
        (2, 0, 1): True,
        (2, 0, 2): False,
        (2, 0, 3): False,
        (2, 1, 0): False,
        (2, 1, 1): False,
        (2, 1, 2): False,
        (2, 1, 3): False,
        (2, 2, 0): False,
        (2, 2, 1): False,
        (2, 2, 2): False,
        (2, 2, 3): False,
        (3, 0, 0): True,
        (3, 0, 1): True,
        (3, 0, 2): False,
        (3, 0, 3): False,
        (3, 1, 0): False,
        (3, 1, 1): False,
        (3, 1, 2): False,
        (3, 1, 3): False,
        (3, 2, 0): False,
        (3, 2, 1): False,
        (3, 2, 2): False,
        (3, 2, 3): False,
    }
    is_vehicle_type_closer_to_pickup_location = {
        (0, 0, 0): False,
        (0, 0, 1): False,
        (0, 0, 2): False,
        (0, 0, 3): False,
        (0, 1, 0): True,
        (0, 1, 1): False,
        (0, 1, 2): True,
        (0, 1, 3): False,
        (0, 2, 0): True,
        (0, 2, 1): False,
        (0, 2, 2): True,
        (0, 2, 3): False,
        (1, 0, 0): True,
        (1, 0, 1): True,
        (1, 0, 2): False,
        (1, 0, 3): False,
        (1, 1, 0): False,
        (1, 1, 1): False,
        (1, 1, 2): False,
        (1, 1, 3): False,
        (1, 2, 0): True,
        (1, 2, 1): False,
        (1, 2, 2): True,
        (1, 2, 3): False,
        (2, 0, 0): True,
        (2, 0, 1): True,
        (2, 0, 2): False,
        (2, 0, 3): False,
        (2, 1, 0): True,
        (2, 1, 1): False,
        (2, 1, 2): False,
        (2, 1, 3): False,
        (2, 2, 0): False,
        (2, 2, 1): False,
        (2, 2, 2): False,
        (2, 2, 3): False,
        (3, 0, 0): True,
        (3, 0, 1): True,
        (3, 0, 2): False,
        (3, 0, 3): False,
        (3, 1, 0): False,
        (3, 1, 1): False,
        (3, 1, 2): False,
        (3, 1, 3): False,
        (3, 2, 0): False,
        (3, 2, 1): False,
        (3, 2, 2): False,
        (3, 2, 3): False,
    }
    weights = (.5, .3, .2)
    demands = np.array(
            (
                ))

    demands = np.array(((2, 2, 3, 3), (2, 0, 1, 2), (1, 1, 1, 1)))
    g = objective.get_objective_function(
        pickup_locations=pickup_locations,
        patient_type_partitions=patient_type_partitions,
        survival_functions=survival_functions,
        primary_travel_times=primary_travel_times,
        secondary_travel_times=secondary_travel_times,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation,
        vehicle_locations=vehicle_locations,
        is_station_closer_to_pickup_location=is_station_closer_to_pickup_location,
        is_vehicle_type_closer_to_pickup_location=is_vehicle_type_closer_to_pickup_location,
        weights=weights,
        demands=demands,
    )
    assert isinstance(g, types.FunctionType)
    allocations = np.array((0, 0, 0, 0, 0, 0))
    assert g(allocations) == 0.0
    allocations = np.array((1, 0, 2, 0, 3, 0))
    assert np.isclose(g(allocations), 2.620185796108)
    allocations = np.array((1, 50, 2, 20, 3, 5))
    assert np.isclose(g(allocations), 4.07839246016)
