import numpy as np

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
