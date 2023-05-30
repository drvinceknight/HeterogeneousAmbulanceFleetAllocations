"""
Code to evaluate the survival probability for a given allocation of ambulances.

TODO Add more theoretic description.
"""
from typing import Callable, Iterable
import numpy.typing as npt
import numpy as np


def get_survival(
    pickup_locations: Iterable,
    vehicle_locations: Iterable,
    patient_types: Iterable,
    weights: dict,
    demands: dict,
    psi: Callable,
) -> float:
    """
    TODO Write docstring
    """
    return sum(
        weights[patient_type]
        * demands[(patient_type, pickup_location)]
        * psi(patient_type, pickup_location, ambulance_station)
        for patient_type in patient_types
        for ambulance_station in vehicle_locations
        for pickup_location in pickup_locations
    )


def get_single_vehicle_patient_survival(
    patient_type: int,
    pickup_location: int,
    ambulance_station: int,
    survival_functions: Iterable[Callable],
    travel_times: npt.NDArray,
    station_utilisation: Iterable,
    vehicle_locations: Iterable,
    vehicle_allocation: npt.NDArray,
    is_station_closer_to_pickup_location: dict,
) -> float:
    """
    TODO Write docstring
    """
    return (
        survival_functions[patient_type](
            travel_times[ambulance_station, pickup_location]
        )
        * (
            1
            - station_utilisation[ambulance_station]
            ** vehicle_allocation[ambulance_station]
        )
        * np.prod(
            [
                station_utilisation[busy_station]
                ** (
                    vehicle_allocation[busy_station]
                    * is_station_closer_to_pickup_location[
                        (pickup_location, busy_station, ambulance_station)
                    ]
                )
                for busy_station in vehicle_locations
            ]
        )
    )
