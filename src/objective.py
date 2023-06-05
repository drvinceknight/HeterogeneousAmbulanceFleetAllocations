"""
Code to evaluate the survival probability for a given allocation of ambulances.

TODO Add more theoretic description.
"""
from typing import Callable, Iterable
import numpy.typing as npt
import numpy as np


def get_objective_function(
    pickup_locations: Iterable,
    patient_type_partitions: Iterable,
    survival_functions: Iterable[Callable],
    primary_travel_times: npt.NDArray,
    secondary_travel_times: npt.NDArray,
    primary_vehicle_station_utilisation: Iterable,
    secondary_vehicle_station_utilisation: Iterable,
    vehicle_locations: Iterable,
    is_station_closer_to_pickup_location: dict,
    is_vehicle_type_closer_to_pickup_location: dict,
    weights: dict,
    demands: dict,
) -> Callable:
    """
    Take all parameters of the given problem and generate a function of a single
    variable (the allocations).
    """
    number_of_vehicle_locations = len(vehicle_locations)

    def tilde_psi(
        primary_vehicle_allocation,
        secondary_vehicle_allocation,
        patient_type,
        pickup_location,
        ambulance_station,
    ):
        """
        Return tilde_psi
        """
        return get_multiple_vehicle_patient_survival(
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

    def psi(
        primary_vehicle_allocation,
        secondary_vehicle_station_utilisation,
        patient_type,
        pickup_location,
        ambulance_station,
    ):
        """
        Return psi
        """
        return get_single_vehicle_patient_survival(
            patient_type=patient_type,
            pickup_location=pickup_location,
            ambulance_station=ambulance_station,
            survival_functions=survival_functions,
            travel_times=primary_travel_times,
            station_utilisation=primary_vehicle_station_utilisation,
            vehicle_locations=vehicle_locations,
            vehicle_allocation=primary_vehicle_allocation,
            is_station_closer_to_pickup_location=is_station_closer_to_pickup_location,
        )

    def g(allocation: npt.NDArray) -> float:
        f"""
        The objective function for the problem defined by the following
        parameters:

        - pickup_locations: {pickup_locations}
        - patient_type_partitions: {patient_type_partitions}
        - survival_functions: {survival_functions}
        - primary_travel_times: {primary_travel_times}
        - secondary_travel_times: {secondary_travel_times}
        - primary_vehicle_station_utilisation: {primary_vehicle_station_utilisation}
        - secondary_vehicle_station_utilisation: {secondary_vehicle_station_utilisation}
        - vehicle_locations: {vehicle_locations}
        - is_station_closer_to_pickup_location: {is_station_closer_to_pickup_location}
        - is_vehicle_type_closer_to_pickup_location: {is_vehicle_type_closer_to_pickup_location}
        - weights: {weights}
        - demands: {demands}

        It takes `allocation` as a single numpy array. The first half
        corresponds to the location of the primary vehicles and the second half
        to the location of the secondary vehicles.
        """
        primary_vehicle_allocation = allocation[:number_of_vehicle_locations]
        secondary_vehicle_allocation = allocation[number_of_vehicle_locations:]
        return sum(
            get_survival(
                pickup_locations=pickup_locations,
                vehicle_locations=vehicle_locations,
                primary_vehicle_allocation=primary_vehicle_allocation,
                secondary_vehicle_allocation=secondary_vehicle_allocation,
                patient_types=patient_types,
                weights=weights,
                demands=demands,
                psi=psi,
            )
            for patient_types, psi in zip(patient_type_partitions, (psi, tilde_psi))
        )

    return g


def get_survival(
    pickup_locations: Iterable,
    vehicle_locations: Iterable,
    patient_types: Iterable,
    primary_vehicle_allocation: npt.NDArray,
    secondary_vehicle_allocation: npt.NDArray,
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
        * psi(
            primary_vehicle_allocation,
            secondary_vehicle_allocation,
            patient_type,
            pickup_location,
            ambulance_station,
        )
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


def get_multiple_vehicle_patient_survival(
    patient_type: int,
    pickup_location: int,
    ambulance_station: int,
    survival_functions: Iterable[Callable],
    primary_travel_times: npt.NDArray,
    secondary_travel_times: npt.NDArray,
    primary_vehicle_station_utilisation: Iterable,
    secondary_vehicle_station_utilisation: Iterable,
    vehicle_locations: Iterable,
    primary_vehicle_allocation: npt.NDArray,
    secondary_vehicle_allocation: npt.NDArray,
    is_station_closer_to_pickup_location: dict,
    is_vehicle_type_closer_to_pickup_location: dict,
) -> float:
    """
    TODO Write docstring

     s(t) -> the probability of surviving given that the second vehicle went to
     the patient
     (1 - pi...) -> that vehicle was available
     \prod_{a \in \mathcal{A}} -> looking at all other potential vehicles
         first term ->  All other secondary vehicles before vehicle in question
         are busy
         second term -> All other closer primary vehicles are busy
    """
    return (
        survival_functions[patient_type](
            secondary_travel_times[ambulance_station, pickup_location]
        )
        * (
            1
            - secondary_vehicle_station_utilisation[ambulance_station]
            ** secondary_vehicle_allocation[ambulance_station]
        )
        * np.prod(
            [
                secondary_vehicle_station_utilisation[busy_station]
                ** (
                    secondary_vehicle_allocation[busy_station]
                    * is_station_closer_to_pickup_location[
                        (pickup_location, busy_station, ambulance_station)
                    ]
                )
                * primary_vehicle_station_utilisation[busy_station]
                ** (
                    primary_vehicle_allocation[busy_station]
                    * is_vehicle_type_closer_to_pickup_location[
                        (pickup_location, busy_station, ambulance_station)
                    ]
                )
                for busy_station in vehicle_locations
            ]
        )
    )
    +(
        survival_functions[patient_type](
            primary_travel_times[ambulance_station, pickup_location]
        )
        * (
            1
            - primary_vehicle_station_utilisation[ambulance_station]
            ** primary_vehicle_allocation[ambulance_station]
        )
        * np.prod(
            [
                primary_vehicle_station_utilisation[busy_station]
                ** (
                    primary_vehicle_allocation[busy_station]
                    * is_station_closer_to_pickup_location[
                        (pickup_location, busy_station, ambulance_station)
                    ]
                )
                * secondary_vehicle_station_utilisation[busy_station]
                ** (
                    secondary_vehicle_allocation[busy_station]
                    * (
                        1
                        - is_vehicle_type_closer_to_pickup_location[
                            (pickup_location, ambulance_station, busy_station)
                        ]
                    )
                )
                for busy_station in vehicle_locations
            ]
        )
    )
