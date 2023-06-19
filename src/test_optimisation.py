import optimisation
import numpy as np


def test_move_vehicle_of_same_type():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(1)
    (
        resulting_primary_allocation,
        resulting_secondary_allocation,
    ) = optimisation.move_vehicle_of_same_type(
        allocation_for_moving=primary_allocation,
        allocation_not_for_moving=secondary_allocation,
        max_allocation=max_allocation,
    )

    assert sum(primary_allocation) == sum(resulting_primary_allocation)
    assert sum(secondary_allocation) == sum(resulting_secondary_allocation)
    assert np.array_equal(secondary_allocation, resulting_secondary_allocation)
    assert np.array_equal(resulting_primary_allocation, np.array([0, 1, 4, 2]))


def test_switch_primary_to_secondary():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(1)
    (
        resulting_primary_allocation,
        resulting_secondary_allocation,
    ) = optimisation.switch_primary_to_secondary(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_allocation=max_allocation,
    )

    assert sum(primary_allocation) - 1 == sum(resulting_primary_allocation)
    assert sum(secondary_allocation) + 3 == sum(resulting_secondary_allocation)
    assert np.array_equal(resulting_secondary_allocation, np.array([5, 9, 1, 0]))
    assert np.array_equal(resulting_primary_allocation, np.array([0, 1, 4, 1]))


def test_switch_secondary_to_primary():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(1)
    (
        resulting_primary_allocation,
        resulting_secondary_allocation,
    ) = optimisation.switch_secondary_to_primary(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_allocation=max_allocation,
    )

    assert sum(primary_allocation) + 1 == sum(resulting_primary_allocation)
    assert sum(secondary_allocation) - 3 == sum(resulting_secondary_allocation)
    assert np.array_equal(resulting_secondary_allocation, np.array([2, 7, 0, 0]))
    assert np.array_equal(resulting_primary_allocation, np.array([1, 1, 5, 1]))


def test_mutate_with_seed_0():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(0)
    resulting_primary_allocation, resulting_secondary_allocation = optimisation.mutate(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_allocation=max_allocation,
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(
        resulting_primary_allocation
    ) + (sum(resulting_secondary_allocation) / 3)
    assert np.array_equal(resulting_secondary_allocation, np.array([3, 9, 0, 0]))
    assert np.array_equal(resulting_primary_allocation, np.array([0, 1, 4, 2]))


def test_mutate_with_seed_1():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(1)
    resulting_primary_allocation, resulting_secondary_allocation = optimisation.mutate(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_allocation=max_allocation,
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(
        resulting_primary_allocation
    ) + (sum(resulting_secondary_allocation) / 3)
    assert np.array_equal(resulting_secondary_allocation, np.array([3, 8, 0, 1]))
    assert np.array_equal(resulting_primary_allocation, np.array([0, 1, 5, 1]))


def test_mutate_with_seed_3():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(3)
    resulting_primary_allocation, resulting_secondary_allocation = optimisation.mutate(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_allocation=max_allocation,
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(
        resulting_primary_allocation
    ) + (sum(resulting_secondary_allocation) / 3)
    assert np.array_equal(resulting_secondary_allocation, np.array([5, 9, 1, 0]))
    assert np.array_equal(resulting_primary_allocation, np.array([0, 0, 5, 1]))


def test_mutate_with_seed_5():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(5)
    resulting_primary_allocation, resulting_secondary_allocation = optimisation.mutate(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_allocation=max_allocation,
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(
        resulting_primary_allocation
    ) + (sum(resulting_secondary_allocation) / 3)
    assert np.array_equal(resulting_secondary_allocation, np.array([2, 7, 0, 0]))
    assert np.array_equal(resulting_primary_allocation, np.array([0, 1, 5, 2]))


def test_optimisation_seed_5():
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
    weights = (0.5, 0.3, 0.2)
    demands = np.array(())

    demands = np.array(((2, 2, 3, 3), (2, 0, 1, 2), (1, 1, 1, 1)))
    initial_allocation = np.array((1, 0, 2, 0, 3, 0))
    population_size = 10
    iterations = 50

    generations = optimisation.run(
        initial_allocation=initial_allocation,
        population_size=population_size,
        pickup_locations=pickup_locations,
        patient_type_partitions=patient_type_partitions,
        survival_functions=survival_functions,
        primary_travel_times=primary_travel_times,
        secondary_travel_times=secondary_travel_times,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation,
        vehicle_locations=vehicle_locations,
        weights=weights,
        demands=demands,
    )
    assert len(generations) == iterations
    assert all(len(generation) == population_size for generation in generations)
