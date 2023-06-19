import genetic
import objective_vectorised as objective
import numpy as np
import random


def test_move_vehicle_of_same_type():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(1)
    (
        resulting_primary_allocation,
        resulting_secondary_allocation,
    ) = genetic.move_vehicle_of_same_type(
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
    ) = genetic.switch_primary_to_secondary(
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
    ) = genetic.switch_secondary_to_primary(
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
    resulting_primary_allocation, resulting_secondary_allocation = genetic.mutate(
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
    resulting_primary_allocation, resulting_secondary_allocation = genetic.mutate(
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
    resulting_primary_allocation, resulting_secondary_allocation = genetic.mutate(
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
    resulting_primary_allocation, resulting_secondary_allocation = genetic.mutate(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_allocation=max_allocation,
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(
        resulting_primary_allocation
    ) + (sum(resulting_secondary_allocation) / 3)
    assert np.array_equal(resulting_secondary_allocation, np.array([2, 7, 0, 0]))
    assert np.array_equal(resulting_primary_allocation, np.array([0, 1, 5, 2]))


def test_create_initial_population():
    number_of_locations = 6
    population_size = 15
    number_of_primary_vehicles = 8
    number_of_secondary_vehicles = 12
    max_primary_allocation = 3
    max_secondary_allocation = 4

    np.random.seed(0)
    population = genetic.create_initial_population(
        number_of_locations=number_of_locations,
        number_of_primary_vehicles=number_of_primary_vehicles,
        number_of_secondary_vehicles=number_of_secondary_vehicles,
        max_primary_allocation=max_primary_allocation,
        max_secondary_allocation=max_secondary_allocation,
        population_size=population_size
    )

    assert population.shape == (population_size, 2, number_of_locations)
    for entry in range(population_size):
        assert population[entry][0].sum() == number_of_primary_vehicles
        assert population[entry][1].sum() == number_of_secondary_vehicles
        assert population[entry][0].max() <= max_primary_allocation
        assert population[entry][1].max() <= max_secondary_allocation

    assert np.allclose(population[0][0], np.array([2, 1, 2, 1, 1, 1]))
    assert np.allclose(population[0][1], np.array([1, 0, 2, 2, 3, 4]))


def test_rank_population():
    # Read in data
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
    primary_vehicle_station_utilisation_61 = np.genfromtxt(
        "./test_data/primary_utilisations_61.csv", delimiter=","
    )  # Directly from simulation
    secondary_vehicle_station_utilisation_61 = np.genfromtxt(
        "./test_data/secondary_utilisations_61.csv", delimiter=","
    )  # Directly from simulation
    allocation_61 = np.genfromtxt("./test_data/allocation_61.csv", delimiter=",")

    # Create population
    random.seed(0)
    population = np.array([
        [
            random.sample(list(allocation_61[:67]), 67), random.sample(list(allocation_61[67:]), 67)
        ] for entry in range(10)
    ])
    assert population.shape == (10, 2, 67)

    ranked_population = genetic.rank_population(
        population=population,
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        primary_vehicle_station_utilisation=primary_vehicle_station_utilisation_61,
        secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation_61
    )

    assert ranked_population.shape == (10, 2, 67)
    previous_objective_value = float('inf')
    for allocation in ranked_population:
        next_objective_value = objective.get_objective(
            demand_rates=demand_rates,
            primary_survivals=primary_survivals,
            secondary_survivals=secondary_survivals,
            weights_single_vehicle=weights_single_vehicle,
            weights_multiple_vehicles=weights_multiple_vehicles,
            beta=beta,
            R=R,
            primary_vehicle_station_utilisation=primary_vehicle_station_utilisation_61,
            secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation_61,
            allocation_primary=allocation[0],
            allocation_secondary=allocation[1]
        )
        assert previous_objective_value >= next_objective_value
        previous_objective_value = next_objective_value
