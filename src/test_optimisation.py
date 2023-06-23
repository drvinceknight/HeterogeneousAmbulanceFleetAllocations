import objective
import optimisation
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


def test_mutate_retain_vehicle_numbers_with_seed_0():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(0)
    (
        resulting_primary_allocation,
        resulting_secondary_allocation,
    ) = optimisation.mutate_retain_vehicle_numbers(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_primary=max_allocation,
        max_secondary=max_allocation,
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(
        resulting_primary_allocation
    ) + (sum(resulting_secondary_allocation) / 3)
    assert np.array_equal(resulting_secondary_allocation, np.array([3, 9, 0, 0]))
    assert np.array_equal(resulting_primary_allocation, np.array([0, 1, 4, 2]))


def test_mutate_retain_vehicle_numbers_with_seed_1():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(1)
    (
        resulting_primary_allocation,
        resulting_secondary_allocation,
    ) = optimisation.mutate_retain_vehicle_numbers(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_primary=max_allocation,
        max_secondary=max_allocation,
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(
        resulting_primary_allocation
    ) + (sum(resulting_secondary_allocation) / 3)
    assert np.array_equal(resulting_secondary_allocation, np.array([3, 8, 0, 1]))
    assert np.array_equal(resulting_primary_allocation, np.array([0, 1, 5, 1]))


def test_mutate_full_with_seed_0():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(0)
    (
        resulting_primary_allocation,
        resulting_secondary_allocation,
    ) = optimisation.mutate_full(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_primary=max_allocation,
        max_secondary=max_allocation,
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(
        resulting_primary_allocation
    ) + (sum(resulting_secondary_allocation) / 3)
    assert np.array_equal(resulting_secondary_allocation, np.array([3, 9, 0, 0]))
    assert np.array_equal(resulting_primary_allocation, np.array([0, 1, 4, 2]))


def test_mutate_full_with_seed_1():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(1)
    (
        resulting_primary_allocation,
        resulting_secondary_allocation,
    ) = optimisation.mutate_full(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_primary=max_allocation,
        max_secondary=max_allocation,
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(
        resulting_primary_allocation
    ) + (sum(resulting_secondary_allocation) / 3)
    assert np.array_equal(resulting_secondary_allocation, np.array([3, 8, 0, 1]))
    assert np.array_equal(resulting_primary_allocation, np.array([0, 1, 5, 1]))


def test_mutate_full_with_seed_3():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(3)
    (
        resulting_primary_allocation,
        resulting_secondary_allocation,
    ) = optimisation.mutate_full(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_primary=max_allocation,
        max_secondary=max_allocation,
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(
        resulting_primary_allocation
    ) + (sum(resulting_secondary_allocation) / 3)
    assert np.array_equal(resulting_secondary_allocation, np.array([5, 9, 1, 0]))
    assert np.array_equal(resulting_primary_allocation, np.array([0, 0, 5, 1]))


def test_mutate_full_with_seed_5():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5

    np.random.seed(5)
    (
        resulting_primary_allocation,
        resulting_secondary_allocation,
    ) = optimisation.mutate_full(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_primary=max_allocation,
        max_secondary=max_allocation,
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
    max_primary = 3
    max_secondary = 4

    np.random.seed(0)
    population = optimisation.create_initial_population(
        number_of_locations=number_of_locations,
        number_of_primary_vehicles=number_of_primary_vehicles,
        number_of_secondary_vehicles=number_of_secondary_vehicles,
        max_primary=max_primary,
        max_secondary=max_secondary,
        population_size=population_size,
    )

    assert population.shape == (population_size, 2, number_of_locations)
    for entry in range(population_size):
        assert population[entry][0].sum() == number_of_primary_vehicles
        assert population[entry][1].sum() == number_of_secondary_vehicles
        assert population[entry][0].max() <= max_primary
        assert population[entry][1].max() <= max_secondary

    assert np.allclose(population[0][0], np.array([2, 1, 2, 1, 1, 1]))
    assert np.allclose(population[0][1], np.array([1, 0, 2, 2, 3, 4]))


def test_rank_population():
    # Read in data
    raw_travel_times = np.genfromtxt(
        "./test_data/travel_times_matrix.csv", delimiter=","
    )
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
    def primary_vehicle_station_utilisation_function_61(**kwargs):
        return np.genfromtxt(
        "./test_data/primary_utilisations_61.csv", delimiter=","
    )  # Directly from simulation
    def secondary_vehicle_station_utilisation_function_61(**kwargs):
        return np.genfromtxt(
        "./test_data/secondary_utilisations_61.csv", delimiter=","
    )  # Directly from simulation
    allocation_61 = np.genfromtxt("./test_data/allocation_61.csv", delimiter=",")

    # Create population
    random.seed(0)
    population = np.array(
        [
            [
                random.sample(list(allocation_61[:67]), 67),
                random.sample(list(allocation_61[67:]), 67),
            ]
            for entry in range(10)
        ]
    )
    assert population.shape == (10, 2, 67)

    ranked_population, objective_values = optimisation.rank_population(
        population=population,
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        primary_vehicle_station_utilisation_function=primary_vehicle_station_utilisation_function_61,
        secondary_vehicle_station_utilisation_function=secondary_vehicle_station_utilisation_function_61
    )

    assert ranked_population.shape == (10, 2, 67)
    assert np.all(objective_values[:-1] >= objective_values[1:])
    previous_objective_value = float("inf")
    for allocation in ranked_population:
        next_objective_value = objective.get_objective(
            demand_rates=demand_rates,
            primary_survivals=primary_survivals,
            secondary_survivals=secondary_survivals,
            weights_single_vehicle=weights_single_vehicle,
            weights_multiple_vehicles=weights_multiple_vehicles,
            beta=beta,
            R=R,
            primary_vehicle_station_utilisation_function=primary_vehicle_station_utilisation_function_61,
            secondary_vehicle_station_utilisation_function=secondary_vehicle_station_utilisation_function_61,
            allocation_primary=allocation[0],
            allocation_secondary=allocation[1],
        )
        assert previous_objective_value >= next_objective_value
        previous_objective_value = next_objective_value


def test_optimise():
    # Read in data
    raw_travel_times = np.genfromtxt(
        "./test_data/travel_times_matrix.csv", delimiter=","
    )
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
    pop_size = 20
    num_iters = 30
    max_alloc = 4
    num_vehicles = 20

    def primary_vehicle_station_utilisation_function(**kwargs):
        return np.array([0.7 for _ in range(67)])


    def secondary_vehicle_station_utilisation_function(**kwargs):
        return np.array([0.4 for _ in range(67)])

    best_primary, best_secondary, objective_by_iteration = optimisation.optimise(
        number_of_locations=67,
        number_of_primary_vehicles=num_vehicles,
        number_of_secondary_vehicles=num_vehicles,
        max_primary=max_alloc,
        max_secondary=max_alloc,
        population_size=pop_size,
        keep_size=5,
        number_of_iterations=num_iters,
        mutation_function=optimisation.mutate_retain_vehicle_numbers,
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        primary_vehicle_station_utilisation_function=primary_vehicle_station_utilisation_function,
        secondary_vehicle_station_utilisation_function=secondary_vehicle_station_utilisation_function,
        seed=0,
        progress_bar=False,
    )
    best_over_time = objective_by_iteration.max(axis=1)

    assert len(best_primary) == 67
    assert len(best_secondary) == 67
    assert max(best_primary) <= max_alloc
    assert max(best_secondary) <= max_alloc
    assert sum(best_primary) == num_vehicles
    assert sum(best_secondary) == num_vehicles
    assert objective_by_iteration.shape == (num_iters, pop_size)
    assert np.all(best_over_time[:-1] <= best_over_time[1:])
