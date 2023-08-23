from typing import Tuple
import numpy as np
import numpy.typing as npt
import objective
import tqdm  # type: ignore
import dask  # type: ignore


def move_vehicle_of_same_type(
    allocation_for_moving: npt.NDArray[np.int64],
    allocation_not_for_moving: npt.NDArray[np.int64],
    max_allocation: int,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Randomly moves on primary from one primary location to another.
    """
    locations_with_vehicle = np.where(allocation_for_moving)[0]
    locations_to_move_to = np.where(allocation_for_moving < max_allocation)[0]
    from_location = np.random.choice(locations_with_vehicle)
    is_not_from_location = locations_to_move_to != from_location
    to_location = np.random.choice(
        locations_to_move_to, p=is_not_from_location / np.sum(is_not_from_location)
    )
    new_allocation_for_moving = np.array(allocation_for_moving)
    new_allocation_not_for_moving = np.array(allocation_not_for_moving)
    new_allocation_for_moving[from_location] -= 1
    new_allocation_for_moving[to_location] += 1
    return new_allocation_for_moving, new_allocation_not_for_moving


def switch_primary_to_secondary(
    primary_allocation: npt.NDArray[np.int64],
    secondary_allocation: npt.NDArray[np.int64],
    max_allocation: int,
    primary_to_secondary_ratio: int = 3,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Randomly remove a primary vehicle and create `primary_to_secondary_ratio` secondary vehicles.
    """
    locations_with_primary = np.where(primary_allocation)[0]
    locations_to_move_to = np.where(secondary_allocation < max_allocation)[0]
    from_location = np.random.choice(locations_with_primary)
    to_locations = np.random.choice(
        locations_to_move_to, size=primary_to_secondary_ratio
    )
    new_primary_allocation = np.array(primary_allocation)
    new_secondary_allocation = np.array(secondary_allocation)
    new_primary_allocation[from_location] -= 1
    for to_loc in to_locations:
        new_secondary_allocation[to_loc] += 1
    return new_primary_allocation, new_secondary_allocation


def switch_secondary_to_primary(
    primary_allocation,
    secondary_allocation,
    max_allocation,
    primary_to_secondary_ratio=3,
):
    """
    Randomly add a primary vehicle and remove `primary_to_secondary_ratio` secondary vehicles.
    """
    locations_with_secondary = np.repeat(
        np.arange(len(secondary_allocation)), secondary_allocation
    )
    locations_to_move_to = np.where(primary_allocation < max_allocation)[0]
    from_locations = np.random.choice(
        locations_with_secondary,
        size=primary_to_secondary_ratio,
        replace=False,
    )
    to_location = np.random.choice(locations_to_move_to)
    new_primary_allocation = np.array(primary_allocation)
    new_secondary_allocation = np.array(secondary_allocation)
    new_primary_allocation[to_location] += 1
    for from_loc in from_locations:
        new_secondary_allocation[from_loc] -= 1
    return new_primary_allocation, new_secondary_allocation


def mutate_full(
    primary_allocation,
    secondary_allocation,
    max_primary,
    max_secondary,
    primary_to_secondary_ratio=3,
):
    number_primary_vehicles = sum(primary_allocation)
    number_secondary_vehicles = sum(secondary_allocation)
    possible_mutations = [
        lambda x, y, max_primary, max_secondary: move_vehicle_of_same_type(
            allocation_for_moving=x,
            allocation_not_for_moving=y,
            max_allocation=max_primary,
        )
    ]
    if number_secondary_vehicles > 0:
        possible_mutations.append(
            lambda x, y, max_primary, max_secondary: move_vehicle_of_same_type(
                allocation_for_moving=y,
                allocation_not_for_moving=x,
                max_allocation=max_secondary,
            )[::-1]
        )
    if (
        number_primary_vehicles * primary_to_secondary_ratio
        > number_secondary_vehicles + primary_to_secondary_ratio
    ):
        possible_mutations.append(
            lambda x, y, max_primary, max_secondary: switch_primary_to_secondary(
                primary_allocation=x,
                secondary_allocation=y,
                max_allocation=max_secondary,
            )
        )
    if number_secondary_vehicles > primary_to_secondary_ratio:
        possible_mutations.append(
            lambda x, y, max_primary, max_secondary: switch_secondary_to_primary(
                primary_allocation=x, secondary_allocation=y, max_allocation=max_primary
            )
        )
    mutation_function = np.random.choice(possible_mutations)
    return mutation_function(
        primary_allocation, secondary_allocation, max_primary, max_secondary
    )


def mutate_retain_vehicle_numbers(
    primary_allocation,
    secondary_allocation,
    max_primary,
    max_secondary,
):
    number_secondary_vehicles = sum(secondary_allocation)
    possible_mutations = [
        lambda x, y, max_primary, max_secondary: move_vehicle_of_same_type(
            allocation_for_moving=x,
            allocation_not_for_moving=y,
            max_allocation=max_primary,
        )
    ]
    if number_secondary_vehicles > 0:
        possible_mutations.append(
            lambda x, y, max_primary, max_secondary: move_vehicle_of_same_type(
                allocation_for_moving=y,
                allocation_not_for_moving=x,
                max_allocation=max_secondary,
            )[::-1]
        )
    mutation_function = np.random.choice(possible_mutations)
    return mutation_function(
        primary_allocation, secondary_allocation, max_primary, max_secondary
    )


def repeat_mutation(
    mutation_function,
    times_to_repeat,
    primary_allocation,
    secondary_allocation,
    max_primary,
    max_secondary,
):
    """
    Repeats the mutation function a number of times
    """
    for _ in range(times_to_repeat):
        primary_allocation, secondary_allocation = mutation_function(
            primary_allocation=primary_allocation,
            secondary_allocation=secondary_allocation,
            max_primary=max_primary,
            max_secondary=max_secondary,
        )
    return primary_allocation, secondary_allocation


def create_initial_population(
    number_of_locations: int,
    number_of_primary_vehicles: int,
    number_of_secondary_vehicles: int,
    max_primary: int,
    max_secondary: int,
    population_size: int,
    randomise_vehicle_numbers: bool = False,
) -> npt.NDArray[np.int64]:
    """
    Creates a (population_size, 2, number_of_locations) array of population_size allocations.
    Each allocation is a (2, number_of_locations) array consisting of a primary allocation and a secondary allocation.
    """
    population = []
    n_primary = number_of_primary_vehicles
    n_secondary = number_of_secondary_vehicles
    total_number_of_vehicles = int(n_primary + (n_secondary / 3))
    for entry in range(population_size):
        # If randomise_vehicle_numbers, randomise the vehicle numbers
        if randomise_vehicle_numbers:
            n_primary = np.random.choice(
                np.arange(
                    int(np.ceil(total_number_of_vehicles * 0.75)),
                    total_number_of_vehicles + 1,
                )
            )
            n_secondary = (total_number_of_vehicles - n_primary) * 3
        # create primary allocation
        primary_allocation = np.zeros(number_of_locations)
        temp = np.random.choice(
            np.arange(number_of_locations).repeat(max_primary),
            n_primary,
            replace=False,
        )
        locs, numbs = np.unique(temp, return_counts=True)
        primary_allocation[locs] += numbs

        # create secondary allocation
        secondary_allocation = np.zeros(number_of_locations)
        temp = np.random.choice(
            np.arange(number_of_locations).repeat(max_secondary),
            n_secondary,
            replace=False,
        )
        locs, numbs = np.unique(temp, return_counts=True)
        secondary_allocation[locs] += numbs

        # add to population
        population.append([primary_allocation, secondary_allocation])
    return np.array(population).astype(np.int64)


def rank_population(
    population,
    demand_rates,
    primary_survivals,
    secondary_survivals,
    weights_single_vehicle,
    weights_multiple_vehicles,
    beta,
    R,
    vehicle_station_utilisation_function,
    num_workers,
    cache=None,
    **kwargs,
):
    """
    Ranks the population according to the objective function
    """
    tasks = [
        dask.delayed(objective.get_objective)(
            demand_rates=demand_rates,
            primary_survivals=primary_survivals,
            secondary_survivals=secondary_survivals,
            weights_single_vehicle=weights_single_vehicle,
            weights_multiple_vehicles=weights_multiple_vehicles,
            beta=beta,
            R=R,
            vehicle_station_utilisation_function=vehicle_station_utilisation_function,
            allocation_primary=allocation[0],
            allocation_secondary=allocation[1],
            cache=cache,
            **kwargs,
        )
        for allocation in population
    ]
    objective_values = -np.array(dask.compute(*tasks, num_workers=num_workers))
    ordering = np.argsort(objective_values)
    return np.array(population[ordering]), -np.array(objective_values)[ordering]


def optimise(
    number_of_locations,
    number_of_primary_vehicles,
    number_of_secondary_vehicles,
    max_primary,
    max_secondary,
    population_size,
    keep_size,
    number_of_iterations,
    mutation_function,
    initial_number_of_mutatation_repetitions,
    cooling_rate,
    demand_rates,
    primary_survivals,
    secondary_survivals,
    weights_single_vehicle,
    weights_multiple_vehicles,
    beta,
    R,
    vehicle_station_utilisation_function,
    seed,
    num_workers,
    randomise_vehicle_numbers=False,
    progress_bar=False,
    **kwargs,
):
    """
    Optimise
    """
    cache = {}
    np.random.seed(seed)
    objective_by_iteration = []
    population = create_initial_population(
        number_of_locations=number_of_locations,
        number_of_primary_vehicles=number_of_primary_vehicles,
        number_of_secondary_vehicles=number_of_secondary_vehicles,
        max_primary=max_primary,
        max_secondary=max_secondary,
        population_size=population_size,
        randomise_vehicle_numbers=randomise_vehicle_numbers,
    )

    new_pop_size = population_size - keep_size

    steps_to_reach_1 = (initial_number_of_mutatation_repetitions - 1) / cooling_rate
    repetitions = np.int64(
        np.ceil(
            np.interp(
                x=np.arange(number_of_iterations),
                xp=[0, steps_to_reach_1, number_of_iterations],
                fp=[initial_number_of_mutatation_repetitions, 1, 1],
            )
        )
    )

    if progress_bar:
        repetitions = tqdm.tqdm(repetitions)
    for number_of_repetitions in repetitions:
        ranked_population, objective_values = rank_population(
            population=population,
            demand_rates=demand_rates,
            primary_survivals=primary_survivals,
            secondary_survivals=secondary_survivals,
            weights_single_vehicle=weights_single_vehicle,
            weights_multiple_vehicles=weights_multiple_vehicles,
            beta=beta,
            R=R,
            vehicle_station_utilisation_function=vehicle_station_utilisation_function,
            num_workers=num_workers,
            cache=cache,
            **kwargs,
        )
        objective_by_iteration.append(objective_values)
        kept_population = ranked_population[:keep_size]
        new_population = []
        for new_solution in range(new_pop_size):
            (
                primary_allocation_to_mutate,
                secondary_allocation_to_mutate,
            ) = kept_population[np.random.choice(range(keep_size))]
            mutated_solution = repeat_mutation(
                mutation_function=mutation_function,
                times_to_repeat=number_of_repetitions,
                primary_allocation=primary_allocation_to_mutate,
                secondary_allocation=secondary_allocation_to_mutate,
                max_primary=max_primary,
                max_secondary=max_secondary,
            )
            new_population.append(mutated_solution)
        population = np.vstack([kept_population, np.array(new_population)])

    ranked_population, objective_values = rank_population(
        population=population,
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=vehicle_station_utilisation_function,
        num_workers=num_workers,
        cache=cache,
        **kwargs,
    )

    best_primary_population, best_secondary_population = ranked_population[0]

    return (
        best_primary_population,
        best_secondary_population,
        np.array(objective_by_iteration),
    )
