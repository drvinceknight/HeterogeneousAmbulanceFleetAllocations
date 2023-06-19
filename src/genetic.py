import numpy as np
import objective_vectorised as objective


def move_vehicle_of_same_type(
    allocation_for_moving, allocation_not_for_moving, max_allocation
):
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
    primary_allocation,
    secondary_allocation,
    max_allocation,
    primary_to_secondary_ratio=3,
):
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
    locations_with_secondary = np.where(secondary_allocation)[0]
    locations_to_move_to = np.where(primary_allocation < max_allocation)[0]
    from_locations = np.random.choice(
        locations_with_secondary, size=primary_to_secondary_ratio
    )
    to_location = np.random.choice(locations_to_move_to)
    new_primary_allocation = np.array(primary_allocation)
    new_secondary_allocation = np.array(secondary_allocation)
    new_primary_allocation[to_location] += 1
    for from_loc in from_locations:
        new_secondary_allocation[from_loc] -= 1
    return new_primary_allocation, new_secondary_allocation


def mutate(
    primary_allocation,
    secondary_allocation,
    max_allocation,
    primary_to_secondary_ratio=3,
):
    number_primary_vehicles = sum(primary_allocation)
    number_secondary_vehicles = sum(secondary_allocation)
    possible_mutations = [
        lambda x, y, max_allocation: move_vehicle_of_same_type(x, y, max_allocation)
    ]
    if number_secondary_vehicles > 0:
        possible_mutations.append(
            lambda x, y, max_allocation: move_vehicle_of_same_type(
                y, x, max_allocation
            )[::-1]
        )
    if (
        number_primary_vehicles * primary_to_secondary_ratio
        > number_secondary_vehicles + primary_to_secondary_ratio
    ):
        possible_mutations.append(
            lambda x, y, max_allocation: switch_primary_to_secondary(
                x, y, max_allocation
            )
        )
    if number_secondary_vehicles > primary_to_secondary_ratio:
        possible_mutations.append(
            lambda x, y, max_allocation: switch_secondary_to_primary(
                x, y, max_allocation
            )
        )

    mutation_function = np.random.choice(possible_mutations)
    print(mutation_function)
    return mutation_function(primary_allocation, secondary_allocation, max_allocation)


def create_initial_population(
    number_of_locations,
    number_of_primary_vehicles,
    number_of_secondary_vehicles,
    max_primary_allocation,
    max_secondary_allocation,
    population_size
):
    """
    Creates a (population_size, 2, number_of_locations) array of population_size allocations.
    Each allocation is a (2, number_of_locations) array consisting of a primary allocation and a secondary allocation.
    """
    population = []
    for entry in range(population_size):
        # create primary allocation
        primary_allocation = np.zeros(number_of_locations)
        temp = np.random.choice(np.arange(number_of_locations).repeat(max_primary_allocation), number_of_primary_vehicles, replace=False)
        locs, numbs = np.unique(temp, return_counts=True)
        primary_allocation[locs] += numbs
        
        # create secondary allocation
        secondary_allocation = np.zeros(number_of_locations)
        temp = np.random.choice(np.arange(number_of_locations).repeat(max_secondary_allocation), number_of_secondary_vehicles, replace=False)
        locs, numbs = np.unique(temp, return_counts=True)
        secondary_allocation[locs] += numbs

        # add to population
        population.append([primary_allocation, secondary_allocation])
    return np.array(population)


def rank_population(
    population,
    demand_rates,
    primary_survivals,
    secondary_survivals,
    weights_single_vehicle,
    weights_multiple_vehicles,
    beta,
    R,
    primary_vehicle_station_utilisation,
    secondary_vehicle_station_utilisation
):
    """
    Ranks the population according to the objective function
    """
    objective_values = []
    for allocation in population:
        objective_values.append(
            -objective.get_objective(
                demand_rates=demand_rates,
                primary_survivals=primary_survivals,
                secondary_survivals=secondary_survivals,
                weights_single_vehicle=weights_single_vehicle,
                weights_multiple_vehicles=weights_multiple_vehicles,
                beta=beta,
                R=R,
                primary_vehicle_station_utilisation=primary_vehicle_station_utilisation,
                secondary_vehicle_station_utilisation=secondary_vehicle_station_utilisation,
                allocation_primary=allocation[0],
                allocation_secondary=allocation[1]
            )
        )
    ordering = np.argsort(objective_values)
    return np.array(population[ordering])
