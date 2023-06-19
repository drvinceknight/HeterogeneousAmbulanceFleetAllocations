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


def mutate(
    primary_allocation,
    secondary_allocation,
    max_primary_allocation,
    max_secondary_allocation
):
    number_primary_vehicles = sum(primary_allocation)
    number_secondary_vehicles = sum(secondary_allocation)
    possible_mutations = [
        lambda x, y: move_vehicle_of_same_type(
            allocation_for_moving=x,
            allocation_not_for_moving=y,
            max_allocation=max_primary_allocation
        )
    ]
    if number_secondary_vehicles > 0:
        possible_mutations.append(
            lambda x, y: move_vehicle_of_same_type(
                allocation_for_moving=y,
                allocation_not_for_moving=x,
                max_allocation=max_secondary_allocation
            )[::-1]
        )
    mutation_function = np.random.choice(possible_mutations)
    return mutation_function(primary_allocation, secondary_allocation)


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
