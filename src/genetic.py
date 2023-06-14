import numpy as np

def move_vehicle_of_same_type(allocation_for_moving, allocation_not_for_moving, max_allocation):
    """
    Randomly moves on primary from one primary location to another.
    """
    locations_with_vehicle = np.where(allocation_for_moving)[0]
    locations_to_move_to = np.where(allocation_for_moving < max_allocation)[0]
    from_location = np.random.choice(locations_with_vehicle)
    is_not_from_location = locations_to_move_to != from_location
    to_location = np.random.choice(locations_to_move_to, p=is_not_from_location/np.sum(is_not_from_location))
    new_allocation_for_moving = np.array(allocation_for_moving)
    new_allocation_not_for_moving = np.array(allocation_not_for_moving)
    new_allocation_for_moving[from_location] -= 1
    new_allocation_for_moving[to_location] += 1
    return new_allocation_for_moving, new_allocation_not_for_moving

def switch_primary_to_secondary(primary_allocation, secondary_allocation, max_allocation, primary_to_secondary_ratio=3):
    """
    Randomly remove a primary vehicle and create `primary_to_secondary_ratio` secondary vehicles.
    """
    locations_with_primary = np.where(primary_allocation)[0]
    locations_to_move_to = np.where(secondary_allocation < max_allocation)[0]
    from_location = np.random.choice(locations_with_primary)
    to_locations = np.random.choice(locations_to_move_to, size=primary_to_secondary_ratio)
    new_primary_allocation = np.array(primary_allocation)
    new_secondary_allocation = np.array(secondary_allocation)
    new_primary_allocation[from_location] -= 1
    for to_loc in to_locations:
        new_secondary_allocation[to_loc] += 1
    return new_primary_allocation, new_secondary_allocation

def switch_secondary_to_primary(primary_allocation, secondary_allocation, max_allocation, primary_to_secondary_ratio=3):
    """
    Randomly add a primary vehicle and remove `primary_to_secondary_ratio` secondary vehicles.
    """
    locations_with_secondary = np.where(secondary_allocation)[0]
    locations_to_move_to = np.where(primary_allocation < max_allocation)[0]
    from_locations = np.random.choice(locations_with_secondary, size=primary_to_secondary_ratio)
    to_location = np.random.choice(locations_to_move_to)
    new_primary_allocation = np.array(primary_allocation)
    new_secondary_allocation = np.array(secondary_allocation)
    new_primary_allocation[to_location] += 1
    for from_loc in from_locations:
        new_secondary_allocation[from_loc] -= 1
    return new_primary_allocation, new_secondary_allocation


def mutate(primary_allocation, secondary_allocation, max_allocation, primary_to_secondary_ratio=3):
    number_primary_vehicles = sum(primary_allocation)
    number_secondary_vehicles = sum(secondary_allocation)
    possible_mutations = [lambda x, y, max_allocation: move_vehicle_of_same_type(x, y, max_allocation)]
    if number_secondary_vehicles > 0:
        possible_mutations.append(lambda x, y, max_allocation: move_vehicle_of_same_type(y, x, max_allocation)[::-1])
    if number_primary_vehicles * primary_to_secondary_ratio > number_secondary_vehicles + primary_to_secondary_ratio:
        possible_mutations.append(lambda x, y, max_allocation: switch_primary_to_secondary(x, y, max_allocation))
    if number_secondary_vehicles > primary_to_secondary_ratio:
        possible_mutations.append(lambda x, y, max_allocation: switch_secondary_to_primary(x, y, max_allocation))

    mutation_function = np.random.choice(possible_mutations)
    print(mutation_function)
    return mutation_function(primary_allocation, secondary_allocation, max_allocation)

