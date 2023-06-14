import genetic
import numpy as np

def test_move_vehicle_of_same_type():
    primary_allocation = np.array([0, 1, 5, 1])
    secondary_allocation = np.array([3, 9, 0, 0])
    max_allocation = 5
    
    np.random.seed(1)
    resulting_primary_allocation, resulting_secondary_allocation = genetic.move_vehicle_of_same_type(
        allocation_for_moving=primary_allocation,
        allocation_not_for_moving=secondary_allocation,
        max_allocation=max_allocation
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
    resulting_primary_allocation, resulting_secondary_allocation = genetic.switch_primary_to_secondary(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_allocation=max_allocation
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
    resulting_primary_allocation, resulting_secondary_allocation = genetic.switch_secondary_to_primary(
        primary_allocation=primary_allocation,
        secondary_allocation=secondary_allocation,
        max_allocation=max_allocation
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
        max_allocation=max_allocation
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(resulting_primary_allocation) + (sum(resulting_secondary_allocation) / 3)
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
        max_allocation=max_allocation
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(resulting_primary_allocation) + (sum(resulting_secondary_allocation) / 3)
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
        max_allocation=max_allocation
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(resulting_primary_allocation) + (sum(resulting_secondary_allocation) / 3)
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
        max_allocation=max_allocation
    )

    assert sum(primary_allocation) + (sum(secondary_allocation) / 3) == sum(resulting_primary_allocation) + (sum(resulting_secondary_allocation) / 3)
    assert np.array_equal(resulting_secondary_allocation, np.array([2, 7, 0, 0]))
    assert np.array_equal(resulting_primary_allocation, np.array([0, 1, 5, 2]))
