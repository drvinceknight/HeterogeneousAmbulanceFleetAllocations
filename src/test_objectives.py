import numpy as np
import types
import objective
import utilisation


def test_get_beta():
    travel_times = np.array(
        [[0, 5, 10, 15, 20], [5, 0, 5, 10, 15], [10, 5, 0, 5, 10], [15, 10, 5, 0, 5]]
    )
    beta = objective.get_beta(travel_times)
    expected_beta = np.array(
        [
            [
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
        ]
    )
    assert np.allclose(beta, expected_beta)


def test_get_R():
    primary_travel_times = np.array(
        [[0, 5, 10, 15, 20], [5, 0, 5, 10, 15], [10, 5, 0, 5, 10], [15, 10, 5, 0, 5]]
    )
    secondary_travel_times = 0.7 * primary_travel_times
    R = objective.get_R(primary_travel_times, secondary_travel_times)
    expected_R = np.array(
        [
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
        ]
    )
    assert np.allclose(R, expected_R)


def test_relationship_between_beta_and_R():
    travel_times = np.array(
        [[0, 5, 10, 15, 20], [5, 0, 5, 10, 15], [10, 5, 0, 5, 10], [15, 10, 5, 0, 5]]
    )
    R = objective.get_R(travel_times, travel_times)
    beta = objective.get_beta(travel_times)
    expected_difference_is_identity = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )
    np.allclose(R - beta, expected_difference_is_identity)


def test_get_survival_vectors():
    primary_travel_times = np.array(
        [[0, 5, 10, 15, 20], [5, 0, 5, 10, 15], [10, 5, 0, 5, 10], [15, 10, 5, 0, 5]]
    )
    secondary_travel_times = primary_travel_times * 0.7
    survival_functions = (
        lambda t: np.ones(t.shape),
        lambda t: np.heaviside(4 - t, 1),
        lambda t: np.heaviside(14 - t, 1),
    )

    expected_primary_survivals = np.array(
        [
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
        ]
    )
    expected_secondary_survivals = np.array(
        [
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )
    primary_survivals, secondary_survivals = objective.get_survival_time_vectors(
        survival_functions, primary_travel_times, secondary_travel_times
    )
    assert np.allclose(primary_survivals, expected_primary_survivals)
    assert np.allclose(secondary_survivals, expected_secondary_survivals)


def test_get_not_busy_vector():
    allocation_1 = [0, 0, 0, 0]
    allocation_2 = [0, 1, 1, 1]
    allocation_3 = [1, 2, 3, 4]
    utilisations = [0.2, 0.5, 0.7, 1.0]
    not_busy_1 = objective.get_is_not_busy_vector(utilisations, allocation_1)
    not_busy_2 = objective.get_is_not_busy_vector(utilisations, allocation_2)
    not_busy_3 = objective.get_is_not_busy_vector(utilisations, allocation_3)
    expected_not_busy_1 = np.array(
        [1 - u**a for u, a in zip(utilisations, allocation_1)]
    )
    expected_not_busy_2 = np.array(
        [1 - u**a for u, a in zip(utilisations, allocation_2)]
    )
    expected_not_busy_3 = np.array(
        [1 - u**a for u, a in zip(utilisations, allocation_3)]
    )
    assert np.allclose(not_busy_1, expected_not_busy_1)
    assert np.allclose(not_busy_2, expected_not_busy_2)
    assert np.allclose(not_busy_3, expected_not_busy_3)


def test_get_all_same_closer_busy_vector():
    allocation_1 = [0, 0, 0, 0]
    allocation_2 = [0, 1, 1, 1]
    allocation_3 = [1, 2, 3, 4]
    utilisations = [0.2, 0.5, 0.7, 1.0]
    travel_times = np.array(
        [[0, 5, 10, 15, 20], [5, 0, 5, 10, 15], [10, 5, 0, 5, 10], [15, 10, 5, 0, 5]]
    )
    beta = objective.get_beta(travel_times)

    all_same_busy_1 = objective.get_all_same_closer_busy_vector(
        utilisations, allocation_1, beta
    )
    all_same_busy_2 = objective.get_all_same_closer_busy_vector(
        utilisations, allocation_2, beta
    )
    all_same_busy_3 = objective.get_all_same_closer_busy_vector(
        utilisations, allocation_3, beta
    )
    expected_all_same_busy_1 = np.array(
        [
            [
                np.prod(
                    [
                        utilisations[alpha] ** (allocation_1[alpha] * beta[p][alpha][a])
                        for alpha in range(4)
                    ]
                )
                for p in range(5)
            ]
            for a in range(4)
        ]
    )
    expected_all_same_busy_2 = np.array(
        [
            [
                np.prod(
                    [
                        utilisations[alpha] ** (allocation_2[alpha] * beta[p][alpha][a])
                        for alpha in range(4)
                    ]
                )
                for p in range(5)
            ]
            for a in range(4)
        ]
    )
    expected_all_same_busy_3 = np.array(
        [
            [
                np.prod(
                    [
                        utilisations[alpha] ** (allocation_3[alpha] * beta[p][alpha][a])
                        for alpha in range(4)
                    ]
                )
                for p in range(5)
            ]
            for a in range(4)
        ]
    )
    assert np.allclose(all_same_busy_1, expected_all_same_busy_1)
    assert np.allclose(all_same_busy_2, expected_all_same_busy_2)
    assert np.allclose(all_same_busy_3, expected_all_same_busy_3)


def test_get_all_primary_closer_busy_vector():
    allocation_1 = [0, 0, 0, 0]
    allocation_2 = [0, 1, 1, 1]
    allocation_3 = [1, 2, 3, 4]
    utilisations = [0.2, 0.5, 0.7, 1.0]
    travel_times = np.array(
        [[0, 5, 10, 15, 20], [5, 0, 5, 10, 15], [10, 5, 0, 5, 10], [15, 10, 5, 0, 5]]
    )
    R = objective.get_R(travel_times, travel_times * 0.7)

    all_primary_closer_busy_1 = objective.get_all_primary_closer_busy_vector(
        utilisations, allocation_1, R
    )
    all_primary_closer_busy_2 = objective.get_all_primary_closer_busy_vector(
        utilisations, allocation_2, R
    )
    all_primary_closer_busy_3 = objective.get_all_primary_closer_busy_vector(
        utilisations, allocation_3, R
    )
    expected_all_primary_closer_busy_1 = np.array(
        [
            [
                np.prod(
                    [
                        utilisations[alpha] ** (allocation_1[alpha] * R[p][alpha][a])
                        for alpha in range(4)
                    ]
                )
                for a in range(4)
            ]
            for p in range(5)
        ]
    )
    expected_all_primary_closer_busy_2 = np.array(
        [
            [
                np.prod(
                    [
                        utilisations[alpha] ** (allocation_2[alpha] * R[p][alpha][a])
                        for alpha in range(4)
                    ]
                )
                for a in range(4)
            ]
            for p in range(5)
        ]
    )
    expected_all_primary_closer_busy_3 = np.array(
        [
            [
                np.prod(
                    [
                        utilisations[alpha] ** (allocation_3[alpha] * R[p][alpha][a])
                        for alpha in range(4)
                    ]
                )
                for a in range(4)
            ]
            for p in range(5)
        ]
    )
    assert np.allclose(all_primary_closer_busy_1, expected_all_primary_closer_busy_1)
    assert np.allclose(all_primary_closer_busy_2, expected_all_primary_closer_busy_2)
    assert np.allclose(all_primary_closer_busy_3, expected_all_primary_closer_busy_3)


def test_get_all_secondary_closer_busy_vector():
    allocation_1 = [0, 0, 0, 0]
    allocation_2 = [0, 1, 1, 1]
    allocation_3 = [1, 2, 3, 4]
    utilisations = [0.2, 0.5, 0.7, 1.0]
    travel_times = np.array(
        [[0, 5, 10, 15, 20], [5, 0, 5, 10, 15], [10, 5, 0, 5, 10], [15, 10, 5, 0, 5]]
    )
    R = objective.get_R(travel_times, travel_times * 0.7)

    all_secondary_closer_busy_1 = objective.get_all_secondary_closer_busy_vector(
        utilisations, allocation_1, R
    )
    all_secondary_closer_busy_2 = objective.get_all_secondary_closer_busy_vector(
        utilisations, allocation_2, R
    )
    all_secondary_closer_busy_3 = objective.get_all_secondary_closer_busy_vector(
        utilisations, allocation_3, R
    )
    expected_all_secondary_closer_busy_1 = np.array(
        [
            [
                np.prod(
                    [
                        utilisations[alpha]
                        ** (allocation_1[alpha] * (1 - R[p][a][alpha]))
                        for alpha in range(4)
                    ]
                )
                for a in range(4)
            ]
            for p in range(5)
        ]
    )
    expected_all_secondary_closer_busy_2 = np.array(
        [
            [
                np.prod(
                    [
                        utilisations[alpha]
                        ** (allocation_2[alpha] * (1 - R[p][a][alpha]))
                        for alpha in range(4)
                    ]
                )
                for a in range(4)
            ]
            for p in range(5)
        ]
    )
    expected_all_secondary_closer_busy_3 = np.array(
        [
            [
                np.prod(
                    [
                        utilisations[alpha]
                        ** (allocation_3[alpha] * (1 - R[p][a][alpha]))
                        for alpha in range(4)
                    ]
                )
                for a in range(4)
            ]
            for p in range(5)
        ]
    )
    assert np.allclose(
        all_secondary_closer_busy_1, expected_all_secondary_closer_busy_1
    )
    assert np.allclose(
        all_secondary_closer_busy_2, expected_all_secondary_closer_busy_2
    )
    assert np.allclose(
        all_secondary_closer_busy_3, expected_all_secondary_closer_busy_3
    )


def test_get_objective():
    primary_travel_times = np.array(
        [[0, 5, 10, 15, 20], [5, 0, 5, 10, 15], [10, 5, 0, 5, 10], [15, 10, 5, 0, 5]]
    )
    secondary_travel_times = 0.7 * primary_travel_times
    beta = objective.get_beta(primary_travel_times)
    R = objective.get_R(primary_travel_times, secondary_travel_times)
    survival_functions = (
        lambda t: np.ones(t.shape),
        lambda t: np.ones(t.shape),
        lambda t: np.ones(t.shape),
    )
    primary_survivals, secondary_survivals = objective.get_survival_time_vectors(
        survival_functions, primary_travel_times, secondary_travel_times
    )
    given_utilisations_primary = np.array([0.2, 0.5, 0.7, 1.0])
    given_utilisations_secondary = np.array([0.6, 0.6, 0.2, 0.2])
    demand_rates = np.array(((2, 2, 3, 3, 7), (2, 0, 1, 2, 4), (1, 1, 1, 1, 1))) * 10

    # Some allocation
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=np.array([0, 0, 1]),
        weights_multiple_vehicles=np.array([1, 1, 0]),
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=np.array([1, 0, 0, 1]),
        allocation_secondary=np.array([0, 2, 1, 1]),
        given_utilisations_primary=given_utilisations_primary,
        given_utilisations_secondary=given_utilisations_secondary,
    )
    assert round(g, 4) == 295.1552

    # Zero allocation (cannot save anyone)
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=np.array([0, 0, 1]),
        weights_multiple_vehicles=np.array([1, 1, 0]),
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=np.array([0, 0, 0, 0]),
        allocation_secondary=np.array([0, 0, 0, 0]),
        given_utilisations_primary=given_utilisations_primary,
        given_utilisations_secondary=given_utilisations_secondary,
    )
    assert round(g, 4) == 0.0

    # Over allocation (always saving everyone)
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=np.array([0, 0, 1]),
        weights_multiple_vehicles=np.array([1, 1, 0]),
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=np.array([1000, 1000, 1000, 1000]),
        allocation_secondary=np.array([1000, 1000, 1000, 1000]),
        given_utilisations_primary=given_utilisations_primary,
        given_utilisations_secondary=given_utilisations_secondary,
    )
    assert round(g, 4) == demand_rates.sum()


def test_caching_of_objective():
    """
    This confirms:

    - The cache is modified in place.
    - The cache is used when the function is called again. This is done by
      replacing the value of the cache with a nonsensical value.
    - The cache can hold multiple values.
    """
    cache = {}

    primary_travel_times = np.array(
        [[0, 5, 10, 15, 20], [5, 0, 5, 10, 15], [10, 5, 0, 5, 10], [15, 10, 5, 0, 5]]
    )
    secondary_travel_times = 0.7 * primary_travel_times
    beta = objective.get_beta(primary_travel_times)
    R = objective.get_R(primary_travel_times, secondary_travel_times)
    survival_functions = (
        lambda t: np.ones(t.shape),
        lambda t: np.ones(t.shape),
        lambda t: np.ones(t.shape),
    )
    primary_survivals, secondary_survivals = objective.get_survival_time_vectors(
        survival_functions, primary_travel_times, secondary_travel_times
    )
    given_utilisations_primary = np.array([0.2, 0.5, 0.7, 1.0])
    given_utilisations_secondary = np.array([0.6, 0.6, 0.2, 0.2])
    demand_rates = np.array(((2, 2, 3, 3, 7), (2, 0, 1, 2, 4), (1, 1, 1, 1, 1))) * 10

    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=np.array([0, 0, 1]),
        weights_multiple_vehicles=np.array([1, 1, 0]),
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=np.array([1000, 1000, 1000, 1000]),
        allocation_secondary=np.array([1000, 1000, 1000, 1000]),
        given_utilisations_primary=given_utilisations_primary,
        given_utilisations_secondary=given_utilisations_secondary,
        cache=cache,
    )
    assert cache == {
        ("[1000 1000 1000 1000]", "[1000 1000 1000 1000]"): demand_rates.sum()
    }

    cache[("[1000 1000 1000 1000]", "[1000 1000 1000 1000]")] = -10
    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=np.array([0, 0, 1]),
        weights_multiple_vehicles=np.array([1, 1, 0]),
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=np.array([1000, 1000, 1000, 1000]),
        allocation_secondary=np.array([1000, 1000, 1000, 1000]),
        given_utilisations_primary=given_utilisations_primary,
        given_utilisations_secondary=given_utilisations_secondary,
        cache=cache,
    )

    assert g == -10

    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=np.array([0, 0, 1]),
        weights_multiple_vehicles=np.array([1, 1, 0]),
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=np.array([0, 0, 0, 0]),
        allocation_secondary=np.array([0, 0, 0, 0]),
        given_utilisations_primary=given_utilisations_primary,
        given_utilisations_secondary=given_utilisations_secondary,
        cache=cache,
    )
    assert cache == {
        ("[1000 1000 1000 1000]", "[1000 1000 1000 1000]"): -10,
        ("[0 0 0 0]", "[0 0 0 0]"): 0,
    }

    g = objective.get_objective(
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=np.array([0, 0, 1]),
        weights_multiple_vehicles=np.array([1, 1, 0]),
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.given_utilisations,
        allocation_primary=np.array([1, 0, 0, 1]),
        allocation_secondary=np.array([0, 2, 1, 1]),
        given_utilisations_primary=given_utilisations_primary,
        given_utilisations_secondary=given_utilisations_secondary,
        cache=cache,
    )
    assert round(g, 4) == 295.1552
    assert round(cache[("[1 0 0 1]", "[0 2 1 1]")], 4) == 295.1552

    assert cache.keys() == {
        ("[1000 1000 1000 1000]", "[1000 1000 1000 1000]"),
        ("[0 0 0 0]", "[0 0 0 0]"),
        ("[1 0 0 1]", "[0 2 1 1]"),
    }
