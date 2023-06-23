import utilisation
import numpy as np


def test_constant_utilisation_primary():
    allocation_1 = np.array([0, 1, 1, 0, 1])
    allocation_2 = np.array([100, 30, 45, 2, 0, 0, 22, 77, 81, 0, 100, 100, 0, 15])
    constant_0 = 0.0
    constant_4 = 0.4
    constant_7 = 0.7
    constant_1 = 1.0

    utils_01 = utilisation.constant_utilisation_primary(allocation_1, constant_0)
    utils_02 = utilisation.constant_utilisation_primary(allocation_2, constant_0)
    utils_41 = utilisation.constant_utilisation_primary(allocation_1, constant_4)
    utils_42 = utilisation.constant_utilisation_primary(allocation_2, constant_4)
    utils_71 = utilisation.constant_utilisation_primary(allocation_1, constant_7)
    utils_72 = utilisation.constant_utilisation_primary(allocation_2, constant_7)
    utils_11 = utilisation.constant_utilisation_primary(allocation_1, constant_1)
    utils_12 = utilisation.constant_utilisation_primary(allocation_2, constant_1)

    assert np.allclose(utils_01, np.zeros(5))
    assert np.allclose(utils_02, np.zeros(14))
    assert np.allclose(utils_41, np.ones(5) * 0.4)
    assert np.allclose(utils_42, np.ones(14) * 0.4)
    assert np.allclose(utils_71, np.ones(5) * 0.7)
    assert np.allclose(utils_72, np.ones(14) * 0.7)
    assert np.allclose(utils_11, np.ones(5))
    assert np.allclose(utils_12, np.ones(14))


def test_constant_utilisation_secondary():
    allocation_1 = np.array([0, 1, 1, 0, 1])
    allocation_2 = np.array([100, 30, 45, 2, 0, 0, 22, 77, 81, 0, 100, 100, 0, 15])
    constant_0 = 0.0
    constant_4 = 0.4
    constant_7 = 0.7
    constant_1 = 1.0

    utils_01 = utilisation.constant_utilisation_secondary(allocation_1, constant_0)
    utils_02 = utilisation.constant_utilisation_secondary(allocation_2, constant_0)
    utils_41 = utilisation.constant_utilisation_secondary(allocation_1, constant_4)
    utils_42 = utilisation.constant_utilisation_secondary(allocation_2, constant_4)
    utils_71 = utilisation.constant_utilisation_secondary(allocation_1, constant_7)
    utils_72 = utilisation.constant_utilisation_secondary(allocation_2, constant_7)
    utils_11 = utilisation.constant_utilisation_secondary(allocation_1, constant_1)
    utils_12 = utilisation.constant_utilisation_secondary(allocation_2, constant_1)

    assert np.allclose(utils_01, np.zeros(5))
    assert np.allclose(utils_02, np.zeros(14))
    assert np.allclose(utils_41, np.ones(5) * 0.4)
    assert np.allclose(utils_42, np.ones(14) * 0.4)
    assert np.allclose(utils_71, np.ones(5) * 0.7)
    assert np.allclose(utils_72, np.ones(14) * 0.7)
    assert np.allclose(utils_11, np.ones(5))
    assert np.allclose(utils_12, np.ones(14))


def test_given_utilisations_primary():
    given_1 = np.array([0.7, 0.8, 0.0, 0.1])
    given_2 = np.array(
        [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.7, 0.7]
    )

    utils_1 = utilisation.given_utilisations_primary(given_1)
    utils_2 = utilisation.given_utilisations_primary(given_2)

    assert np.allclose(utils_1, given_1)
    assert np.allclose(utils_2, given_2)


def test_given_utilisations_secondary():
    given_1 = np.array([0.7, 0.8, 0.0, 0.1])
    given_2 = np.array(
        [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.7, 0.7]
    )

    utils_1 = utilisation.given_utilisations_secondary(given_1)
    utils_2 = utilisation.given_utilisations_secondary(given_2)

    assert np.allclose(utils_1, given_1)
    assert np.allclose(utils_2, given_2)


def test_proportional_utilisations_primary():
    allocation = [0, 1, 2, 0, 2]
    service_rate_primary = 10
    demand_rates = np.array(
        [
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            [
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
            ],
            [
                [3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3],
            ],
        ]
    )
    expected_utils = np.array([0, 15, 7.5, 0, 7.5])
    obtained_utils = utilisation.proportional_utilisations_primary(
        allocation_primary=allocation, demand_rates=demand_rates, service_rate_primary=service_rate_primary
    )
    assert np.allclose(expected_utils, obtained_utils)


def test_proportional_utilisations_secondary():
    allocation = [0, 1, 2, 0, 2]
    service_rate_secondary = 10
    demand_rates = np.array(
        [
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            [
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
            ],
            [
                [3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3],
            ],
        ]
    )
    expected_utils = np.array([0, 7.5, 3.75, 0, 3.75])
    obtained_utils = utilisation.proportional_utilisations_secondary(
        allocation_secondary=allocation, demand_rates=demand_rates, service_rate_secondary=service_rate_secondary
    )
    assert np.allclose(expected_utils, obtained_utils)
