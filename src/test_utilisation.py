import utilisation
import numpy as np


def test_constant_utilisation():
    allocation_1 = np.array([0, 1, 1, 0, 1])
    allocation_2 = np.array([100, 30, 45, 2, 0, 0, 22, 77, 81, 0, 100, 100, 0, 15])
    constant_0 = 0.0
    constant_4 = 0.4
    constant_7 = 0.7
    constant_1 = 1.0

    utils_p1, utils_s0 = utilisation.constant_utilisation(
        allocation_1, allocation_2, constant_1, constant_0
    )
    utils_p7, utils_s4 = utilisation.constant_utilisation(
        allocation_1, allocation_2, constant_7, constant_4
    )
    utils_p4, utils_s7 = utilisation.constant_utilisation(
        allocation_1, allocation_2, constant_4, constant_7
    )
    utils_p0, utils_s1 = utilisation.constant_utilisation(
        allocation_1, allocation_2, constant_0, constant_1
    )

    assert np.allclose(utils_p0, np.zeros(5))
    assert np.allclose(utils_s0, np.zeros(14))
    assert np.allclose(utils_p4, np.ones(5) * 0.4)
    assert np.allclose(utils_s4, np.ones(14) * 0.4)
    assert np.allclose(utils_p7, np.ones(5) * 0.7)
    assert np.allclose(utils_s7, np.ones(14) * 0.7)
    assert np.allclose(utils_p1, np.ones(5))
    assert np.allclose(utils_s1, np.ones(14))


def test_given_utilisations():
    given_1 = np.array([0.7, 0.8, 0.0, 0.1])
    given_2 = np.array(
        [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.7, 0.7]
    )

    utils_1, utils_2 = utilisation.given_utilisations(given_1, given_2)

    assert np.allclose(utils_1, given_1)
    assert np.allclose(utils_2, given_2)


def test_proportional_utilisations_primary():
    allocation_primary = [0, 1, 2, 0, 2]
    service_rate_primary = 10
    allocation_secondary = [0, 1, 2, 0, 2]
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
    expected_utils_primary = np.array([0, 15, 7.5, 0, 7.5])
    expected_utils_secondary = np.array([0, 7.5, 3.75, 0, 3.75])
    (
        obtained_utils_primary,
        obtained_utils_secondary,
    ) = utilisation.proportional_utilisations(
        allocation_primary=allocation_primary,
        allocation_secondary=allocation_secondary,
        demand_rates=demand_rates,
        service_rate_primary=service_rate_primary,
        service_rate_secondary=service_rate_secondary,
    )
    assert np.allclose(expected_utils_primary, obtained_utils_primary)
    assert np.allclose(expected_utils_secondary, obtained_utils_secondary)
