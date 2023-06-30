import utilisation
import objective
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


def test_get_lambda_differences_primary():
    ## Time units in minutes
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
    demand_rates = np.genfromtxt("./test_data/demand.csv", delimiter=",") / 1440
    service_rate_primary = 1 / (4.5 * 60)
    service_rate_secondary = 1 / (3.5 * 60)
    allocation_primary = np.ones(67)
    allocation_secondary = np.ones(67)

    diffs_0 = utilisation.get_lambda_differences_primary(
        lhs=np.zeros(67),
        service_rate_primary=service_rate_primary,
        allocation_primary=allocation_primary,
        beta=beta,
        demand_rates=demand_rates,
    )

    assert round(diffs_0.sum(), 7) == 0.1823491
    assert round(diffs_0.min(), 7) == 0.0000986
    assert round(diffs_0.max(), 7) == 0.0140919


def test_get_lambda_differences_secondary():
    ## Time units in minutes
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
    demand_rates = np.genfromtxt("./test_data/demand.csv", delimiter=",") / 1440
    service_rate_primary = 1 / (4.5 * 60)
    service_rate_secondary = 1 / (3.5 * 60)
    allocation_primary = np.ones(67)
    allocation_secondary = np.ones(67)
    primary_utilisations = np.ones(67) * 0.6

    diffs_0 = utilisation.get_lambda_differences_secondary(
        lhs=np.zeros(67),
        service_rate_secondary=service_rate_secondary,
        allocation_secondary=allocation_secondary,
        allocation_primary=allocation_primary,
        utilisations_primary=primary_utilisations,
        beta=beta,
        R=R,
        demand_rates=demand_rates,
    )

    assert round(diffs_0.sum(), 7) == 0.1029984
    assert round(diffs_0.min(), 7) == 0.0000397
    assert round(diffs_0.max(), 7) == 0.0072913


def test_solve_utilisations():
    ## Time units in minutes
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
    demand_rates = np.genfromtxt("./test_data/demand.csv", delimiter=",") / 1440
    service_rate_primary = 1 / (4.5 * 60)
    service_rate_secondary = 1 / (3.5 * 60)
    allocation_primary = np.ones(67)
    allocation_secondary = np.ones(67)

    primary_utilisations, secondary_utilisations = utilisation.solve_utilisations(
        allocation_primary=allocation_primary,
        allocation_secondary=allocation_secondary,
        beta=beta,
        R=R,
        demand_rates=demand_rates,
        service_rate_primary=service_rate_primary,
        service_rate_secondary=service_rate_secondary,
    )

    assert len(primary_utilisations) == 67
    assert len(secondary_utilisations) == 67
    assert max(primary_utilisations) <= 1.0
    assert max(secondary_utilisations) <= 1.0
    assert min(primary_utilisations) >= 0.0
    assert min(secondary_utilisations) >= 0.0

    # Some slight demand loss (due to unmet demand)
    assert (
        primary_utilisations * service_rate_primary * allocation_primary
    ).sum() * 1440 == 262.1546962546663
    assert demand_rates.sum() * 1440 == 262.5826343840001

    # More significant demand loss (due to primary vehicles getting there first)
    assert np.isclose(
        (secondary_utilisations * service_rate_secondary * allocation_secondary).sum()
        * 1440,
        158.21533513661336,
    )
    assert demand_rates[:-1].sum() * 1440 == 175.664826165


def test_solve_utilisations_when_flooding():
    ## Time units in minutes
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
    demand_rates = np.genfromtxt("./test_data/demand.csv", delimiter=",") * 100 / 1440
    service_rate_primary = 1 / (4.5 * 60)
    service_rate_secondary = 1 / (3.5 * 60)
    allocation_primary = np.ones(67)
    allocation_secondary = np.ones(67)

    primary_utilisations, secondary_utilisations = utilisation.solve_utilisations(
        allocation_primary=allocation_primary,
        allocation_secondary=allocation_secondary,
        beta=beta,
        R=R,
        demand_rates=demand_rates,
        service_rate_primary=service_rate_primary,
        service_rate_secondary=service_rate_secondary,
    )
    assert np.allclose(primary_utilisations, np.array([0.99 for _ in range(67)]))
    assert np.allclose(secondary_utilisations, np.array([0.99 for _ in range(67)]))
