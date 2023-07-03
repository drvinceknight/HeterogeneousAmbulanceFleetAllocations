import numpy as np
import objective
import utilisation
import optimisation
import argparse
import pathlib

if __name__ == "__main__":
    """
    RECOMMENDED PARAMETERS (GUESSES AT THE MOMENT, BASED ON USING 64 CORES)
        max_primary=10,
        max_secondary=10,
        population_size=240,
        keep_size=40,
        number_of_iterations=500,
        initial_number_of_mutatation_repetitions=6,
        cooling_rate=0.25,
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "total_primary",
        type=int,
        help="Total number of primary vehicles in the allocation.",
    )
    parser.add_argument(
        "total_secondary",
        type=int,
        help="Total number of secondary vehicles in the allocation.",
    )
    parser.add_argument(
        "max_primary",
        type=int,
        help="Maximum number of primary vehicles to place in the same location.",
    )
    parser.add_argument(
        "max_secondary",
        type=int,
        help="Maximum number of secondary vehicles to place in the same location.",
    )
    parser.add_argument(
        "population_size",
        type=int,
        help="Number of potential solutions in a population.",
    )
    parser.add_argument(
        "keep_size",
        type=int,
        help="Number of solutions to keep for the next generation.",
    )
    parser.add_argument(
        "number_of_iterations",
        type=int,
        help="Number of iterations to run the optimisation for.",
    )
    parser.add_argument(
        "initial_number_of_mutatation_repetitions",
        type=int,
        help="The number of mutations to carry out successivly at the beginning of the optimisation.",
    )
    parser.add_argument(
        "cooling_rate",
        type=float,
        help="The rate at which the number of successive mutations decreases.",
    )
    parser.add_argument(
        "demand_scenario",
        type=str,
        help="The demand scenario to use (13, 19, 34, or 45).",
    )
    parser.add_argument("num_workers", type=int, help="The number of cores to use.")
    parser.add_argument("progress_bar", type=bool, help="Use a progress bar or not.")
    args = parser.parse_args()

    ## Read in all data (time units in minutes)
    raw_travel_times = np.genfromtxt("./data/travel_times_matrix.csv", delimiter=",")
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
    vehicle_locations, pickup_locations = tuple(map(range, raw_travel_times.shape))
    weights_single_vehicle = np.array([0, 0, 1])
    weights_multiple_vehicles = np.array([1, 1, 0])
    primary_survivals, secondary_survivals = objective.get_survival_time_vectors(
        survival_functions, primary_vehicle_travel_times, secondary_vehicle_travel_times
    )
    service_rate_primary = 1 / (4.5 * 60)
    service_rate_secondary = 1 / (3.5 * 60)
    demand_rates = (
        np.genfromtxt(f"./data/demand_{args.demand_scenario}.csv", delimiter=",") / 1440
    )
    results_dir = pathlib.Path("./results")
    results_dir.mkdir(exist_ok=True)

    # Carry out the optimisation
    (
        best_primary,
        best_secondary,
        objective_by_iteration,
    ) = optimisation.optimise(
        number_of_locations=67,
        number_of_primary_vehicles=args.total_primary,
        number_of_secondary_vehicles=args.total_secondary,
        max_primary=args.max_primary,
        max_secondary=args.max_secondary,
        population_size=args.population_size,
        keep_size=args.keep_size,
        number_of_iterations=args.number_of_iterations,
        mutation_function=optimisation.mutate_retain_vehicle_numbers,
        initial_number_of_mutatation_repetitions=args.initial_number_of_mutatation_repetitions,
        cooling_rate=args.cooling_rate,
        demand_rates=demand_rates,
        primary_survivals=primary_survivals,
        secondary_survivals=secondary_survivals,
        weights_single_vehicle=weights_single_vehicle,
        weights_multiple_vehicles=weights_multiple_vehicles,
        beta=beta,
        R=R,
        vehicle_station_utilisation_function=utilisation.solve_utilisations,
        seed=0,
        num_workers=args.num_workers,
        progress_bar=args.progress_bar,
        service_rate_primary=service_rate_primary,
        service_rate_secondary=service_rate_secondary,
    )
    np.savetxt(
        f"./results/allocation_primary_demand={args.demand_scenario}_primary={args.total_primary}_secondary={args.total_secondary}.csv",
        best_primary,
        delimiter=",",
    )
    np.savetxt(
        f"./results/allocation_secondary_demand={args.demand_scenario}_primary={args.total_primary}_secondary={args.total_secondary}.csv",
        best_secondary,
        delimiter=",",
    )
    np.savetxt(
        f"./results/population_objectives_demand={args.demand_scenario}_primary={args.total_primary}_secondary={args.total_secondary}.csv",
        objective_by_iteration,
        delimiter=",",
    )
