# upgraded-octo-barnacle
Code for optimising ambulance allocation


To run the experiments from home with:
    + max_primary=10
    + max_secondary=10
    + population_size=240
    + keep_size=40
    + number_of_iterations=500
    + initial_number_of_mutatation_repetitions=6
    + cooling_rate=0.25
    + demand_scenario=13
    + min_resource_level=60
    + max_resource_level=125
    + num_workers=62
    + progress_bar=False

Run:

$ python src/experiment.py 10 10 240 40 500 6 0.25 13 60 125 62 False