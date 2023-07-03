# upgraded-octo-barnacle
Code for optimising ambulance allocation


To run the experiments from the root of this repository with:
 - total_primary=70
 - total_secondary=26
 - max_primary=10
 - max_secondary=10
 - population_size=240
 - keep_size=40
 - number_of_iterations=500
 - initial_number_of_mutatation_repetitions=6
 - cooling_rate=0.25
 - demand_scenario=13
 - num_workers=62
 - progress_bar=False

Run:

```bash
$ python src/experiment.py 70 26 10 10 240 40 500 6 0.25 13 62 False
```
