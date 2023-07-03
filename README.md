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
 - scenario_id=AAAA
 - num_workers=62

Run:

```bash
<<<<<<< HEAD
$ python src/experiment.py 70 26 10 10 240 40 500 6 0.25 13 33333 62
=======
$ python src/experiment.py 70 26 10 10 240 40 500 6 0.25 13 AAAA 62
>>>>>>> 307f569 (experiment script saves hyperparameters along with results; takes an id; fixes progres_bar argparse)
```

To also implement a progress bar across the iterations, run:

```bash
<<<<<<< HEAD
$ python src/experiment.py 70 26 10 10 240 40 500 6 0.25 13 33333 62 --progress_bar
=======
$ python src/experiment.py 70 26 10 10 240 40 500 6 0.25 13 AAAA 62 --progress_bar
>>>>>>> 307f569 (experiment script saves hyperparameters along with results; takes an id; fixes progres_bar argparse)
```
