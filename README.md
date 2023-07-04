# upgraded-octo-barnacle

Source code for the optimisation problem presented in the paper *Evaluating
Heterogeneous Ambulance Fleet Allocations in Jakarta*.

## Creating a virtual environment

```bash
$ python -m venv env/
```

Activate environment:

```bash
$ source env/bin/activate
```

Install all dependencies (including development)

```bash
$ python -m pip install -r requirements.txt
```

## Running an experiment


To run the experiments from the root of this repository with:

 - total_primary=70
 - total_secondary=26
 - max_primary=10
 - max_secondary=10
 - population_size=240
 - keep_size=40
 - number_of_iterations=500
 - initial_number_of_mutation_repetitions=6
 - cooling_rate=0.25
 - demand_scenario=13
 - scenario_id=33333
 - num_workers=62

Run:

```bash
$ python src/experiment.py 70 26 10 10 240 40 500 6 0.25 13 33333 62
```

To also implement a progress bar across the iterations, run:

```bash
$ python src/experiment.py 70 26 10 10 240 40 500 6 0.25 13 33333 62 --progress_bar
```
