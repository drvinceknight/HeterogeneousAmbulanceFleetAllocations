# Hyperparameters
pop_size = 100
keep_size = 20
initial_mutations = 6
cooling_rate = 0.1
num_generations = 200
max_vehicles = 10

# Parallelising
n_cores = 100

resource_levels = [
    60,
    68,
    76,
    84,
    92,
    100,
    108,
    116,
    124,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    62,
    70,
    78,
    86,
    94,
    102,
    110,
    118,
    66,
    74,
    82,
    90,
    98,
    106,
    114,
    122,
]

full_command_string = ""

id_number = 90000
id_number_full = 80000

for demand in [13, 19, 34, 45]:
    for level in resource_levels:
        echo_command = f'echo "Doing resource level {level} on demand {demand}."\n'
        command_single = f"python src/experiment.py {level} 0 {max_vehicles} {max_vehicles} {pop_size} {keep_size} {num_generations} {initial_mutations} {cooling_rate} {demand} {id_number} {n_cores} --progress_bar\n"
        command_multiple = f"python src/experiment-full-mutation.py {level} {max_vehicles} {max_vehicles} {pop_size} {keep_size} {num_generations} {initial_mutations} {cooling_rate} {demand} {id_number_full} {n_cores} --progress_bar\n"
        full_command_string += echo_command
        full_command_string += command_single
        full_command_string += command_multiple
        id_number += 1
        id_number_full += 1

for demand in [13, 19, 34, 45]:
    for level in resource_levels:
        echo_command = f'echo "Doing resource level {level + 1} on demand {demand}."\n'
        command_single = f"python src/experiment.py {level + 1} 0 {max_vehicles} {max_vehicles} {pop_size} {keep_size} {num_generations} {initial_mutations} {cooling_rate} {demand} {id_number} {n_cores} --progress_bar\n"
        command_multiple = f"python src/experiment-full-mutation.py {level + 1} {max_vehicles} {max_vehicles} {pop_size} {keep_size} {num_generations} {initial_mutations} {cooling_rate} {demand} {id_number_full} {n_cores} --progress_bar\n"
        full_command_string += echo_command
        full_command_string += command_single
        full_command_string += command_multiple
        id_number += 1
        id_number_full += 1

with open("jobs.txt", "w") as f:
    f.write(full_command_string)
