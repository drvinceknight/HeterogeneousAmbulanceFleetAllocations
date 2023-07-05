# Hyperparameters
pop_size = 100
keep_size = 15
initial_mutations = 6
cooling_rate = 0.1
num_generations = 200
max_vehicles = 15

# Parallelising
n_cores = 100

current = (81, 13)

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
    61,
    69,
    77,
    85,
    93,
    101,
    109,
    117,
    125,
    65,
    73,
    81,
    89,
    97,
    105,
    113,
    121,
    63,
    71,
    79,
    87,
    95,
    103,
    111,
    119,
    67,
    75,
    83,
    91,
    99,
    107,
    115,
    123,
]

vehicle_type_combinations = [current]
for rl in resource_levels:
    primary = rl
    secondary = 0
    while secondary <= primary:
        vehicle_type_combinations.append((primary, secondary))
        primary -= 1
        secondary += 3


full_command_string = ""

id_number = 0
for demand in [13, 19, 34, 45]:
    for combination in vehicle_type_combinations:
        command = f"python src/experiment.py {combination[0]} {combination[1]} {max_vehicles} {max_vehicles} {pop_size} {keep_size} {num_generations} {initial_mutations} {cooling_rate} {demand} {str(id_number).zfill(6)} {n_cores}\n"
        full_command_string += command
        id_number += 1

with open("jobs.txt", "w") as f:
    f.write(full_command_string)
