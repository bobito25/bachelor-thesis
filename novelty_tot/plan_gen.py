from typing import Generator
from lifted_pddl import Parser
import os
import random
from pathlib import Path
import subprocess


def generate_random_plan(num_actions: int, domain_path: str, problem_path: str) -> list[tuple[str, tuple]]:
    """
    Generate a random plan of a given length using the provided domain and problem files.
    The plan consists of tuples of action names and their parameters.
    """
    parser = Parser()
    parser.parse_domain(domain_path)
    parser.parse_problem(problem_path)
    plan = []
    for _ in range(num_actions):
        applicable_actions = get_applicable_actions_with_parser(parser)
        if not applicable_actions:
            break
        action = random.choice(applicable_actions)
        plan.append(action)
        parser.set_current_state(parser.get_next_state(action[0], action[1]))
    return plan


def check_if_goal_state(domain_path: str, problem_path: str, state_atoms: set[tuple[str, tuple]]) -> bool:
    """
    Check if the given state is a goal state using the provided state atoms and the domain and problem files.
    """
    parser = Parser()
    parser.parse_domain(domain_path)
    parser.parse_problem(problem_path)
    for goal_atom in parser.goals:
        is_true, action_name, params = goal_atom
        action = tuple([action_name, params])
        if is_true:
            if action in state_atoms:
                continue
            else:
                return False
        else:
            if action in state_atoms:
                return False
            else:
                continue
    return True


def check_if_goal_state_with_parser(parser: Parser) -> bool:
    """
    Check if the given state is a goal state using the provided state atoms and the parser.
    """
    state_atoms = parser.atoms
    for goal_atom in parser.goals:
        is_true, action_name, params = goal_atom
        action = tuple([action_name, params])
        if is_true:
            if action in state_atoms:
                continue
            else:
                return False
        else:
            if action in state_atoms:
                return False
            else:
                continue
    return True


def goal_atoms_to_string(parser: Parser, goal_atoms: set[tuple[bool, str, tuple]]) -> str:
    """Convert goal atoms to a string representation."""
    goal_strs = []
    for goal_atom in goal_atoms:
        is_true, action_name, params = goal_atom
        pddl = parser.encode_ground_actions_as_pddl({action_name: [params]}, output_format='str')
        if is_true:
            goal_str = str(list(pddl)[0])
        else:
            goal_str = "(not " + str(pddl) + ")"
        goal_strs.append(goal_str)
    return " and ".join(goal_strs)


def get_applicable_actions_with_parser(parser: Parser) -> list[tuple[str, tuple]]:
    """
    Get applicable actions from the parser.
    The actions are returned as a list of tuples, where each tuple contains the action name and its parameters.
    """
    applicable_actions_dict = parser.get_applicable_actions()

    applicable_actions = []
    for action_name, possible_action_params in applicable_actions_dict.items():
        for params in possible_action_params:
            applicable_actions.append((action_name, params))
    return applicable_actions


def generate_non_goal_plan(num_actions: int, domain_path: str, problem_path: str) -> list[tuple[str, tuple]]:
    """
    Generate a non-goal plan up to a given length using the provided domain and problem files.
    The plan consists of tuples of action names and their parameters.
    The plan is guaranteed to not reach the goal state.
    This is done by recursively generating a plan step by step and checking if the goal is reached in each step.
    If the goal is reached in the current step, one step is taken back and a different action is generated.
    It is a tree search in the state space of the problem.
    The plan is not guaranteed to be of the exact length specified, but it will be less than or equal to that length.
    """
    def _recurse(num_actions: int, parser: Parser) -> list[tuple[str, tuple]]:
        if num_actions <= 0:
            raise ValueError("num_actions must be greater than 0")

        applicable_actions = get_applicable_actions_with_parser(parser)
        
        if not applicable_actions:
            return []

        random.shuffle(applicable_actions)

        # Store the original state to backtrack if needed
        orig_state = parser.atoms.copy()

        # Try to find a valid action that doesn't lead to the goal state
        for action in applicable_actions:
            name, params = action
            parser.set_current_state(parser.get_next_state(name, params))

            if check_if_goal_state(domain_path, problem_path, parser.atoms):
                # If the goal is reached, backtrack and try a different action
                parser.set_current_state(orig_state)
                continue

            if num_actions > 1:
                # Recurse to generate the rest of the plan
                sub_plan = _recurse(num_actions - 1, parser)
                if sub_plan == []:
                    # Backtrack if no valid plan was found
                    parser.set_current_state(orig_state)
                    continue
                else:
                    return [action] + sub_plan
            else:
                # If this is the last action, return it
                return [action]

        # No applicable actions that don't lead to the goal state
        return []

    parser = Parser()
    parser.parse_domain(domain_path)
    parser.parse_problem(problem_path)

    return _recurse(num_actions, parser)


def gen_all_plans(domain_path: str, problem_path: str, max_length: int) -> Generator[tuple[list[tuple[str, tuple]], bool], None, None]:
    """Generate all plans up to a maximum length plus whether they reach the goal state (BFS)."""
    parser = Parser()
    parser.parse_domain(domain_path)
    parser.parse_problem(problem_path)

    cur_states = [(parser.atoms.copy(), [])]  # (state_atoms, plan)

    for _ in range(max_length):
        next_states = []
        for state_atoms, plan in cur_states:
            parser.set_current_state(state_atoms)
            applicable_actions = get_applicable_actions_with_parser(parser)
            for action in applicable_actions:
                # Ensure action is applied to the correct parent state
                parser.set_current_state(state_atoms)
                name, params = action
                new_state = parser.get_next_state(name, params)
                # Goal test
                parser.set_current_state(new_state)
                is_goal = check_if_goal_state_with_parser(parser)
                # Store successor
                new_plan = plan + [action]
                next_states.append((new_state.copy(), new_plan))
                yield new_plan, is_goal
        if not next_states:
            break
        cur_states = next_states.copy()


def interactive_traversal(domain_path: str, problem_path: str) -> None:
    """Interactively traverse the state space of a planning problem."""
    parser = Parser()
    parser.parse_domain(domain_path)
    parser.parse_problem(problem_path)

    while True:
        print("Current state:")
        print(state_atoms_to_string(parser, parser.atoms))
        if check_if_goal_state_with_parser(parser):
            print("Goal state reached!")
            break
        applicable_actions = get_applicable_actions_with_parser(parser)
        if not applicable_actions:
            print("No applicable actions, dead end!")
            break
        print("Applicable actions:")
        for i, action in enumerate(applicable_actions):
            print(f"{i}: {action}")
        choice = input("Choose an action by number (or 'q' to quit): ")
        if choice.lower() == 'q':
            break
        try:
            choice_idx = int(choice)
            if 0 <= choice_idx < len(applicable_actions):
                action = applicable_actions[choice_idx]
                name, params = action
                parser.set_current_state(parser.get_next_state(name, params))
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Invalid input, try again.")


def plan_to_pddl(domain_path: str, problem_path: str, plan: list[tuple[str, tuple]]) -> str:
    """Convert a plan to a pddl string representation."""
    parser = Parser()
    parser.parse_domain(domain_path)
    parser.parse_problem(problem_path)
    action_strs = []
    for action, params in plan:
        parsed_str = parser.encode_ground_actions_as_pddl({action: [params]}, output_format='str')
        action_strs.append(str(list(parsed_str)[0]))
    return "\n".join(action_strs)


def state_atoms_to_string(parser: Parser, state_atoms: set[tuple[str,tuple]]) -> str:
    """Convert state atoms to a string representation."""
    pddl_atoms = list(parser.encode_atoms_as_pddl(state_atoms, output_format='str'))
    return "{" + ", ".join(pddl_atoms) + "}"


def compute_optimal_plan(domain_path, instance_path, verbose=False) -> str:
    """Compute a plan using Fast Downward."""
    fast_downward_path = os.getenv("FAST_DOWNWARD")
    if not fast_downward_path:
        if verbose:
            print("Environment variable FAST_DOWNWARD is not set. Please set it to the path of Fast Downward.")
        return ""
    plan_file_path = "sas_plan"
    path = os.path.join(fast_downward_path, "fast-downward.py")
    args = ["python", path, domain_path, instance_path, "--search", "astar(lmcut())"]
    if verbose:
        print(f"Using FastDownward at {fast_downward_path}")
        print("Running command:", " ".join(args))
        subprocess.run(args, check=False)
    else:
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if not os.path.exists(plan_file_path):
        if verbose:
            print(f"Error in using FastDownward: plan file not found at {plan_file_path}.")
        return ""
    raw = Path(plan_file_path).read_text()
    plan = raw.split(";")[0]
    if verbose:
        print(f"Plan computed: {plan}")
    return plan


def generate_blocksworld_instances(num_instances: int, output_dir: str, max_plan_length: int = 100, num_blocks: int = 4, overwrite: bool = False) -> None:
    domain_path = "test_data/PDDL/Blocksworld/domain.pddl"
    for id in range(num_instances):
        instance_path = os.path.join(output_dir, f"instance-{id+1}.pddl")
        if os.path.exists(instance_path) and not overwrite:
            print(f"Instance {id+1} already exists, skipping...")
            continue
        max_retries = 15
        tries = 0
        while tries < max_retries:
            generate_blocksworld_instance(instance_path, num_blocks=num_blocks)
            tries += 1
            plan = compute_optimal_plan(domain_path, instance_path)
            if not plan:
                print(f"Failed to compute plan for instance {id+1}, trying again...")
                continue
            plan_length = len(plan.split("\n"))
            if plan_length < max_plan_length:
                print(f"Generated instance {id+1} with plan length {plan_length}.")
                break
            else:
                print(f"Generated instance {id+1} with plan length {plan_length}, which is too long. Trying again...")
        else:
            print(f"Failed to generate a valid instance {id+1} after {max_retries} tries.")
            return


def generate_blocksworld_instance(output_instance_path: str, num_blocks: int = 4) -> None:
    #blocksworld_data_gen_path = "test_data\\PDDL\\Blocksworld\\bwstates.1\\bwstates.exe"
    #blocksworld_2pddl_path = "test_data\\PDDL\\Blocksworld\\2pddl\\2pddl.exe"

    blocksworld_data_gen_path = "./test_data/PDDL/Blocksworld/bwstates.1/bwstates"
    blocksworld_2pddl_path = "./test_data/PDDL/Blocksworld/2pddl/2pddl"

    # run generator
    seed = random.randint(0, 100000)
    data_gen_cmd = f"{blocksworld_data_gen_path} -r {seed} -n {num_blocks} -s 2"
    output_data = subprocess.run(data_gen_cmd, shell=True, capture_output=True, text=True)

    # save the output to a file
    with open(f"STATES_DATA", "w") as f:
        f.write(output_data.stdout)
    
    # convert to pddl
    to_pddl_cmd = f"{blocksworld_2pddl_path} -d STATES_DATA -n {num_blocks}"
    pddl_output = subprocess.run(to_pddl_cmd, shell=True, capture_output=True, text=True)

    # save the pddl output to a file
    with open(f"{output_instance_path}", "w") as f:
        f.write(pddl_output.stdout)

    # delete the states data file
    subprocess.run("rm STATES_DATA", shell=True)


def generate_logistics_instances(num_instances: int, output_dir: str, max_plan_length: int = 8, overwrite: bool = False, domain_params: dict = None) -> None:
    domain_path = "test_data/PDDL/logistics/domain.pddl"
    instances = set()
    for id in range(num_instances):
        instance_path = os.path.join(output_dir, f"instance-{id+1}.pddl")
        if os.path.exists(instance_path) and not overwrite:
            print(f"Instance {id+1} already exists, skipping...")
            continue
        max_retries = 15
        tries = 0
        while tries < max_retries:
            instance_str = generate_logistics_instance(**(domain_params or {}))
            if instance_str in instances:
                print(f"Instance {id+1} is a duplicate, trying again...")
                continue
            instances.add(instance_str)
            with open(instance_path, "w") as f:
                f.write(instance_str)
            tries += 1
            plan = compute_optimal_plan(domain_path, instance_path)
            if not plan:
                print(f"Failed to compute plan for instance {id+1}, trying again...")
                continue
            plan_length = len(plan.split("\n"))
            if plan_length < max_plan_length:
                print(f"Generated instance {id+1} with plan length {plan_length}.")
                break
            else:
                print(f"Generated instance {id+1} with plan length {plan_length}, which is too long. Trying again...")
        else:
            print(f"Failed to generate a valid instance {id+1} after {max_retries} tries.")
            return


def generate_logistics_instance(num_airplanes: int = 1, num_cities: int = 3, city_size: int = 2, num_packages: int = 1) -> str:
    """
    Generate a logistics planning problem instance and return the output as a string.
    The parameters define the number of airplanes, cities, size of each city, and number of packages.
    The output is a pddl string representation of the generated instance.
    """
    seed = random.randint(0, 1000000)
    cmd = f"./test_data/PDDL/logistics/logistics -a {num_airplanes} -c {num_cities} -s {city_size} -p {num_packages} -r {seed}"
    output_data = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    return output_data.stdout
