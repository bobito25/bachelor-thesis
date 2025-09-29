from copy import deepcopy
from itertools import combinations
import random

from lifted_pddl import Parser

from plan_gen import check_if_goal_state_with_parser, get_applicable_actions_with_parser


def calc_width(domain_path: str, problem_path: str, max_width: int = 10) -> int:
    """
    Calculate the effective width of problem in the given domain
    The width of a problem is defined as the minimum novelty needed to solve the problem.
    This is found using the iterative width algorithm.
    The algorithm searches non-heuristically in state-space (breadth-first)
    and prunes states according to novelty of state in comparison to proviously generated states.
    """
    for width in range(1, max_width):
        # start breadth-first search for current width
        parser = Parser()
        parser.parse_domain(domain_path)
        parser.parse_problem(problem_path)

        previous_states: set[frozenset] = set()
        state_queue: list[set] = []

        # Add initial state to queue
        initial_state = parser.atoms
        state_queue.append(initial_state)

        while state_queue:
            cur_state: set = state_queue.pop(0)
            parser.set_current_state(cur_state)

            novelty = calc_novelty(cur_state, previous_states)

            if novelty > width or novelty == -1:
                # prune state
                continue
            
            # Add state to previous states
            previous_states.add(frozenset(cur_state))

            # Check if state is goal state
            if check_if_goal_state_with_parser(parser):
                return width

            # Generate next states
            applicable_actions = get_applicable_actions_with_parser(parser)
            for action, params in applicable_actions:
                # Get next state
                next_state = parser.get_next_state(action, params)
                # Check if state is already in queue
                state_queue.append(next_state)
        
    # If no goal state is found with any width, return -1
    return -1


def calc_novelty(state: set, previous_states: set[frozenset]) -> int:
    """
    Calculate novelty of state in comparison to previous states.
    Novelty is defined as the size of the smallest subset of atoms of the current state that is not present in any of the previous states.
    """
    for novelty in range(1, len(state) + 1):
        for subset in combinations(state, novelty):
            if not any(set(subset).issubset(prev_state) for prev_state in previous_states):
                return novelty
    return -1


def partial_novelty_breadth_first_search(parser: Parser, branching_factor: int, length: int, width: int) -> tuple[set[frozenset[tuple[str, tuple]]], set[tuple[str, tuple]]]:
    """
    Do a partial breadth-first search  with a given branching factor up to a given length of generated states.
    Prune non novel states according to the given width.
    Return all generated non-pruned states and the last state which can be novel or non-novel.
    """
    state_queue: list[set] = []

    # Add initial state to queue
    initial_state = parser.atoms
    state_queue.append(initial_state)

    # Store all generated states
    generated_states: set[frozenset] = set()
    # save copy before last addition for returning
    last_generated_states: set[frozenset] = set()

    while state_queue and len(generated_states) < length:
        cur_state: set = state_queue.pop(0)
        parser.set_current_state(cur_state)

        novelty = calc_novelty(cur_state, generated_states)

        last_generated_states = deepcopy(generated_states)

        if novelty > width or novelty == -1:
            # prune state
            # but we also want to return non novel states
            if len(generated_states) == length - 1:
                # return non novel state that would have been pruned
                break
            continue

        # Add state to previous states
        generated_states.add(frozenset(cur_state))

        # Generate next states
        applicable_actions = get_applicable_actions_with_parser(parser)

        if len(applicable_actions) > branching_factor:
            # Limit the number of applicable actions to the branching factor
            applicable_actions = applicable_actions[:branching_factor]

        for action, params in applicable_actions:
            # Get next state
            next_state = parser.get_next_state(action, params)
            # Check if state is already in queue
            state_queue.append(next_state)

    return last_generated_states, cur_state


def find_non_duplicate_pruned_state(parser: Parser, branching_factor: int, max_length: int, instance_width: int) -> tuple[set[frozenset[tuple[str, tuple]]], set[tuple[str, tuple]]]:
    """
    Generate a state that is pruned by the given width using a partial breadth-first search.
    Return all previous states and the new state that would have been pruned.
    """
    state_queue: list[set] = []

    # Add initial state to queue
    initial_state = parser.atoms
    state_queue.append(initial_state)

    # Store all generated states
    generated_states: set[frozenset] = set()
    # save copy before last addition for returning
    last_generated_states: set[frozenset] = set()

    pruned_non_duplicated_states = []
    last_generated_states_for_pruned = []

    while state_queue and len(generated_states) < max_length:
        cur_state: set = state_queue.pop(0)
        parser.set_current_state(cur_state)

        novelty = calc_novelty(cur_state, generated_states)

        last_generated_states = deepcopy(generated_states)

        if novelty == -1:
            # state is duplicate
            continue

        if novelty > instance_width:
            # prune state
            pruned_non_duplicated_states.append(cur_state)
            last_generated_states_for_pruned.append(deepcopy(last_generated_states))
            continue

        # Add state to previous states
        generated_states.add(frozenset(cur_state))

        # Generate next states
        applicable_actions = get_applicable_actions_with_parser(parser)

        if len(applicable_actions) > branching_factor:
            # Limit the number of applicable actions to the branching factor
            applicable_actions = applicable_actions[:branching_factor]

        for action, params in applicable_actions:
            # Get next state
            next_state = parser.get_next_state(action, params)
            # Check if state is already in queue
            state_queue.append(next_state)

    if not pruned_non_duplicated_states:
        raise ValueError(f"Could not generate a pruned non-duplicate state for max_length = {max_length}.")

    pruned_non_duplicated_state = pruned_non_duplicated_states[0]
    last_generated_states = last_generated_states_for_pruned[0]

    return last_generated_states, pruned_non_duplicated_state

