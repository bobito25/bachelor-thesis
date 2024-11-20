import itertools
from dataclasses import dataclass
from typing import List, Tuple, Generator
from collections import Counter
from copy import deepcopy
from ete3 import Tree, TextFace

@dataclass
class State:
    numbers: List[float | int]
    successors: List['State']
    prev: 'State' = None
    operation: Tuple[float, str, float] = None
    novelty: int = None

def brute_force_tree(input: List[float]) -> State:
    print("Solving for: ", input)
    init_state = State(numbers=sorted(input), successors=[])
    cur_states = [init_state]
    for i in range(3):
        next_states = []
        for cur_state in cur_states:
            for a, b in itertools.combinations(cur_state.numbers, 2):
                new_numbers = []
                op_tuples = []

                new_numbers.append(a + b)
                op_tuples.append((a, '+', b))
                new_numbers.append(a - b)
                op_tuples.append((a, '-', b))
                if b != a:
                    new_numbers.append(b - a)
                    op_tuples.append((b, '-', a))
                new_numbers.append(a * b)
                op_tuples.append((a, '*', b))
                if b != 0:
                    new_numbers.append(a / b)
                    op_tuples.append((a, '/', b))
                if a != 0:
                    new_numbers.append(b / a)
                    op_tuples.append((b, '/', a))

                kept_numbers = cur_state.numbers.copy()
                kept_numbers.remove(a)
                kept_numbers.remove(b)

                for num, op_tup in zip(new_numbers, op_tuples):
                    new_state = State(
                        numbers=sorted(kept_numbers + [num]),
                        successors=[],
                        prev=cur_state,
                        operation=op_tup)
                    cur_state.successors.append(new_state)
                    next_states.append(new_state)
        cur_states = next_states
        assert all([len(state.numbers) == 3-i for state in cur_states])
    return init_state, cur_states

tree, result_states = brute_force_tree([1, 2, 3, 4])
print("Example path:")
print(tree.numbers)
print(tree.successors[0].numbers)
print(tree.successors[0].successors[0].numbers)
print(tree.successors[0].successors[0].successors[0].numbers)

def dfs_walk(state: State, leaf_nodes: bool = True) -> Generator[Tuple[float, str, float], None, None]:
    if leaf_nodes or state.successors:
        yield state
    for s in state.successors:
        yield from dfs_walk(s, leaf_nodes)

def bfs_walk(state: State, leaf_nodes: bool = True) -> Generator[Tuple[float, str, float], None, None]:
    queue = [state]
    while queue:
        s = queue.pop(0)
        if not leaf_nodes and not s.successors:
            return
        yield s
        queue.extend(s.successors)

def get_num_nodes (state: State) -> int:
    return 1 + sum([get_num_nodes(s) for s in state.successors])

assert get_num_nodes(tree) == len(list(dfs_walk(tree))) == len(list(bfs_walk(tree)))

total_nodes_num = get_num_nodes(tree)
print("Total nodes:", total_nodes_num)

inner_nodes_num = len(list(bfs_walk(tree, leaf_nodes=False)))
print("Inner nodes:", inner_nodes_num)

leaf_nodes_num = len(result_states)
print("Leaf nodes:", leaf_nodes_num)

assert total_nodes_num == inner_nodes_num + leaf_nodes_num
assert inner_nodes_num == total_nodes_num - leaf_nodes_num
assert inner_nodes_num == len(list(dfs_walk(tree, leaf_nodes=False)))

def get_max_depth(state: State) -> int:
    if not state.successors:
        return 0
    return 1 + max([get_max_depth(s) for s in state.successors])

assert get_max_depth(tree) == 3

def round_tree(state: State) -> State:
    for s in bfs_walk(state):
        s.numbers = [round(num, 2) for num in s.numbers]
    return state

tree = round_tree(tree)

def remove_zeros_after_decimal(state: State) -> State:
    for s in bfs_walk(state):
        s.numbers = [int(num) if isinstance(num, int) or num.is_integer() else num for num in s.numbers]
    return state

tree = remove_zeros_after_decimal(tree)

print("Number correct:", sum([24 in state.numbers for state in result_states]))

def get_nodes_at_depth(state: State, depth: int) -> List[State]:
    if depth == 0:
        return [state]
    nodes = []
    for s in state.successors:
        nodes.extend(get_nodes_at_depth(s, depth-1))
    return nodes

assert len(get_nodes_at_depth(tree, 0)) == 1
assert len(get_nodes_at_depth(tree, 3)) == len(result_states)

def get_num_duplicate_states(root_state: State) -> int:
    # novelty pruning
    duplicate_states = 0
    duplicate_inner_states = 0
    prev_states = []
    for s in bfs_walk(root_state):
        if s.numbers in prev_states:
            duplicate_states += 1
            if len(s.successors) > 0:
                duplicate_inner_states += 1
        prev_states.append(s.numbers)
    return duplicate_states, duplicate_inner_states

num_duplicate_states, num_duplicate_inner_states = get_num_duplicate_states(tree)
print("Duplicate states:", num_duplicate_states)
print("Duplicate inner states:", num_duplicate_inner_states)

def prune_duplicate(root_state: State) -> State:
    new_tree = deepcopy(root_state)
    prev_states = []
    cur_states = [new_tree]
    while cur_states:
        s = cur_states.pop(0)
        if s.numbers in prev_states:
            s.successors = []
            s.prev.successors.remove(s)
        else:
            prev_states.append(s.numbers)
            cur_states.extend(s.successors)
    return new_tree

pruned_tree_duplicate = prune_duplicate(tree)

assert get_num_duplicate_states(pruned_tree_duplicate)[0] == 0

print("Pruned tree duplicate:")
print("Number of nodes", get_num_nodes(pruned_tree_duplicate))
print("Number of leaf nodes", len(get_nodes_at_depth(pruned_tree_duplicate, 3)))
print("Number of correct leaf nodes", len([s for s in get_nodes_at_depth(pruned_tree_duplicate, 3) if 24 in s.numbers]))

assert all(s.numbers == sorted(s.numbers) for s in dfs_walk(tree))

MAX_WIDTH = 4
def calc_novelty(numbers: List[str], prev_states: List[List[str]]) -> int:
    for width in range(1, MAX_WIDTH):
        prev_cs = [sorted(c) for prev_s in prev_states for c in itertools.combinations(prev_s, width)]
        for c in itertools.combinations(numbers, width):
            if sorted(c) not in prev_cs:
                return width
    return 0

def calc_all_novelties(state: State) -> State:
    prev_states = []
    for s in bfs_walk(state):
        new_state = list(map(str,s.numbers)) + ['l'+str(len(s.numbers))]
        s.novelty = calc_novelty(new_state, prev_states)
        if new_state not in prev_states:
            prev_states.append(new_state)
    return state

tree = calc_all_novelties(tree)

novelty_counts = Counter([s.novelty for s in bfs_walk(tree)])

print("Novelty counts:", novelty_counts)

novelty_counts_inner = Counter([s.novelty for s in bfs_walk(tree, leaf_nodes=False)])

print("Novelty counts inner:", novelty_counts_inner)

novelty_counts_leaf = Counter([s.novelty for s in result_states])

print("Novelty counts leaf:", novelty_counts_leaf)

novelty_counts_correct = Counter([s.novelty for s in result_states if 24 in s.numbers])

print("Novelty counts correct:", novelty_counts_correct)

def prune_novelty(root_state: State, min_novelty: int, max_novelty: int) -> State:
    # prune novelty using state numbers and length of state as atoms
    new_tree = deepcopy(root_state)
    prev_states = []
    cur_states = [new_tree]
    while cur_states:
        s = cur_states.pop(0)
        state = list(map(str,s.numbers)) + ['l'+str(len(s.numbers))]
        s.novelty = calc_novelty(state, prev_states)
        if min_novelty <= s.novelty <= max_novelty:
            prev_states.append(state)
            cur_states.extend(s.successors)
        else:
            # prune
            s.successors = []
            s.prev.successors.remove(s)
            cur = s.prev
            # prune dead branches
            while cur and not cur.successors:
                prev_states.remove(list(map(str,cur.numbers)) + ['l'+str(len(cur.numbers))])
                cur.successors = []
                cur.prev.successors.remove(cur)
                cur = cur.prev
    return new_tree

for width in range(1, MAX_WIDTH):
    pruned_tree = prune_novelty(tree, 1, width)
    assert get_max_depth(pruned_tree) == 3
    assert get_num_duplicate_states(pruned_tree)[0] == 0
    print("Pruned tree width", width, ":")
    print("Number of nodes", get_num_nodes(pruned_tree))
    print("Number of leaf nodes", len(get_nodes_at_depth(pruned_tree, 3)))
    print("Number of correct leaf nodes", len([s for s in get_nodes_at_depth(pruned_tree, 3) if 24 in s.numbers]))

def to_ete3_tree(state: State) -> str:
    t = Tree()
    t.add_face(TextFace(" ".join(map(str, state.numbers))), column=0)
    for s in state.successors:
        t.add_child(to_ete3_tree(s))
    return t

p_tree = prune_novelty(tree, 1, 1)
tree_ete3 = to_ete3_tree(p_tree)


tree_ete3.show()
