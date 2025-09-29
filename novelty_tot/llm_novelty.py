import logging
import random
import re

from llm_engine import Engine as LLMEngine
from llm_compare import is_yes

llm_novelty_logger = logging.getLogger("nov_tot." + __name__)

async def summarize_novelty(engine: LLMEngine, previous_content: str, new_state: str) -> str:
    """
    Summarize the novelty of the state.
    """
    prompt = f"Concisely summarize the novelty of the new step compared to previous steps:\n\nPrevious contents:\n{previous_content}\n\nNew step: {new_state}"
    response = await engine.query(prompt)
    return response

def novelty_prompt(new_state: str, previous_states: list[str], examples: bool = False) -> str:
    """
    Create a prompt for the LLM to check if the new state is novel compared to previous states.
    """
    previous_states_str = "\n###\n".join(previous_states)
    max_size = 48000
    if len(previous_states_str) > max_size:
        llm_novelty_logger.warning("Previous states string is too long, truncating..")
        # truncate by replacing the middle with "[...]"
        previous_states_str = previous_states_str[:max_size//2] + " [...] " + previous_states_str[-(max_size//2):]

    query = f"Is -START-\n'{new_state}'\n-END- a novel state compared to this list of states -START-\n'{previous_states_str}'\n-END-? Respond with 'yes' or 'no'. Do not add any other text."
    if examples:
        examples_str = "Your job is to estimate the novelty of a new step compared to previous steps." \
                       "Here are some examples:\n" \
                       "-START-\nDrive truck from London to Paris\n-END- is novel compared to -START-\nDrive truck from Paris to London\n-END-: yes\n" \
                       "-START-\nLoad red package onto truck in Paris\n-END- is novel compared to -START-\nDrive truck from Berlin to Paris-END-: yes\n"
        query = examples_str + "\n" + query
    return query

async def is_novel(query: callable, new_state: str, previous_states: list[str]) -> bool:
    """
    Check if the new state is novel compared to the previous states using an LLM.
    """
    prompt = novelty_prompt(new_state, previous_states)
    response = await query(prompt)
    llm_novelty_logger.debug(f"Novelty check query: {prompt}")
    llm_novelty_logger.debug(f"Novelty check response: {response}")
    return is_yes(response)

async def compare_novelty_summaries(engine: LLMEngine, new_state_novelty: str, previous_states_novelty: list[str]) -> str:
    """
    Compare the novelty of the new state with the previous states using an LLM.
    """
    previous_states_novelty_str = "\n-END-\n-START-\n".join(previous_states_novelty)
    query = f"Compare the novelty of the new step -START-\n'{new_state_novelty}'\n-END- with the novelty summaries of previous steps: -START-\n'{previous_states_novelty_str}'\n-END- and decide whether the new step is novel in comparison to the other steps. Respond with 'yes' or 'no'. Do not add any other text."
    response = await engine.query(query)
    return is_yes(response)


async def estimate_novelty(engine: LLMEngine, new_state: str, previous_states: list[str], max_size: int = 48000, reasoning: bool = True, examples: bool = False) -> tuple[str, int]:
    """
    Estimate the novelty of the new state compared to previous states using an LLM.
    """
    if examples:
        raise NotImplementedError("Examples not implemented for estimate_novelty_prompt")

    instructions = """You are an expert in reasoning and search algorithms.
Your task is to decide whether a new state is novel given a set of previously encountered states, according to the following definition:
A state's novelty is the size of the smallest tuple of features (facts, attributes, or properties) that are present in the state but have never appeared together in any previous state.
If all features of the new state have appeared together in at least one previous state, the novelty is -1.

Instructions

You will be given:
A list of previous states (each represented as a set of features/attributes).
A new candidate state (also a set of features/attributes).

Compare the new state with all previous states.
Identify the smallest tuple of features in the new state that has never occurred in any previous state (or determine there is no such tuple).
Determine the novelty value of the new state."""

    previous_states_str = "\n###\n".join(previous_states)
    if len(previous_states_str) > max_size:
        llm_novelty_logger.warning("Previous states string is too long, shortening..")
        engine.num_truncated_queries += 1
    max_shortenings = 10000  # prevent infinite loop
    for _ in range(max_shortenings):
        if len(previous_states_str) <= max_size:
            break
        # remove a random previous state
        previous_states.pop(random.randint(0, len(previous_states) - 1))
        previous_states_str = "\n###\n".join(previous_states)

    if not previous_states:
        previous_states_str = "None"

    if reasoning:
        query_prompt = "Give your reasoning and then '[OUTPUT]' followed by the novelty value as an integer. Do not add any other text."
    else:
        query_prompt = "Output just the novelty value as an integer. Do not add any other text.\n\n"

    prompt = instructions + "\n\nPrevious States:\n" + previous_states_str + "\n\nNew State:\n" + new_state + "\n\n" + query_prompt

    response = await engine.query(prompt)

    if reasoning:
        extracted_response = response.split("[OUTPUT]")[-1].strip() 
    else:
        extracted_response = response.strip()

    if engine.thinking:
        response = f"<think>{response.thinking}</think>" + response

    m = re.search(r'-?\d+', extracted_response)
    extracted_novelty = int(m.group(0)) if m else 0

    return response, extracted_novelty


async def estimate_problem_width(engine: LLMEngine, context: str, query_state: str, goal_atoms: str, reasoning: bool = True) -> tuple[str, int]:
    """
    Estimate the width of the problem instance using an LLM.
    """
    instructions = """You are an expert in reasoning and search algorithms.

You are given:
Environment / rules: [describe the dynamics, constraints, or how the world changes]
Initial state: [describe the starting situation]
Goal conditions: [list the desired target conditions or atoms]

Task: Estimate the problem width.
The width is the smallest number w of distinct conditions, variables, or features that must be considered together in order to make systematic progress toward the goal.
Width ≈ 1 means each goal or subgoal can be reached by tracking single conditions independently.
Width ≈ 2 means pairs of conditions must be tracked jointly (dependencies between two facts matter).
Width ≈ 3 means triples matter, and so on.

When reasoning, follow these steps:
Identify the key variables, features, or atoms that describe the problem.
Analyze the dependencies: can goals be achieved by considering features one at a time, or do combinations matter?
Estimate the minimal number of features that must be tracked jointly to guarantee progress toward the goal.
"""
    prompt = instructions + "\n\nProblem Environment Description:\n" + context + "\n\nQuery State:\n" + query_state + "\n\nGoal Conditions:\n" + goal_atoms

    if reasoning:
        prompt += "\nOutput your reasoning and then '[OUTPUT]' followed by the estimated width as an integer. Do not add any other text."
    else:
        prompt += "\nOutput the estimated width as an integer. Do not add any other text."

    response = await engine.query(prompt)

    if reasoning:
        extracted_response = response.split("[OUTPUT]")[-1].strip()
    else:
        extracted_response = response.strip()

    if engine.thinking:
        response = f"<think>{response.thinking}</think>" + response

    m = re.search(r'-?\d+', extracted_response)
    extracted_width = int(m.group(0)) if m else 0

    return response, extracted_width
