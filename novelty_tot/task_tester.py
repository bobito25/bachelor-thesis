from pydantic import BaseModel
from enum import Enum
from lifted_pddl import Parser
import os
import json
import asyncio
import random

from llm_engine import Engine
from llm_compare import check_answer_for_question_simple, extract_answer_for_question, check_answer_for_question_action_simple
from non_llm_compare import check_action_one_of, check_action_equals
from plan_gen import generate_non_goal_plan, compute_optimal_plan, plan_to_pddl, get_applicable_actions_with_parser
from plan_width import calc_novelty, calc_width, partial_novelty_breadth_first_search, find_non_duplicate_pruned_state
from llm_novelty import is_novel, estimate_novelty, estimate_problem_width
from pddl_translation.translator import Translator


class TaskType(Enum):
    ACTION_GEN = 1
    ACTION_GEN_SINGLE = 6
    SUCC_STATE_MAPPING = 2
    SUCC_STATE_MAPPING_SEPERATE = 5
    GOAL_STATE_VERIFICATION = 3
    NOVELTY = 4
    NOVELTY_NO_DUPLICATES = 7
    NOVELTY_ESTIMATION = 8
    NOVELTY_ESTIMATION_NO_DUPLICATES = 9
    PROBLEM_WIDTH_ESTIMATION = 10


class SuccActionStateTask(BaseModel):
    id: str
    query_state: str
    poss_next_actions: list[str]
    corr_succ_states: dict[str, str]
    optimal_action: str
    goal_atoms: str


class GoalStateTask(BaseModel):
    id: str
    start_state: str
    query_plan: str
    goal_atoms: str
    is_goal_state: bool


class NoveltyTask(BaseModel):
    id: str
    query_state: str
    previous_states: list[str]
    novelty: int
    instance_width: int
    goal_atoms: str


class Result(BaseModel):
    task_id: str
    success: bool
    response: str
    extracted_answer: str


def get_succ_action_state_tasks(prompt_translator: Translator) -> list[SuccActionStateTask]:
    """Get the tasks for the successor action state mapping task."""
    # Check if the tasks are already cached
    cache_dir = "test_data/" + prompt_translator.type
    cache_file = "succ_action_state_tasks.json"
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            SuccActionStateTask(
                id=task["id"],
                query_state=task["query_state"],
                poss_next_actions=task["poss_next_actions"],
                corr_succ_states=task["corr_succ_states"],
                optimal_action=task["optimal_action"],
                goal_atoms=task["goal_atoms"]
            )
            for task in data
        ]

    tasks: list[SuccActionStateTask] = []
    instances_dir = 'test_data/PDDL/Blocksworld/instances'
    domain_path = 'test_data/PDDL/Blocksworld/domain.pddl'
    for fname in os.listdir(instances_dir):
        if not fname.endswith('.pddl'):
            continue
        parser = Parser()
        parser.parse_domain(domain_path)
        parser.parse_problem(os.path.join(instances_dir, fname))
        query_state = prompt_translator.state_to_string(parser.atoms)
        poss_next_actions = []
        corr_succ_states = {}

        applicable_actions = get_applicable_actions_with_parser(parser)
        for action_name, action_params in applicable_actions:
            applicable_action_str = prompt_translator.action_to_string((action_name, action_params))
            poss_next_actions.append(applicable_action_str)
            next_state = parser.get_next_state(action_name, action_params)
            next_state_str = prompt_translator.state_to_string(next_state)
            corr_succ_states[applicable_action_str] = next_state_str

        # get optimal next action by computing the optimal plan
        problem_path = os.path.join(instances_dir, fname)
        raw_plan = compute_optimal_plan(domain_path, problem_path)
        if raw_plan:
            plan_str = prompt_translator.plan_to_string(raw_plan.strip().splitlines())
            optimal_action = plan_str.split(", ")[0]
            if optimal_action not in poss_next_actions:
                raise ValueError(f"Optimal action {optimal_action} not found in possible next actions for {fname}.")
        else:
            raise ValueError(f"Error computing plan for {fname}.")
        
        # convert goal atoms to string
        goal_atoms = prompt_translator.goal_atoms_to_string(parser.goals)

        tasks.append(
            SuccActionStateTask(
                id=fname,
                query_state=query_state,
                poss_next_actions=poss_next_actions,
                corr_succ_states=corr_succ_states,
                optimal_action=optimal_action,
                goal_atoms=goal_atoms
            )
        )

    # Save to cache
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([task.model_dump() for task in tasks], f, indent=2, ensure_ascii=False)

    return tasks


def get_goal_state_tasks(prompt_translator: Translator) -> list[GoalStateTask]:
    """Get the tasks for the goal state verification task."""
    # Check if the tasks are already cached
    cache_dir = "test_data/" + prompt_translator.type
    cache_file = "goal_state_tasks.json"
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            GoalStateTask(
                id=task["id"],
                start_state=task["start_state"],
                query_plan=task["query_plan"],
                goal_atoms=task["goal_atoms"],
                is_goal_state=task["is_goal_state"]
            )
            for task in data
        ]

    tasks: list[GoalStateTask] = []
    domain_path = 'test_data/PDDL/Blocksworld/domain.pddl'
    instances_dir = 'test_data/PDDL/Blocksworld/instances'
    for idx, fname in enumerate(os.listdir(instances_dir)):
        if not fname.endswith('.pddl'):
            continue
        parser = Parser()
        parser.parse_domain(domain_path)
        problem_path = os.path.join(instances_dir, fname)
        parser.parse_problem(problem_path)

        if idx % 2 == 0:
            # goal plan via Fast‑Downward
            raw_plan = compute_optimal_plan(domain_path, problem_path)
            is_goal = True
        else:
            # non‑goal plan
            plan = generate_non_goal_plan(5, domain_path, problem_path)
            raw_plan = plan_to_pddl(domain_path, problem_path, plan)
            is_goal = False

        plan_actions = raw_plan.strip().splitlines() if raw_plan else []

        query_plan = prompt_translator.plan_to_string(plan_actions)

        start_state = prompt_translator.state_to_string(parser.atoms)
        goal_atoms = prompt_translator.goal_atoms_to_string(parser.goals)
        tasks.append(
            GoalStateTask(
                id=fname,
                start_state=start_state,
                query_plan=query_plan,
                goal_atoms=goal_atoms,
                is_goal_state=is_goal
            )
        )

    # Save to cache
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([task.model_dump() for task in tasks], f, indent=2, ensure_ascii=False)

    return tasks


def get_novelty_tasks(prompt_translator: Translator, verbose: bool = False) -> list[NoveltyTask]:
    """Get the tasks for the novelty task."""
    if not verbose:
        print = lambda *args, **kwargs: None  # disable print
    # Check if the tasks are already cached
    cache_dir = "test_data/" + prompt_translator.type
    cache_file = "novelty_tasks.json"
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks = [
            NoveltyTask(
                id=task["id"],
                query_state=task["query_state"],
                previous_states=task["previous_states"],
                novelty=task["novelty"],
                instance_width=task["instance_width"],
                goal_atoms=task["goal_atoms"]
            )
            for task in data
        ]
        print(f"Found {len(tasks)} cached novelty tasks.")
        # average number of previous states
        average_previous_states_size = sum(len(task.previous_states) for task in tasks) / len(tasks) if tasks else 0
        print(f"Average number of previous states: {average_previous_states_size}")
        return tasks

    tasks: list[NoveltyTask] = []
    domain_path = 'test_data/PDDL/Blocksworld/domain.pddl'
    instances_dir = 'test_data/PDDL/Blocksworld/instances'
    for fname in os.listdir(instances_dir):
        if not fname.endswith('.pddl'):
            continue
        parser = Parser()
        parser.parse_domain(domain_path)
        problem_path = os.path.join(instances_dir, fname)
        parser.parse_problem(problem_path)
        goal_atoms = prompt_translator.goal_atoms_to_string(parser.goals)
        instance_width = calc_width(domain_path, problem_path)
        if instance_width == -1:
            raise ValueError(f"Instance {fname} has no solution with given max width.")

        branching_factor = 2
        length = random.randint(3, 15)
        previous_states, query_state = partial_novelty_breadth_first_search(parser, branching_factor=branching_factor, length=length, width=instance_width)

        # calculate novelty
        novelty = calc_novelty(query_state, previous_states)

        # convert to strings
        query_state = prompt_translator.state_to_string(query_state)
        previous_states_strs = [prompt_translator.state_to_string(state) for state in previous_states]

        tasks.append(
            NoveltyTask(
                id=fname,
                query_state=query_state,
                previous_states=previous_states_strs,
                novelty=novelty,
                instance_width=instance_width,
                goal_atoms=goal_atoms
            )
        )
    # Save to cache
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([task.model_dump() for task in tasks], f, indent=2, ensure_ascii=False)
    return tasks


def get_novelty_tasks_no_duplicates(prompt_translator: Translator, max_num: int = 50, verbose: bool = False) -> list[NoveltyTask]:
    """Get the tasks for the novelty task, ensuring that the new state is not a duplicate of a previous state."""
    if not verbose:
        print = lambda *args, **kwargs: None  # disable print
    # Check if the tasks are already cached
    cache_dir = "test_data/" + prompt_translator.type
    cache_file = "novelty_tasks_no_duplicates.json"
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks = [
            NoveltyTask(
                id=task["id"],
                query_state=task["query_state"],
                previous_states=task["previous_states"],
                novelty=task["novelty"],
                instance_width=task["instance_width"],
                goal_atoms=task["goal_atoms"]
            )
            for task in data
        ]
        print(f"Found {len(tasks)} cached novelty tasks with no duplicates.")
        # average number of previous states
        average_previous_states_size = sum(len(task.previous_states) for task in tasks) / len(tasks) if tasks else 0
        print(f"Average number of previous states: {average_previous_states_size}")
        return tasks

    tasks: list[NoveltyTask] = []
    domain_path = 'test_data/PDDL/Blocksworld/domain.pddl'
    instances_dir = 'test_data/PDDL/Blocksworld/more_instances'
    for fname in os.listdir(instances_dir):
        if not fname.endswith('.pddl'):
            continue
        parser = Parser()
        parser.parse_domain(domain_path)
        problem_path = os.path.join(instances_dir, fname)
        parser.parse_problem(problem_path)
        goal_atoms = prompt_translator.goal_atoms_to_string(parser.goals)
        instance_width = calc_width(domain_path, problem_path)
        if instance_width == -1:
            raise ValueError(f"Instance {fname} has no solution with given max width.")

        branching_factor = 100
        max_length = random.randint(5, 15)

        max_extension = 20

        for _ in range(max_extension):
            try:
                previous_states, query_state = find_non_duplicate_pruned_state(parser, branching_factor=branching_factor, max_length=max_length, instance_width=instance_width)

                # calculate novelty
                novelty = calc_novelty(query_state, previous_states)
            except ValueError:
                max_length += 1
                continue
            break
        else:
            # skip
            continue
            # raise ValueError(f"Could not generate a non-duplicate pruned state for instance {fname}.")

        # convert to strings
        query_state = prompt_translator.state_to_string(query_state)
        previous_states_strs = [prompt_translator.state_to_string(state) for state in previous_states]

        tasks.append(
            NoveltyTask(
                id=fname,
                query_state=query_state,
                previous_states=previous_states_strs,
                novelty=novelty,
                instance_width=instance_width,
                goal_atoms=goal_atoms
            )
        )
    print(f"Generated {len(tasks)} novelty tasks with no duplicates.")
    average_previous_states_size = sum(len(task.previous_states) for task in tasks) / len(tasks) if tasks else 0
    print(f"Average number of previous states: {average_previous_states_size}")
    # get the max_num states with the least previous states
    tasks = sorted(tasks, key=lambda task: len(task.previous_states))[:max_num]
    print(f"Using {len(tasks)} tasks with the least previous states.")
    average_previous_states_size = sum(len(task.previous_states) for task in tasks) / len(tasks) if tasks else 0
    print(f"Average number of previous states after filtering: {average_previous_states_size}")
    # Save to cache
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([task.model_dump() for task in tasks], f, indent=2, ensure_ascii=False)
    return tasks


class TaskRunner:
    """Tests a given ToT configuration by running different tasks"""

    def __init__(self, llm_engine: Engine, prompt_translator: Translator, context: str = "", seed: int = 42) -> None:
        random.seed(seed)
        self.succ_action_state_tasks = get_succ_action_state_tasks(prompt_translator)
        self.goal_state_tasks = get_goal_state_tasks(prompt_translator)
        self.novelty_tasks = get_novelty_tasks(prompt_translator)
        self.novelty_tasks_no_duplicates = get_novelty_tasks_no_duplicates(prompt_translator)
        self.llm_engine = llm_engine
        self.context = context
        self.prompt_translator = prompt_translator

    async def run(self, task: TaskType, *args, **kwargs) -> list[Result] | list[list[Result]]:
        if task is TaskType.ACTION_GEN:
            return await self._test_action_gen(*args, **kwargs)
        elif task is TaskType.ACTION_GEN_SINGLE:
            return await self._test_action_gen_single(*args, **kwargs)
        elif task is TaskType.SUCC_STATE_MAPPING:
            return await self._test_succ_state_mapping(*args, **kwargs)
        elif task is TaskType.SUCC_STATE_MAPPING_SEPERATE:
            return await self._test_succ_state_mapping_seperate(*args, **kwargs)
        elif task is TaskType.GOAL_STATE_VERIFICATION:
            return await self._test_goal_state_verification(*args, **kwargs)
        elif task is TaskType.NOVELTY:
            return await self._test_novelty(*args, **kwargs)
        elif task is TaskType.NOVELTY_NO_DUPLICATES:
            return await self._test_novelty_no_duplicates(*args, **kwargs)
        elif task is TaskType.NOVELTY_ESTIMATION:
            return await self._test_novelty_estimation(*args, **kwargs)
        elif task is TaskType.NOVELTY_ESTIMATION_NO_DUPLICATES:
            return await self._test_novelty_estimation_no_duplicates(*args, **kwargs)
        elif task is TaskType.PROBLEM_WIDTH_ESTIMATION:
            return await self._test_problem_width_estimation(*args, **kwargs)
        else:
            raise ValueError(f"Invalid task: {task}. Valid tasks are: {', '.join([t.name for t in TaskType])}")

    async def _run_with_error_handling(self, run_single: callable, task) -> Result:
        try:
            return await run_single(task)
        except Exception as e:
            print(f"Error occurred while running task {task.id}:\n{e}")
            return Result(
                task_id=task.id,
                success=False,
                response=str(e),
                extracted_answer="N/A"
            )

    async def _test_action_gen(self, num_tasks: int = 1) -> list[Result]:
        prompt = "Output exactly a list of possible actions for the given state. Be concise."
        async def run_single(task: SuccActionStateTask):
            query = f"{self.context}\n{prompt}\nState: {task.query_state}"
            response = await self.llm_engine.query(query)
            extracted_answer = await extract_answer_for_question(self.llm_engine.query, response, query, thinking_override=False)
            if response.thinking:
                response += f"<think>{response.thinking}</think>"
            ground_truth = ", ".join(task.poss_next_actions)
            success = await check_answer_for_question_action_simple(self.llm_engine.query, extracted_answer, ground_truth, query, thinking_override=True)
            return Result(
                task_id=task.id,
                success=success,
                response=response,
                extracted_answer=extracted_answer
            )
        tasks = [self._run_with_error_handling(run_single, task) for task in self.succ_action_state_tasks[:num_tasks]]
        return await asyncio.gather(*tasks)
    
    async def _test_action_gen_single(self, num_tasks: int = 1) -> list[Result]:
        """Test whether the LLM can generate a single possible actions for a given state.
        Also check whether it is optimal."""
        prompt = "Output exactly the next action for solving the problem. Be concise and specific."
        async def run_single(task: SuccActionStateTask):
            query = f"{self.context}\n{prompt}\nState: {task.query_state}\nGoal: {task.goal_atoms}"
            response = await self.llm_engine.query(query)
            extracted_answer = await extract_answer_for_question(self.llm_engine.query, response, query, thinking_override=False)
            if response.thinking:
                response += f"<think>{response.thinking}</think>"
            success_valid = check_action_one_of(extracted_answer, task.poss_next_actions, translator=self.prompt_translator)
            result_valid = Result(
                task_id=task.id,
                success=success_valid,
                response=response,
                extracted_answer=extracted_answer
            )
            success_optimal = check_action_equals(extracted_answer, task.optimal_action, translator=self.prompt_translator)
            result_optimal = Result(
                task_id=task.id + "_optimal",
                success=success_optimal,
                response=response,
                extracted_answer=extracted_answer
            )
            return [result_valid, result_optimal]
        tasks = [self._run_with_error_handling(run_single, task) for task in self.succ_action_state_tasks[:num_tasks]]
        return await asyncio.gather(*tasks)

    async def _test_succ_state_mapping(self, num_tasks: int = 1) -> list[Result]:
        """Test whether the LLM can generate a mapping of possible actions to their corresponding successor states."""
        prompt = "Output exactly a mapping of possible actions to their corresponding successor states. Be concise."
        async def run_single(task: SuccActionStateTask):
            query = f"{self.context}\n{prompt}\nState: {task.query_state}\nActions: {', '.join(task.poss_next_actions)}"
            response = await self.llm_engine.query(query)
            extracted_answer = await extract_answer_for_question(self.llm_engine.query, response, query, thinking_override=False)
            if response.thinking:
                response += f"<think>{response.thinking}</think>"
            success = await check_answer_for_question_simple(self.llm_engine.query, extracted_answer, str(task.corr_succ_states), query, thinking_override=True)
            return Result(
                task_id=task.id,
                success=success,
                response=response,
                extracted_answer=extracted_answer
            )
        tasks = [self._run_with_error_handling(run_single, task) for task in self.succ_action_state_tasks[:num_tasks]]
        return await asyncio.gather(*tasks)
    
    async def _test_succ_state_mapping_seperate(self, num_tasks: int = 1) -> list[Result]:
        """Test whether the LLM can generate a mapping of possible actions to their corresponding successor states but with each action being queried separately."""
        prompt = "Output exactly the successor state for the given action. Be concise."
        async def run_single(task: SuccActionStateTask):
            results = []
            for i, action in enumerate(task.poss_next_actions):
                query = f"{self.context}\n{prompt}\nState: {task.query_state}\nAction: {action}"
                response = await self.llm_engine.query(query)
                extracted_answer = await extract_answer_for_question(self.llm_engine.query, response, query, thinking_override=False)
                if response.thinking:
                    response += f"<think>{response.thinking}</think>"
                success = await check_answer_for_question_simple(self.llm_engine.query, extracted_answer, task.corr_succ_states[action], query, thinking_override=True)
                results.append(Result(
                    task_id=task.id + f"_{i}",
                    success=success,
                    response=response,
                    extracted_answer=extracted_answer
                ))
            return results
        tasks = [self._run_with_error_handling(run_single, task) for task in self.succ_action_state_tasks[:num_tasks]]
        return await asyncio.gather(*tasks)

    async def _test_goal_state_verification(self, num_tasks: int = 1) -> list[Result]:
        """Test whether the LLM can verify if a given plan leads to a state that fulfills the goal conditions."""
        prompt = "Verify if the given plan leads to a state that fulfills the goal conditions. Be concise."
        async def run_single(task: GoalStateTask):
            query = f"{self.context}\n{prompt}\nStarting State: {task.start_state}\nPlan: {task.query_plan}\nGoal: {task.goal_atoms}"
            response = await self.llm_engine.query(query)
            extracted_answer = await extract_answer_for_question(self.llm_engine.query, response, query, thinking_override=False)
            if response.thinking:
                response += f"<think>{response.thinking}</think>"
            ground_truth = "yes/true" if task.is_goal_state else "no/false"
            success = await check_answer_for_question_simple(self.llm_engine.query, extracted_answer, ground_truth, query, thinking_override=True)
            return Result(
                task_id=task.id,
                success=success,
                response=response,
                extracted_answer=extracted_answer
            )
        tasks = [self._run_with_error_handling(run_single, task) for task in self.goal_state_tasks[:num_tasks]]
        return await asyncio.gather(*tasks)

    async def _test_novelty(self, num_tasks: int = 1) -> list[Result]:
        """Test whether the LLM can determine the novelty of a state compared to previous states."""
        async def run_single(task: NoveltyTask):
            response = await is_novel(self.llm_engine.query, task.query_state, task.previous_states)
            if task.novelty == -1 or task.novelty > task.instance_width:
                # would prune in IW so LLM should prune too -> llm should say not novel
                success = not response
            else:
                # would not prune in IW so LLM should say novel
                success = response
            return Result(
                task_id=task.id,
                success=success,
                response=str(response),
                extracted_answer="N/A"
            )
        tasks = [self._run_with_error_handling(run_single, task) for task in self.novelty_tasks[:num_tasks]]
        return await asyncio.gather(*tasks)

    async def _test_novelty_no_duplicates(self, num_tasks: int = 1) -> list[Result]:
        """Test whether the LLM can determine the novelty of a state compared to previous states.
        This version ensures the pruning is not done because the new state is a duplicate of a previous state."""
        async def run_single(task: NoveltyTask):
            response = await is_novel(self.llm_engine.query, task.query_state, task.previous_states)
            if task.novelty == -1:
                raise ValueError("Task has novelty -1, must be an error in task generation.")
            elif task.novelty > task.instance_width:
                # would prune in IW so LLM should prune too -> llm should say not novel
                success = not response
            else:
                raise ValueError("Task has novelty less or equal to instance width, must be an error in task generation.")
            return Result(
                task_id=task.id,
                success=success,
                response=str(response),
                extracted_answer="N/A"
            )
        tasks = [self._run_with_error_handling(run_single, task) for task in self.novelty_tasks_no_duplicates[:num_tasks]]
        return await asyncio.gather(*tasks)

    async def _test_novelty_estimation(self, num_tasks: int = 1) -> list[Result]:
        """Test whether the LLM can estimate the novelty of a state compared to previous states as a number."""
        async def run_single(task: NoveltyTask):
            reasoning = not self.llm_engine.thinking
            response, estimated_novelty = await estimate_novelty(self.llm_engine, task.query_state, task.previous_states, reasoning=reasoning)
            if estimated_novelty == task.novelty:
                success = True
            else:
                success = False
            return Result(
                task_id=task.id,
                success=success,
                response=response,
                extracted_answer=str(estimated_novelty)
            )
        tasks = [self._run_with_error_handling(run_single, task) for task in self.novelty_tasks[:num_tasks]]
        return await asyncio.gather(*tasks)

    async def _test_novelty_estimation_no_duplicates(self, num_tasks: int = 1) -> list[Result]:
        """Test whether the LLM can estimate the novelty of a state compared to previous states as a number.
        This version ensures the pruning is not done because the new state is a duplicate of a previous state."""
        async def run_single(task: NoveltyTask):
            reasoning = not self.llm_engine.thinking
            response, estimated_novelty = await estimate_novelty(self.llm_engine, task.query_state, task.previous_states, reasoning=reasoning)
            if estimated_novelty == task.novelty:
                success = True
            else:
                success = False
            return Result(
                task_id=task.id,
                success=success,
                response=response,
                extracted_answer=str(estimated_novelty)
            )
        tasks = [self._run_with_error_handling(run_single, task) for task in self.novelty_tasks_no_duplicates[:num_tasks]]
        return await asyncio.gather(*tasks)

    async def _test_problem_width_estimation(self, num_tasks: int = 1) -> list[Result]:
        """Test whether the LLM can estimate the width of a problem instance."""
        async def run_single(task: NoveltyTask):
            reasoning = not self.llm_engine.thinking
            response, estimated_width = await estimate_problem_width(self.llm_engine, self.context, task.query_state, task.goal_atoms, reasoning=reasoning)
            if estimated_width == task.instance_width:
                success = True
            else:
                success = False
            return Result(
                task_id=task.id,
                success=success,
                response=response,
                extracted_answer=str(estimated_width)
            )
        tasks = [self._run_with_error_handling(run_single, task) for task in self.novelty_tasks[:num_tasks]]
        return await asyncio.gather(*tasks)
