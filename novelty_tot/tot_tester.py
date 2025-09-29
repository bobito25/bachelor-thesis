import json
import os

from lifted_pddl import Parser
from pydantic import BaseModel
import asyncio
import traceback

from tot_config import Configuration 
from tot import TreeOfThoughts
from llm_engine import Engine
from llm_compare import check_answer_for_question_simple, extract_plan_for_question
from plan_gen import compute_optimal_plan
from plan_val import validate_plan
from pddl_translation.translator import Translator
from pddl_translation.pddl_translation import text_to_pddl_plan_blocksworld, text_to_pddl_plan_logistics
from llm_novelty import estimate_problem_width


class Task(BaseModel):
    id: str
    query: str
    start_state: str
    goal_atoms: str
    example_solution: str


class Result(BaseModel):
    task_id: str
    success: bool
    answer: str
    token_usage: int = 0
    num_times_pruned: int = 0


def get_tasks(prompt_translator: Translator):
    """Get the tasks for the goal state verification task."""
    # Check if the tasks are already cached
    cache_dir = "test_data/" + prompt_translator.type
    cache_file = "tot_plan_tasks.json"
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            Task(
                id=task["id"],
                query=task["query"],
                start_state=task["start_state"],
                goal_atoms=task["goal_atoms"],
                example_solution=task["example_solution"]
            )
            for task in data
        ]

    tasks: list[Task] = []
    domain_path = 'test_data/PDDL/Blocksworld/domain.pddl'
    instances_dir = 'test_data/PDDL/Blocksworld/instances'
    for idx, fname in enumerate(os.listdir(instances_dir)):
        if not fname.endswith('.pddl'):
            continue
        parser = Parser()
        parser.parse_domain(domain_path)
        problem_path = os.path.join(instances_dir, fname)
        parser.parse_problem(problem_path)

        # plan via Fast‑Downward
        raw_plan = compute_optimal_plan(domain_path, problem_path)
        plan_actions = raw_plan.strip().splitlines() if raw_plan else []
        plan = ', '.join(plan_actions)

        start_state = prompt_translator.state_to_string(parser.atoms)
        goal_atoms = prompt_translator.goal_atoms_to_string(parser.goals)
        plan = prompt_translator.plan_to_string(plan_actions)
        query = f"Given the start state {start_state} and the goal atoms {goal_atoms}, generate a plan to reach the goal state. [PLAN]"
        tasks.append(
            Task(
                id=fname,
                query=query,
                start_state=start_state,
                goal_atoms=goal_atoms,
                example_solution=plan
            )
        )

    # Save to cache
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([task.model_dump() for task in tasks], f, indent=2, ensure_ascii=False)

    return tasks


def get_hard_tasks(prompt_translator: Translator):
    """Get the tasks for the goal state verification task."""
    # Check if the tasks are already cached
    cache_dir = "test_data/" + prompt_translator.type
    cache_file = "tot_plan_tasks_hard.json"
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            Task(
                id=task["id"],
                query=task["query"],
                start_state=task["start_state"],
                goal_atoms=task["goal_atoms"],
                example_solution=task["example_solution"]
            )
            for task in data
        ]

    tasks: list[Task] = []
    domain_path = 'test_data/PDDL/Blocksworld/domain.pddl'
    instances_dir = 'test_data/PDDL/Blocksworld/hard_instances'
    for idx, fname in enumerate(os.listdir(instances_dir)):
        if not fname.endswith('.pddl'):
            continue
        parser = Parser()
        parser.parse_domain(domain_path)
        problem_path = os.path.join(instances_dir, fname)
        parser.parse_problem(problem_path)

        # plan via Fast‑Downward
        raw_plan = compute_optimal_plan(domain_path, problem_path)
        plan_actions = raw_plan.strip().splitlines() if raw_plan else []
        plan = ', '.join(plan_actions)

        start_state = prompt_translator.state_to_string(parser.atoms)
        goal_atoms = prompt_translator.goal_atoms_to_string(parser.goals)
        plan = prompt_translator.plan_to_string(plan_actions)
        query = f"Given the start state {start_state} and the goal atoms {goal_atoms}, generate a plan to reach the goal state. [PLAN]"
        tasks.append(
            Task(
                id=fname,
                query=query,
                start_state=start_state,
                goal_atoms=goal_atoms,
                example_solution=plan
            )
        )

    # Save to cache
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([task.model_dump() for task in tasks], f, indent=2, ensure_ascii=False)

    return tasks


def get_tasks_logistics(prompt_translator: Translator):
    """Get the tasks for the logistics domain."""
    # Check if the tasks are already cached
    cache_dir = "test_data/" + prompt_translator.type
    cache_file = "tot_plan_tasks_logistics.json"
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            Task(
                id=task["id"],
                query=task["query"],
                example_solution=task["example_solution"]
            )
            for task in data
        ]
    tasks: list[Task] = []
    domain_path = 'test_data/PDDL/logistics/domain.pddl'
    instances_dir = 'test_data/PDDL/logistics/instances'
    for idx, fname in enumerate(os.listdir(instances_dir)):
        if not fname.endswith('.pddl'):
            continue
        parser = Parser()
        parser.parse_domain(domain_path)
        problem_path = os.path.join(instances_dir, fname)
        parser.parse_problem(problem_path)

        # plan via Fast‑Downward
        raw_plan = compute_optimal_plan(domain_path, problem_path)
        plan_actions = raw_plan.strip().splitlines() if raw_plan else []
        plan = ', '.join(plan_actions)

        start_state = prompt_translator.state_to_string(parser.atoms)
        goal_atoms = prompt_translator.goal_atoms_to_string(parser.goals)
        plan = prompt_translator.plan_to_string(plan_actions)

        query = f"Given the start state {start_state} and the goal atoms {goal_atoms}, generate a plan to reach the goal state. [PLAN]"
        tasks.append(
            Task(
                id=fname,
                query=query,
                example_solution=plan
            )
        )

    # Save to cache
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([task.model_dump() for task in tasks], f, indent=2, ensure_ascii=False)

    return tasks


class TestRunner:
    """Tests a given ToT configuration by running a number of tests/tasks"""

    def __init__(self, llm_engine: Engine, prompt_translator: Translator, context: str = "", logistics: bool = False, hard: bool = False):
        if not hard:
            self.tasks = get_tasks(prompt_translator)
        else:
            self.tasks = get_hard_tasks(prompt_translator)
        if logistics:
            self.logistics_tasks = get_tasks_logistics(prompt_translator)
        self.llm_engine = llm_engine
        if context != "":
            self.context = context + "\n"
        else:
            self.context = context
        self.math_context = "Solve the math problem in small steps."
        self.prompt_translator = prompt_translator

    async def run_task(self, tot: TreeOfThoughts, task: Task, parallel: bool = True, solution_given: bool = False, breadth_first: bool = True) -> Result:
        """Run the task using the Tree of Thoughts framework.

        Args:
            tot (TreeOfThoughts): The Tree of Thoughts instance to use.
            task (Task): The task to run.
            parallel (bool): Whether to run the task in parallel.
            solution_given (bool): Whether the example solution is given as a reference so the LLM can verify the solution.
            breadth_first (bool): Whether to use breadth-first search.
        """
        if solution_given:
            query = self.context + task.query
            success, answer = await tot.run_against(query, task.example_solution, parallel=parallel, breadth_first=breadth_first)
            if not answer:
                answer = "No answer found."
        else:
            query = self.context + task.query
            if tot.config.novelty_estimation:
                # estimate problem width
                reasoning = not self.llm_engine.thinking
                _, problem_width = await estimate_problem_width(self.llm_engine, self.context, task.start_state, task.goal_atoms, reasoning=reasoning)
            else:
                problem_width = None
            answer = await tot.run(query, parallel=parallel, breadth_first=breadth_first, problem_width=problem_width)
            if not answer:
                answer = "No answer found."
                success = False
            else:
                # verify the plan
                success = await check_answer_for_question_simple(
                    self.llm_engine.query,
                    answer,
                    task.example_solution,
                    task.query,
                    thinking_override=True
                )
                if not success:
                    # try to validate as plan for blocksworld
                    pddl_plan = text_to_pddl_plan_blocksworld(answer, self.prompt_translator)
                    if pddl_plan:
                        # Validate the PDDL plan
                        domain_pddl_path = "test_data/PDDL/Blocksworld/domain.pddl"
                        problem_pddl_path = "test_data/PDDL/Blocksworld/instances/" + task.id
                        # make sure paths exist
                        if not os.path.exists(domain_pddl_path):
                            print(f"Domain PDDL file does not exist: {domain_pddl_path}")
                            success = False
                            answer += "\nThe domain PDDL file does not exist."
                        if not os.path.exists(problem_pddl_path):
                            print(f"Problem PDDL file does not exist: {problem_pddl_path}")
                            success = False
                            answer += "\nThe problem PDDL file does not exist."
                        success = validate_plan(domain_pddl_path, problem_pddl_path, pddl_plan)
                        if success:
                            answer += "\nThe plan is valid according to the PDDL validator."
                        else:
                            answer += "\nThe plan is not valid according to the PDDL validator."
                    else:
                        success = False
                        answer += "\nThe plan could not be converted to PDDL format."

        return Result(
            task_id=task.id,
            success=success,
            answer=answer,
            token_usage=tot.token_usage if hasattr(tot, 'token_usage') else 0,
            num_times_pruned=tot.num_times_pruned if hasattr(tot, 'num_times_pruned') else 0
        )

    async def run_task_with_error_handling(self, tot: TreeOfThoughts, task: Task, parallel: bool = True, solution_given: bool = False, breadth_first: bool = True) -> Result:
        """Wrapper around run_task that handles exceptions."""
        try:
            return await self.run_task(tot, task, parallel, solution_given, breadth_first)
        except Exception as e:
            # More verbose error reporting with full traceback
            tb = traceback.format_exc()
            print(f"Error running task {task.id}: {e.__class__.__name__}: {e}\n{tb}")
            return Result(
                task_id=task.id,
                success=False,
                answer=f"Error occurred: {e.__class__.__name__}: {e}\nTraceback:\n{tb}",
                token_usage=0
            )

    async def run(self, config: Configuration, num_tasks: int = 1, instances: list[str] = None, write_tot: bool = False, max_depth: int = 5, branch_factor: int = 2, parallel_tasks: bool = True, parallel_tot: bool = False, breadth_first: bool = True, output_dir: str = "test_tots") -> tuple[list[Result], dict]:
        results = []
        tasks_to_run = [self.tasks[i] for i in instances] if instances else self.tasks[:num_tasks]
        tots = [
            TreeOfThoughts(
                llm_engine=self.llm_engine,
                config=config,
                max_depth=max_depth,
                branch_factor=branch_factor,
                context=self.context
            )
            for _ in tasks_to_run
        ]
        if parallel_tasks:
            coros = [
                self.run_task_with_error_handling(
                    tots[i],
                    task,
                    parallel=parallel_tot,
                    breadth_first=breadth_first
                )
                for i, task in enumerate(tasks_to_run)
            ]
            results = await asyncio.gather(*coros, return_exceptions=False)
        else:
            for tot in tots:
                result = await self.run_task(tot, task, parallel=False)
                results.append(result)
        summary = {
            "total_tasks": num_tasks,
            "successful_tasks": len([r for r in results if r.success]),
            "failed_tasks": len([r for r in results if not r.success]),
            "total_token_usage": sum(r.token_usage for r in results),
            "total_num_times_pruned": sum(r.num_times_pruned for r in results),
            "num_early_stops": self.llm_engine.num_early_stops,
            "num_max_context_exceeded": self.llm_engine.num_max_context_exceeded,
            "num_max_retries_exceeded": self.llm_engine.num_max_retries_exceeded,
            "num_retries": self.llm_engine.num_retries,
            "num_truncated_queries": self.llm_engine.num_truncated_queries,
            "num_shortened_prompts": sum(tot.num_shortened_prompts for tot in tots)
        }
        # make sure output_dir exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if write_tot and not parallel_tasks and tasks_to_run:
            # Only write the last ToT if not parallel (since each run has its own ToT)
            with open(f"{output_dir}/test_tot.json", "w", encoding="utf-8") as f:
                f.write(tot.serialize())
        elif write_tot and parallel_tasks and tasks_to_run:
            # Write all ToTs if parallel
            for i, task in enumerate(tasks_to_run):
                with open(f"{output_dir}/test_tot_{task.id}.json", "w", encoding="utf-8") as f:
                    f.write(tots[i].serialize())

        return results, summary

    async def run_logistics(self, config: Configuration, num_tasks: int = 1, instances: list[str] = None, write_tot: bool = False, max_depth: int = 5, branch_factor: int = 2, parallel_tasks: bool = True, parallel_tot: bool = False, breadth_first: bool = True, output_dir: str = "test_tots_logistics") -> tuple[list[Result], dict]:
        """Run the logistics domain tasks using Tree of Thoughts framework and save results."""
        if not hasattr(self, 'logistics_tasks'):
            raise ValueError("Logistics tasks not initialized. Please initialize TestRunner with logistics=True.")
        
        results = []
        tasks_to_run = [self.logistics_tasks[i] for i in instances] if instances else self.logistics_tasks[:num_tasks]
        tots = [
            TreeOfThoughts(
                llm_engine=self.llm_engine,
                config=config,
                max_depth=max_depth,
                branch_factor=branch_factor,
                context=self.context
            )
            for _ in tasks_to_run
        ]
        
        if parallel_tasks:
            coros = [
                self.run_task_logistics_with_error_handling(
                    tots[i],
                    task,
                    parallel=parallel_tot,
                    breadth_first=breadth_first
                )
                for i, task in enumerate(tasks_to_run)
            ]
            results = await asyncio.gather(*coros, return_exceptions=False)
        else:
            for tot in tots:
                result = await self.run_task_logistics(tot, task, parallel=False)
                results.append(result)
        
        summary = {
            "total_tasks": num_tasks,
            "successful_tasks": len([r for r in results if r.success]),
            "failed_tasks": len([r for r in results if not r.success]),
            "total_token_usage": sum(r.token_usage for r in results),
            "total_num_times_pruned": sum(r.num_times_pruned for r in results),
            "num_early_stops": self.llm_engine.num_early_stops,
            "num_max_context_exceeded": self.llm_engine.num_max_context_exceeded,
            "num_max_retries_exceeded": self.llm_engine.num_max_retries_exceeded,
            "num_retries": self.llm_engine.num_retries,
            "num_truncated_queries": self.llm_engine.num_truncated_queries,
            "num_shortened_prompts": sum(tot.num_shortened_prompts for tot in tots)
        }
        
        # make sure output_dir exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save results to JSON file
        results_data = {
            "summary": summary,
            "results": [result.model_dump() for result in results]
        }
        with open(f"{output_dir}/logistics_results.json", "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        if write_tot and not parallel_tasks and tasks_to_run:
            # Only write the last ToT if not parallel
            with open(f"{output_dir}/test_tot_logistics.json", "w", encoding="utf-8") as f:
                f.write(tot.serialize())
        elif write_tot and parallel_tasks and tasks_to_run:
            # Write all ToTs if parallel
            for i, task in enumerate(tasks_to_run):
                with open(f"{output_dir}/test_tot_logistics_{task.id}.json", "w", encoding="utf-8") as f:
                    f.write(tots[i].serialize())

        return results, summary

    async def run_task_logistics(self, tot: TreeOfThoughts, task: Task, parallel: bool = True, solution_given: bool = False, breadth_first: bool = True) -> Result:
        """Run a logistics domain task using the Tree of Thoughts framework."""
        if solution_given:
            query = self.context + task.query
            success, answer = await tot.run_against(query, task.example_solution, parallel=parallel, breadth_first=breadth_first)
            if not answer:
                answer = "No answer found."
        else:
            query = self.context + task.query
            answer = await tot.run(query, parallel=parallel, breadth_first=breadth_first)
            if not answer:
                answer = "No answer found."
                success = False
            else:
                # verify the plan
                success = await check_answer_for_question_simple(
                    self.llm_engine.query,
                    answer,
                    task.example_solution,
                    task.query,
                    thinking_override=True
                )
                if not success:
                    # try to validate as plan for logistics domain
                    pddl_plan = text_to_pddl_plan_logistics(answer, self.prompt_translator)
                    if pddl_plan:
                        # Validate the PDDL plan
                        domain_pddl_path = "test_data/PDDL/logistics/domain.pddl"
                        problem_pddl_path = "test_data/PDDL/logistics/instances/" + task.id
                        # make sure paths exist
                        if not os.path.exists(domain_pddl_path):
                            print(f"Domain PDDL file does not exist: {domain_pddl_path}")
                            success = False
                            answer += "\nThe domain PDDL file does not exist."
                        if not os.path.exists(problem_pddl_path):
                            print(f"Problem PDDL file does not exist: {problem_pddl_path}")
                            success = False
                            answer += "\nThe problem PDDL file does not exist."
                        success = validate_plan(domain_pddl_path, problem_pddl_path, pddl_plan)
                        if not success:
                            answer += "\nThe plan is not valid according to the PDDL validator."
                    else:
                        success = False
                        answer += "\nThe plan could not be converted to PDDL format."

        return Result(
            task_id=task.id,
            success=success,
            answer=answer,
            token_usage=tot.token_usage if hasattr(tot, 'token_usage') else 0,
            num_times_pruned=tot.num_times_pruned if hasattr(tot, 'num_times_pruned') else 0
        )

    async def run_task_logistics_with_error_handling(self, tot: TreeOfThoughts, task: Task, parallel: bool = True, solution_given: bool = False, breadth_first: bool = True) -> Result:
        """Wrapper around run_task_logistics that handles exceptions."""
        try:
            return await self.run_task_logistics(tot, task, parallel, solution_given, breadth_first)
        except Exception as e:
            # More verbose error reporting with full traceback
            tb = traceback.format_exc()
            print(f"Error running logistics task {task.id}: {e.__class__.__name__}: {e}\n{tb}")
            return Result(
                task_id=task.id,
                success=False,
                answer=f"Error occurred: {e.__class__.__name__}: {e}\nTraceback:\n{tb}",
                token_usage=0
            )

    async def run_naive(self, num_tasks: int = 1) -> list[Result]:
        """Run the tasks without ToT, just using the LLM engine."""
        results = []
        for task in self.tasks[:num_tasks]:
            answer = await self.llm_engine.query(self.context + "\n" + task.query)
            if not answer:
                answer = "No answer found."
                success = False
            else:
                extracted_answer = await extract_plan_for_question(
                    self.llm_engine.query,
                    answer,
                    task.query
                )
                answer = extracted_answer.strip()
                success = await check_answer_for_question_simple(
                    self.llm_engine.query,
                    extracted_answer,
                    task.example_solution,
                    task.query,
                    thinking_override=True
                )
                if not success:
                    # try to validate as plan for blocksworld
                    pddl_plan, _ = text_to_pddl_plan_blocksworld(extracted_answer, self.prompt_translator)
                    if pddl_plan:
                        # Validate the PDDL plan
                        domain_pddl_path = "test_data/PDDL/Blocksworld/domain.pddl"
                        problem_pddl_path = "test_data/PDDL/Blocksworld/instances/" + task.id
                        success = validate_plan(domain_pddl_path, problem_pddl_path, pddl_plan)
                        if not success:
                            answer += "\nThe plan is not valid according to the PDDL validator."
                    else:
                        success = False
                        answer += "\nThe plan could not be converted to PDDL format."
            results.append(Result(
                task_id=task.id,
                success=success,
                answer=answer,
                token_usage=answer.token_usage,
            ))
        summary = {
            "total_tasks": num_tasks,
            "successful_tasks": len([r for r in results if r.success]),
            "failed_tasks": len([r for r in results if not r.success]),
        }
        return results, summary

    async def run_math_task_with_error_handling(self, tot: TreeOfThoughts, task: dict, parallel: bool = True, solution_given: bool = False, breadth_first: bool = True) -> Result:
        """Runs the math task using tot and handles exceptions. Assumes task is a dict with keys 'id', 'problem', and 'solution'."""
        try:
            # run the math task
            query = self.math_context + task['problem']
            if tot.config.novelty_estimation:
                # estimate problem width
                reasoning = not self.llm_engine.thinking
                _, problem_width = await estimate_problem_width(self.llm_engine, self.math_context, task["problem"], "Correct solution is reached", reasoning=reasoning)
            else:
                problem_width = None
            answer = await tot.run(query, parallel=parallel, breadth_first=breadth_first, problem_width=problem_width)
            if not answer:
                answer = "No answer found."
                success = False
            else:
                # verify the plan
                success = await check_answer_for_question_simple(
                    self.llm_engine.query,
                    answer,
                    task["solution"],
                    task["problem"],
                    thinking_override=True
                )
            return Result(
                task_id=task['id'],
                success=success,
                answer=answer,
                token_usage=tot.token_usage if hasattr(tot, 'token_usage') else 0,
                num_times_pruned=tot.num_times_pruned if hasattr(tot, 'num_times_pruned') else 0
            )
        except Exception as e:
            # More verbose error reporting with full traceback
            tb = traceback.format_exc()
            print(f"Error running math task {task['id']}: {e.__class__.__name__}: {e}\n{tb}")
            return Result(
                task_id=task['id'],
                success=False,
                answer=f"Error occurred: {e.__class__.__name__}: {e}\nTraceback:\n{tb}",
            )

    async def run_math(self, config: Configuration, num_tasks: int = 1, write_tot: bool = False, max_depth: int = 5, branch_factor: int = 2, parallel_tasks: bool = True, parallel_tot: bool = False, breadth_first: bool = True, output_dir: str = "test_tots_math") -> tuple[list[Result], dict]:
        """Run the tasks on the math dataset using tot."""
        with open("test_data/math_50.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]
        
        tasks_to_run = data[:num_tasks]

        for i, task in enumerate(tasks_to_run):
            task["id"] = str(i)

        results = []
        tots = [
            TreeOfThoughts(
                llm_engine=self.llm_engine,
                config=config,
                max_depth=max_depth,
                branch_factor=branch_factor,
                context=self.math_context
            )
            for _ in tasks_to_run
        ]
        if parallel_tasks:
            coros = [
                self.run_math_task_with_error_handling(
                    tots[i],
                    task,
                    parallel=parallel_tot,
                    breadth_first=breadth_first
                )
                for i, task in enumerate(tasks_to_run)
            ]
            results = await asyncio.gather(*coros, return_exceptions=False)
        else:
            for i, tot in enumerate(tots):
                task = tasks_to_run[i]
                result = await self.run_math_task_with_error_handling(tot, task, parallel=False)
                results.append(result)
        summary = {
            "total_tasks": num_tasks,
            "successful_tasks": len([r for r in results if r.success]),
            "failed_tasks": len([r for r in results if not r.success]),
            "total_token_usage": sum(r.token_usage for r in results),
            "total_num_times_pruned": sum(r.num_times_pruned for r in results),
            "num_early_stops": self.llm_engine.num_early_stops,
            "num_max_context_exceeded": self.llm_engine.num_max_context_exceeded,
            "num_max_retries_exceeded": self.llm_engine.num_max_retries_exceeded,
            "num_retries": self.llm_engine.num_retries,
            "num_truncated_queries": self.llm_engine.num_truncated_queries,
            "num_shortened_prompts": sum(tot.num_shortened_prompts for tot in tots)
        }
        # make sure output_dir exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if write_tot and not parallel_tasks and tasks_to_run:
            # Only write the last ToT if not parallel (since each run has its own ToT)
            with open(f"{output_dir}/test_tot.json", "w", encoding="utf-8") as f:
                f.write(tot.serialize())
        elif write_tot and parallel_tasks and tasks_to_run:
            # Write all ToTs if parallel
            for i, task in enumerate(tasks_to_run):
                with open(f"{output_dir}/test_tot_{task['id']}.json", "w", encoding="utf-8") as f:
                    f.write(tots[i].serialize())

        return results, summary

    async def run_game_of_n_task_with_error_handling(self, tot: TreeOfThoughts, task: dict, parallel: bool = True, solution_given: bool = False, breadth_first: bool = True) -> Result:
        """Runs the game of n task using tot and handles exceptions. Assumes task is a dict with keys 'id', 'numbers', 'target', and 'solution'."""
        try:
            # run the game of n task
            query = self.context + f"\nStarting numbers:\n{','.join(map(str, task['numbers']))}\nTarget Number:\n{task['target']}\n"
            if tot.config.novelty_estimation:
                # estimate problem width
                reasoning = not self.llm_engine.thinking
                _, problem_width = await estimate_problem_width(self.llm_engine, self.context, ','.join(map(str, task['numbers'])), str(task['target']), reasoning=reasoning)
            else:
                problem_width = None
            answer = await tot.run(query, parallel=parallel, breadth_first=breadth_first, problem_width=problem_width)
            if not answer:
                answer = "No answer found."
                success = False
            else:
                # extract the answer
                query = "Change the following answer to be in the format 'n1 op1 n2 op2 n3 op3 n4' for the 4 numbers and 3 operations. Parentheses are also allowed and should be used to show what operations were performed first. If this is not possible, output None instead. Make sure your output can be evaluated using the python eval function. Only provide the expression, nothing else.\nAnswer to format:\n"
                query += answer
                extracted_answer = await self.llm_engine.query(query, thinking_override=True)
                extracted_answer = extracted_answer.strip().lower()
                if "none" in extracted_answer:
                    success = False
                    answer = f"Could not extract valid expression from answer: {answer}."
                else:
                    ops = ["+", "-", "*"]
                    # filter out invalid characters
                    filtered_answer = ''.join(c for c in extracted_answer if c.isdigit() or c in ops or c in '() ')
                    try:
                        result = eval(filtered_answer)
                        if abs(result - int(task['target'])) < 1e-6:
                            success = True
                            answer = filtered_answer
                        else:
                            success = False
                            answer = f"Extracted expression '{filtered_answer}' evaluates to {result}, which does not match target {task['target']}\nOriginal answer:\n{answer}."
                    except Exception as e:
                        success = False
                        answer = f"Error evaluating extracted expression '{filtered_answer}': {e}. Original answer:\n{answer}."

            return Result(
                task_id=task['id'],
                success=success,
                answer=answer,
                token_usage=tot.token_usage if hasattr(tot, 'token_usage') else 0,
                num_times_pruned=tot.num_times_pruned if hasattr(tot, 'num_times_pruned') else 0
            )
        except Exception as e:
            # More verbose error reporting with full traceback
            tb = traceback.format_exc()
            print(f"Error running game of n task {task['id']}: {e.__class__.__name__}: {e}\n{tb}")
            return Result(
                task_id=task['id'],
                success=False,
                answer=f"Error occurred: {e.__class__.__name__}: {e}\nTraceback:\n{tb}",
            )

    async def run_game_of_n(self, config: Configuration, num_tasks: int = 1, write_tot: bool = False, max_depth: int = 5, branch_factor: int = 2, parallel_tasks: bool = True, parallel_tot: bool = False, breadth_first: bool = True, output_dir: str = "test_tots_game_of_n") -> tuple[list[Result], dict]:
        """Run the tasks on the game of n dataset using tot."""
        self.context = "Use the given numbers to get to the target number.\nIn each step you may combine two numbers, never using a number twice but you must use all numbers.\nYour allowed operations are +, -, and *. The final solution should show how the numbers were combined including the order."
        
        # get tasks
        with open("test_data/game_of_n.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        tasks_to_run = data[:num_tasks]

        for i, task in enumerate(tasks_to_run):
            task["id"] = str(i)
    
        results = []
        tots = [
            TreeOfThoughts(
                llm_engine=self.llm_engine,
                config=config,
                max_depth=max_depth,
                branch_factor=branch_factor,
                context=self.context
            )
            for _ in tasks_to_run
        ]

        if parallel_tasks:
            coros = [
                self.run_game_of_n_task_with_error_handling(
                    tots[i],
                    task,
                    parallel=parallel_tot,
                    breadth_first=breadth_first
                )
                for i, task in enumerate(tasks_to_run)
            ]
            results = await asyncio.gather(*coros, return_exceptions=False)
        else:
            for i, tot in enumerate(tots):
                task = tasks_to_run[i]
                result = await self.run_game_of_n_task_with_error_handling(tot, task, parallel=False)
                results.append(result)
        
        summary = {
            "total_tasks": num_tasks,
            "successful_tasks": len([r for r in results if r.success]),
            "failed_tasks": len([r for r in results if not r.success]),
            "total_token_usage": sum(r.token_usage for r in results),
            "total_num_times_pruned": sum(r.num_times_pruned for r in results),
            "num_early_stops": self.llm_engine.num_early_stops,
            "num_max_context_exceeded": self.llm_engine.num_max_context_exceeded,
            "num_max_retries_exceeded": self.llm_engine.num_max_retries_exceeded,
            "num_retries": self.llm_engine.num_retries,
            "num_truncated_queries": self.llm_engine.num_truncated_queries,
            "num_shortened_prompts": sum(tot.num_shortened_prompts for tot in tots)
        }

        # make sure output_dir exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if write_tot and not parallel_tasks and tasks_to_run:
            # Only write the last ToT if not parallel (since each run has its own ToT)
            with open(f"{output_dir}/test_tot.json", "w", encoding="utf-8") as f:
                f.write(tot.serialize())
        elif write_tot and parallel_tasks and tasks_to_run:
            # Write all ToTs if parallel
            for i, task in enumerate(tasks_to_run):
                with open(f"{output_dir}/test_tot_{task['id']}.json", "w", encoding="utf-8") as f:
                    f.write(tots[i].serialize())
        return results, summary
