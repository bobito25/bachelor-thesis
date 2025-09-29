from __future__ import annotations
import asyncio
from dataclasses import dataclass
import json
from typing import List, Generator
import uuid
import logging

from llm_engine import Engine
from llm_compare import extract_answer_for_question, check_answer_for_question, does_answer_solve_problem
from llm_novelty import is_novel, estimate_novelty
from tot_config import Configuration

tot_logger = logging.getLogger("nov_tot." + __name__)

class TreeOfThoughts:
    """
    Tree of Thoughts class.
    """

    @dataclass
    class State:
        """
        States are nodes in Tree of Thoughts.
        """
        contents: str
        children: List[TreeOfThoughts.State]
        parent: TreeOfThoughts.State
        depth: int
        finished: bool = False
        goal_accomplished: bool = False
        uuid: str = None
        original_response: str = None
        used_prompt: str = None
        novelty_summary: str = None
        action: str = None
        used_action_prompt: str = None
        
        def __post_init__(self):
            """
            Post initialization method for ID generation.
            """
            if self.uuid is None:
                new_uuid = uuid.uuid4().hex
                self.uuid = new_uuid

        def create_child(self, contents: str) -> TreeOfThoughts.State:
            """
            Create child state.
            """
            child = TreeOfThoughts.State(
                contents=contents,
                children=[],
                parent=self,
                depth=self.depth + 1,
            )
            self.children.append(child)
            return child
        
        def get_combined_contents(self, divider: str = "\n", include_init: bool = True) -> str:
            """
            Get combined contents of state and its ancestors.
            """
            if self.parent is None:
                if include_init:
                    return self.contents
                else:
                    return ""
            prev_contents = self.parent.get_combined_contents(divider=divider, include_init=include_init)
            if prev_contents == "":
                return self.contents
            else:
                return prev_contents + divider + self.contents
        
        def get_combined_actions(self, divider: str = "\n") -> str:
            """
            Get combined actions of state and its ancestors.
            """
            if self.parent is None:
                return self.action if self.action else ""
            prev_actions = self.parent.get_combined_actions(divider=divider)
            if prev_actions == "":
                return self.action if self.action else ""
            else:
                return prev_actions + divider + (self.action if self.action else "")


        def walk(self, exclude_self: bool = False) -> Generator[TreeOfThoughts.State, None, None]:
            """
            Walk through this state and all of its descendants using breadth-first search.
            """
            queue = [self]
            first_state = True
            while queue:
                state = queue.pop(0)
                if exclude_self and first_state:
                    first_state = False
                else:
                    yield state
                for child in state.children:
                    queue.append(child)

        def print_tree(self, level=0):
            """
            Recursively prints the tree structure with indentation.

            :param level: The current depth level (used for indentation).
            """
            print("  " * level + self.contents)
            for child in self.children:
                child.print_tree(level + 1)
        
        def to_dict(self) -> dict:
            """
            Serialize the state and all descendants into a dictionary.
            Note: The parent is not serialized to avoid circular references.
            """
            return {
                "contents": self.contents,
                "depth": self.depth,
                "finished": self.finished,
                "goal_accomplished": self.goal_accomplished,
                "uuid": self.uuid,
                "original_response": self.original_response,
                "used_prompt": self.used_prompt,
                "novelty_summary": self.novelty_summary,
                "children": [child.to_dict() for child in self.children],
                "action": self.action,
                "used_action_prompt": self.used_action_prompt,
            }
        
        @classmethod
        def from_dict(cls, data: dict, parent: TreeOfThoughts.State = None) -> TreeOfThoughts.State:
            """
            Deserialize a state (and its children) from a dictionary.
            """
            state = cls(
                contents=data["contents"],
                children=[],  # will be populated below
                parent=parent,
                depth=data["depth"],
                finished=data.get("finished", False),
                goal_accomplished=data.get("goal_accomplished", False),
                uuid=data.get("uuid", None),
                original_response=data.get("original_response", None),
                used_prompt=data.get("used_prompt", None),
                novelty_summary=data.get("novelty_summary", None),
                action=data.get("action", None),
                used_action_prompt=data.get("used_action_prompt", None),
            )
            for child_data in data.get("children", []):
                child = cls.from_dict(child_data, parent=state)
                state.children.append(child)
            return state

    def __init__(self, llm_engine: Engine, config: Configuration, max_depth = 3, branch_factor = 3, token_usage = 0, context = ""):
        """
        Initialize Tree of Thoughts object.

        Args:
            llm_engine (Engine): Object for interacting with LLM.
            context (str): Context for all problems.
            max_depth (int): Maximum depth of a run of Tree of Thoughts.
            branch_factor (int): Branch factor of Tree of Thoughts
        """
        self.llm_engine = llm_engine
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.root_state = None
        self.states_by_depth = {}
        self.states_with_goal = []
        self.config = config
        self.mode = config.mode
        self.stream = config.mode == "stream"
        self.novelty = config.novelty
        self.explicit_state_action = config.mode == "explicit_state_action"
        self.novelty_estimation = config.novelty_estimation
        if llm_engine:
            print(f"Tree of Thoughts initialized with mode: {self.mode}, max_depth: {self.max_depth}, branch_factor: {self.branch_factor}, novelty: {self.novelty}, explicit_state_action: {self.explicit_state_action}, novelty_estimation: {self.novelty_estimation}")
        if not self.stream:
            self.succ_prompt = config.succ_prompt
        if self.mode == "base":
            self.gen_succ_prompt = config.gen_succ_prompt
        elif self.explicit_state_action:
            self.stream_prompt = config.stream_prompt
        self.given_solution = None
        self.token_usage = token_usage
        self.context = context
        self.num_times_pruned = 0
        self.num_shortened_prompts = 0

    async def query(self, query: str) -> str:
        response = await self.llm_engine.query(query)
        self.token_usage += response.token_usage
        return response

    def reset(self):
        """
        Reset the state of the Tree of Thoughts object to allow reuse for a new problem.
        Clears all stored states and problem-specific data.
        """
        self.init_state = None
        self.states_by_depth = {}
        self.states_with_goal = []

    async def run(self, problem_statement: str, parallel: bool = True, breadth_first: bool = True, problem_width: int | None = None) -> str | None:
        """
        Get solution using Tree of Thoughts.
        """
        if getattr(self, "init_state", None) is not None:
            print("Warning: Overwriting existing state of ToT.")
            self.reset()
        self.init_state = TreeOfThoughts.State(
            contents=problem_statement,
            children=[],
            parent=None,
            depth=0,
        )
        self.states_by_depth[0] = [self.init_state]
        self.run_final_answer = None
        if self.novelty_estimation:
            if not self.explicit_state_action:
                raise ValueError("novelty_estimation can only be used with explicit_state_action mode.")
            if not self.novelty:
                raise ValueError("novelty_estimation can only be used if novelty is True.")
            if problem_width is None:
                raise ValueError("problem_width must be provided if novelty_estimation is True.")
            self.problem_width = problem_width
        if self.novelty and parallel:
                raise ValueError("novelty is not supported with parallel execution.")
        if breadth_first:
            await self._expand_breadth_first(self.init_state, self.max_depth, parallel=parallel)
        else:
            await self._expand_depth_first(self.init_state, self.max_depth, parallel=parallel)
        return await self._get_solution()
    
    async def run_against(self, problem_statement: str, solution: str, parallel: bool = True, breadth_first = True) -> tuple[bool, str | None]:
        """
        Run Tree of Thoughts against a solution to see whether the tree of thoughts can find the solution.
        """
        answer = await self.run(problem_statement, parallel=parallel, breadth_first=breadth_first)
        return answer != None, answer

    async def _expand_node(self, node: 'TreeOfThoughts.State', next_level: list = None, depth: int = None, overwrite: bool = False, parallel: bool = True):
        """
        Common logic for expanding a single node: handles children, overwriting, and generating successors.
        Optionally appends new children to next_level (for BFS).
        """
        if node.children and not overwrite:
            if next_level is not None:
                next_level.extend(node.children)
            return []
        else:
            if node.children:
                # Clear children if overwriting
                for child in node.children:
                    try:
                        index = next(i for i, s in enumerate(self.states_by_depth[child.depth]) if s.uuid == child.uuid)
                    except StopIteration:
                        raise ValueError("Child not found in depth list.")
                    del self.states_by_depth[child.depth][index]
                node.children = []

            async def process_branch():
                tot_logger.info(f"Processing branch at depth {node.depth+1} (parent uuid: {node.uuid})")
                succ = await self._generate_successor(node)
                if succ:
                    tot_logger.info(f"Generated successor at depth {node.depth+1} (parent uuid: {node.uuid})")
                    if next_level is not None:
                        next_level.append(succ)
                    if depth is not None:
                        await self._expand_depth_first(succ, depth - 1, parallel=parallel)
                return succ

            coros = [process_branch() for _ in range(self.branch_factor)]
            if coros:
                scheduled = [asyncio.create_task(c) for c in coros]
                tot_logger.info(f"Starting {len(coros)} {'parallel' if parallel else 'sequential'} LLM requests at depth {node.depth+1} (parent uuid: {node.uuid})")
                if parallel:
                    results = await asyncio.gather(*scheduled)
                else:
                    # sequential: still await the scheduled Tasks to ensure they run to completion
                    results = []
                    for task in scheduled:
                        results.append(await task)
                tot_logger.info(f"Finished {len(coros)} {'parallel' if parallel else 'sequential'} LLM requests at depth {node.depth+1} (parent uuid: {node.uuid})")
                return results
            return []

    async def _expand_depth_first(self, state: 'TreeOfThoughts.State', depth: int, overwrite: bool = False, stop_at_goal: bool = True, parallel: bool = True):
        """
        Expand state to depth (DFS).
        """
        if depth == 0 or state.goal_accomplished or state.finished or (stop_at_goal and self.states_with_goal):
            return
        tot_logger.debug(f"Trying to expand state at depth {state.depth} (parent uuid: {state.uuid})")
        if state.children and not overwrite:
            tasks = [self._expand_depth_first(child, depth - 1, parallel=parallel) for child in state.children]
            tot_logger.debug(f"Found existing children and overwrite disabled, expanding {len(tasks)} children at depth {state.depth+1} (parent uuid: {state.uuid})")
            if parallel:
                await asyncio.gather(*tasks)
            else:
                for t in tasks:
                    await t
        else:
            await self._expand_node(state, depth=depth, parallel=parallel)

    async def _expand_breadth_first(self, state: 'TreeOfThoughts.State', depth: int, overwrite: bool = False, stop_at_goal: bool = True, parallel: bool = True):
        """
        Expand state to depth using breadth-first search.
        """
        if depth == 0 or state.goal_accomplished or state.finished or (stop_at_goal and self.states_with_goal):
            return

        current_level = [state]
        for d in range(depth):
            if stop_at_goal and self.states_with_goal:
                break
            next_level = []
            tasks = []
            for node in current_level:
                if node.goal_accomplished or node.finished:
                    continue
                # Use the helper to expand node and collect new children in next_level
                tasks.append(self._expand_node(node, next_level=next_level, parallel=parallel, overwrite=overwrite))
            if tasks:
                if parallel:
                    await asyncio.gather(*tasks)
                else:
                    for t in tasks:
                        await t
            current_level = next_level
            if not current_level:
                break

    async def goal_verifier(self, state: 'TreeOfThoughts.State') -> bool:
        """
        Verify if the current state is a goal state.
        """
        tot_logger.debug(f"Verifying goal for state at depth {state.depth} (state uuid: {state.uuid})")
        try:
            # Extract the answer from the state contents
            if self.stream:
                extracted_answer = await extract_answer_for_question(self.query, state.get_combined_contents(include_init=False, divider=""), self.init_state.contents)
            elif self.explicit_state_action:
                extracted_answer = await extract_answer_for_question(self.query, state.get_combined_actions(divider="\n"), self.init_state.contents)
            else:
                extracted_answer = await extract_answer_for_question(self.query, state.get_combined_contents(include_init=False), self.init_state.contents)

            if self.given_solution is not None:
                # If a solution is given, check if the state matches the solution
                return await check_answer_for_question(self.llm_engine.query, extracted_answer, self.given_solution, self.init_state.contents)
            else:
                return await does_answer_solve_problem(self.llm_engine.query, extracted_answer, self.init_state.contents)
        except Exception as e:
            tot_logger.error(f"Error verifying goal for state at depth {state.depth} (state uuid: {state.uuid}): {e}")
            return False

    async def _generate_successor(self, state: State, stop_at_goal: bool = True) -> State | None:
        """
        Generate successor state and verify if it is a goal state.
        """
        if stop_at_goal and self.states_with_goal:
            tot_logger.debug(f"Stopping, already found {len(self.states_with_goal)} goal states.")
            return None

        def parse_next_step_from_response(response: str) -> str:
            """
            Parse response from LLM to get next step.
            """
            return response.split("[OUTPUT]")[-1].strip()

        def shorten_prompt(prompt: str, max_length: int = 1000) -> str:
            """
            Shorten the prompt by replacing the middle with "...".
            """
            if len(prompt) <= max_length:
                return prompt
            marker = " [...] "
            # If max_length is too small to accommodate marker plus a couple chars, just return prompt
            if max_length <= len(marker) + 2:
                return prompt
            # Split remaining allowed length between left and right portions
            left = (max_length - len(marker)) // 2
            right = max_length - len(marker) - left
            shortened = prompt[:left] + marker + prompt[-right:]
            self.num_shortened_prompts += 1
            return shortened

        # Query LLM for next step
        if self.stream:
            complete_prompt = self.stream_prompt + state.get_combined_contents(divider="")
            response = await self.query(complete_prompt, max_tokens=10)
            succ_state = response
        elif self.explicit_state_action:
            if state.action is None:
                # root state, no action yet
                complete_prompt_action = self.config.succ_prompt + "\n" + self.init_state.contents + "\n" + self.config.gen_action_prompt
            else:
                complete_prompt_action = self.config.succ_prompt + "\n" + self.init_state.contents + "\n" + self.config.progress_prompt + "\n" + state.get_combined_actions() + "\n" + self.config.current_state_prompt + "\n" + state.contents + "\n" + self.config.gen_action_prompt

            complete_prompt_action = shorten_prompt(complete_prompt_action, max_length=16000)
            response_action = await self.query(complete_prompt_action)
            parsed_response_action = parse_next_step_from_response(response_action)
            tot_logger.debug(f"LLM next step actions query: {complete_prompt_action}")
            tot_logger.debug(f"LLM next step actions response: {response_action}")

            if state.parent is None:
                # root state
                complete_prompt = self.config.succ_prompt + "\n" + state.contents + "\n" + self.config.map_succ_prompt + "\n" + parsed_response_action
            else:
                complete_prompt = self.config.succ_prompt + "\n" + self.context + "\n" + self.config.current_state_prompt + "\n" + state.contents + "\n" + self.config.map_succ_prompt + "\n" + parsed_response_action
            complete_prompt = shorten_prompt(complete_prompt, max_length=14000)
            response = await self.query(complete_prompt)
            parsed_response = parse_next_step_from_response(response)
            succ_state = parsed_response
        else:
            if state.parent is None:
                # root state
                complete_prompt = self.config.succ_prompt + "\n" + self.init_state.contents + "\n" + self.config.gen_succ_prompt
            else:
                complete_prompt = self.config.succ_prompt + "\n"
                complete_prompt += self.init_state.contents + "\n" + self.config.progress_prompt + "\n"
                complete_prompt += state.get_combined_contents(include_init=False)
                complete_prompt += "\n" + self.config.gen_succ_prompt
            complete_prompt = shorten_prompt(complete_prompt, max_length=16000)
            response = await self.query(complete_prompt)
            parsed_response = parse_next_step_from_response(response)
            succ_state = parsed_response
        tot_logger.debug(f"LLM next step query: {complete_prompt}")
        tot_logger.debug(f"LLM next step response: {response}")
        
        if self.novelty:
            if self.stream:
                raise NotImplementedError("Novelty is not implemented for streaming mode.")
            elif self.novelty_estimation:
                previous_states = [s.contents for s in self.init_state.walk(exclude_self=True) if s.contents]
                reasoning = not self.llm_engine.thinking
                _, estimated_novelty = await estimate_novelty(self.llm_engine, parsed_response, previous_states, max_size=48000, reasoning=reasoning, examples=False)
                tot_logger.info(f"Estimated novelty for state at depth {state.depth+1} (parent uuid: {state.uuid}): {estimated_novelty}")
                prune = estimated_novelty == -1 or estimated_novelty > self.problem_width
                if prune:
                    self.num_times_pruned += 1
                    tot_logger.info(f"Pruning non-novel state at depth {state.depth+1} (parent uuid: {state.uuid})")
                    return None
            else:
                previous_states = [s.contents for s in self.init_state.walk(exclude_self=True) if s.contents]
                novel = await is_novel(self.query, parsed_response, previous_states)
                if not novel:
                    self.num_times_pruned += 1
                    tot_logger.info(f"Pruning non-novel state at depth {state.depth+1} (parent uuid: {state.uuid})")
                    return None

        # Create child state
        succ = state.create_child(succ_state)
        if self.explicit_state_action:
            succ.original_response = response_action + "\n" +  response
            succ.action = parsed_response_action
            succ.used_action_prompt = complete_prompt_action
        else:
            succ.original_response = response
        succ.used_prompt = complete_prompt
        if succ.depth not in self.states_by_depth:
            self.states_by_depth[succ.depth] = [succ]
        else:
            self.states_by_depth[succ.depth].append(succ)

        # Verify if the llm is done
        if self.stream:
            if "[FINISHED]" in succ.get_combined_contents(divider=""):
                succ.finished = True
        else:
            if "[FINISHED]" in succ.original_response:
                succ.finished = True

        # Check if the goal is accomplished
        if succ.finished:
            succ.goal_accomplished = await self.goal_verifier(succ)

        if succ.goal_accomplished:
            self.states_with_goal.append(succ)

        return succ

    async def _get_solution(self) -> str | None:
        """
        Get solution from Tree of Thoughts.
        """
        if not self.states_with_goal:
            return None
        if self.explicit_state_action:
            extracted_answer = await extract_answer_for_question(self.query, self.states_with_goal[0].get_combined_actions(divider="\n"), self.init_state.contents)
        else:
            divider = "" if self.stream else "\n"
            complete_path = self.states_with_goal[0].get_combined_contents(divider=divider, include_init=False)
            extracted_answer = await extract_answer_for_question(self.query, complete_path, self.init_state.contents)
        tot_logger.info(f"Extracted answer: {extracted_answer}")
        self.run_final_answer = extracted_answer
        return extracted_answer

    def _rebuild_indices(self) -> None:
        """
        Rebuild states_by_depth and states_with_goal from the tree.
        """
        self.states_by_depth = {}
        self.states_with_goal = []
        for state in self.init_state.walk():
            if state.depth not in self.states_by_depth:
                self.states_by_depth[state.depth] = []
            self.states_by_depth[state.depth].append(state)
            if state.goal_accomplished:
                self.states_with_goal.append(state)

    def serialize(self) -> str:
        """
        Serialize the TreeOfThoughts object (the tree structure and config) into a JSON string.
        Note: llm_engine is not serialized.
        """
        data = {
            "max_depth": self.max_depth,
            "branch_factor": self.branch_factor,
            "config": self.config.model_dump(),
            "init_state": self.init_state.to_dict() if self.init_state else None,
            "token_usage": self.token_usage,
            "context": self.context,
            "run_final_answer": getattr(self, "run_final_answer", None),
        }
        return json.dumps(data)

    @classmethod
    def deserialize(cls, json_str: str, llm_engine: Engine) -> TreeOfThoughts:
        """
        Deserialize a TreeOfThoughts object from a JSON string.
        llm_engine must be provided.
        """
        data = json.loads(json_str)
        instance = cls(
            llm_engine=llm_engine,
            config=Configuration.model_validate(data["config"]),
            max_depth=data["max_depth"],
            branch_factor=data["branch_factor"],
            token_usage=data.get("token_usage", 0),
            context=data.get("context", "")
        )
        instance.run_final_answer = data.get("run_final_answer", None)
        if data.get("init_state") is not None:
            instance.init_state = cls.State.from_dict(data["init_state"], None)
            instance._rebuild_indices()
        return instance
