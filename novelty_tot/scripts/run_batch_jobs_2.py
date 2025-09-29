import asyncio
import os
import json
from datetime import datetime
from llm_engine import Engine, LlmApiClient
from tot_config import Configuration
from tot_tester import TestRunner
from task_tester import TaskRunner, TaskType, Result
from pddl_translation.translator import StandardTranslator, NatLangTranslator


async def run_action_gen_tests(runner: TaskRunner, thinking: bool = False, model: str = "Qwen/Qwen3-14B", translator: StandardTranslator | NatLangTranslator = StandardTranslator()):
    results_action_gen = await runner.run(TaskType.ACTION_GEN, num_tasks=50)
    num_correct_action_gen = sum([result.success for result in results_action_gen])
    print(f"Action Generation: {num_correct_action_gen}/{len(results_action_gen)} correct")
    log_results(results_action_gen, model, TaskType.ACTION_GEN, thinking, translator)


async def run_action_gen_single_tests(runner: TaskRunner, thinking: bool = False, model: str = "Qwen/Qwen3-14B", translator: StandardTranslator | NatLangTranslator = StandardTranslator()):
    results_action_gen_single = await runner.run(TaskType.ACTION_GEN_SINGLE, num_tasks=50)
    num_correct_action_gen_single_valid = sum([result[0].success for result in results_action_gen_single])
    num_correct_action_gen_single_optimal = sum([result[1].success for result in results_action_gen_single])
    print(f"Action Generation Single: {num_correct_action_gen_single_valid}/{len(results_action_gen_single)} valid, {num_correct_action_gen_single_optimal}/{len(results_action_gen_single)} optimal")
    flat_results = [result for sublist in results_action_gen_single for result in sublist]
    log_results(flat_results, model, TaskType.ACTION_GEN_SINGLE, thinking, translator)


async def run_succ_state_mapping_tests(runner: TaskRunner, thinking: bool = False, model: str = "Qwen/Qwen3-14B", translator: StandardTranslator | NatLangTranslator = StandardTranslator()):
    results_succ_state_mapping = await runner.run(TaskType.SUCC_STATE_MAPPING, num_tasks=50)
    num_correct_succ_state_mapping = sum([result.success for result in results_succ_state_mapping])
    print(f"Success State Mapping: {num_correct_succ_state_mapping}/{len(results_succ_state_mapping)} correct")
    log_results(results_succ_state_mapping, model, TaskType.SUCC_STATE_MAPPING, thinking, translator)


async def run_succ_state_mapping_separate_tests(runner: TaskRunner, thinking: bool = False, model: str = "Qwen/Qwen3-14B", translator: StandardTranslator | NatLangTranslator = StandardTranslator()):
    results_succ_state_mapping_separate = await runner.run(TaskType.SUCC_STATE_MAPPING_SEPERATE, num_tasks=50)
    flat_results = [result for sublist in results_succ_state_mapping_separate for result in sublist]
    num_correct_succ_state_mapping_separate = sum([result.success for result in flat_results])
    print(f"Success State Mapping Separate: {num_correct_succ_state_mapping_separate}/{len(flat_results)} correct")
    log_results(flat_results, model, TaskType.SUCC_STATE_MAPPING_SEPERATE, thinking, translator)


async def run_goal_state_verification_tests(runner: TaskRunner, thinking: bool = False, model: str = "Qwen/Qwen3-14B", translator: StandardTranslator | NatLangTranslator = StandardTranslator()):
    results_goal_state_ver = await runner.run(TaskType.GOAL_STATE_VERIFICATION, num_tasks=50)
    num_correct_goal_state_ver = sum([result.success for result in results_goal_state_ver])
    print(f"Goal State Verification: {num_correct_goal_state_ver}/{len(results_goal_state_ver)} correct")
    log_results(results_goal_state_ver, model, TaskType.GOAL_STATE_VERIFICATION, thinking, translator)


async def run_novelty_tests(runner: TaskRunner, thinking: bool = False, model: str = "Qwen/Qwen3-14B", translator: StandardTranslator | NatLangTranslator = StandardTranslator()):
    results_novelty = await runner.run(TaskType.NOVELTY, num_tasks=50)
    num_correct_novelty = sum([result.success for result in results_novelty])
    print(f"Novelty: {num_correct_novelty}/{len(results_novelty)} correct")
    log_results(results_novelty, model, TaskType.NOVELTY, thinking, translator)


def log_results(results: list[Result], model: str, task_type: TaskType, thinking: bool = False, translator: StandardTranslator | NatLangTranslator = StandardTranslator()):
    thinking_dir = "/thinking/" if thinking else "/non_thinking/"
    results_dir = "results/" + model + thinking_dir + translator.type
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(os.path.join(results_dir, f"results_{task_type.name.lower()}.json"), "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=4)


async def run_task_tests(runner: TaskRunner, thinking: bool = False, model: str = "Qwen/Qwen3-14B", translator: StandardTranslator | NatLangTranslator = StandardTranslator()):
    to_run = [
        run_action_gen_tests(runner, thinking=thinking, model=model, translator=translator),
        run_action_gen_single_tests(runner, thinking=thinking, model=model, translator=translator),
        run_succ_state_mapping_tests(runner, thinking=thinking, model=model, translator=translator),
        run_succ_state_mapping_separate_tests(runner, thinking=thinking, model=model, translator=translator),
        run_goal_state_verification_tests(runner, thinking=thinking, model=model, translator=translator)
    ]
    await asyncio.gather(*to_run)

async def run_tot_tests(engine: Engine, translator: StandardTranslator | NatLangTranslator, context: str, config_name: str, novelty: bool = None):
    runner = TestRunner(engine, translator, context=context, logistics=True)
    config=Configuration.from_file(config_name)
    if novelty is not None:
        config.novelty = novelty
    parallel = "novelty" not in config_name
    if novelty is not None and novelty:
        parallel = False
    print(f"Running TOT tests with config: {config_name}, parallel: {parallel}, novelty: {novelty}")
    output_dir = f"test_tots_logistics/{config_name}/"
    if novelty:
        output_dir += "novelty/"
    else:
        output_dir += "non_novelty/"
    start_time = datetime.now()
    print(f"{config_name} Start time: {start_time}")
    results, summary = await runner.run_logistics(config, num_tasks=50, write_tot=True, branch_factor=2, max_depth=8, parallel_tot=parallel, parallel_tasks=True, breadth_first=False, output_dir=output_dir)
    end_time = datetime.now()
    print(f"{config_name} End time: {end_time}")
    print(f"{config_name} Duration: {end_time - start_time}")
    print(f"{config_name} Results: {results}")
    print(f"{config_name} Summary: {summary}")
    
    # save results and summary to file
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    tag = "all50_logistics"
    if novelty:
        tag += "_novelty"
    results_path = f"/work/rleap1/leon.hamm/bachelor-thesis-dev/novelty_tot/tot_results/{config_name}/"
    results_path += tag
    results_path += f"_{datetime_str}.json"
    if not os.path.exists(os.path.dirname(results_path)):
        os.makedirs(os.path.dirname(results_path))
    with open(results_path, "w") as f:
        output_data = {
            "results": [result.model_dump() for result in results],
            "summary": summary,
            "duration": str(end_time - start_time),
        }
        json.dump(output_data, f, indent=4)

async def main():
    """
    Designed do be run using run_cluster_batch.sh
    """
    vllm_hosts = os.getenv("VLLM_NODES")

    vllm_hosts = vllm_hosts.split(",") if vllm_hosts else []
    if not vllm_hosts:
        print("No VLLM nodes found. Exiting.")
        return

    vllm_urls = [f"http://{host}:8000/v1" for host in vllm_hosts]

    client_vllms = [
        LlmApiClient(
            api_key="None",
            base_url=url,
            model="Qwen/Qwen3-14B",
            max_parallel_requests=250,
        ) for url in vllm_urls
    ]

    model = "Qwen/Qwen3-14B"
    os.environ["VAL"] = "/work/rleap1/leon.hamm/bachelor-thesis-dev/VAL/validate"   
    context = "I have to plan logistics to transport packages within cities via trucks and between cities via airplanes. Locations within a city are directly connected (trucks can move between any two such locations), and so are the cities. In each city there is exactly one truck and each city has one location that serves as an airport.\nHere are the actions that can be performed:\n\nLoad a package into a truck. \nLoad a package into an airplane.\nUnload a package from a truck. \nUnload a package from an airplane. \nDrive a truck from one location to another location. \nFly an airplane from one city to another city.\n\nThe following are the restrictions on the actions:\nA package can be loaded into a truck only if the package and the truck are in the same location.\nOnce a package is loaded into a truck, the package is not at the location and is in the truck.   \nA package can be loaded into an airplane only if the package and the airplane are in the same location.\nOnce a package is loaded into an airplane, the package is not at the location and is in the airplane.\nA package can be unloaded from a truck only if the package is in the truck.\nOnce a package is unloaded from a truck, the package is not in the truck and is at the location of the truck.\nA package can be unloaded from an airplane only if the package in the airplane.\nOnce a package is unloaded from an airplane, the package is not in the airplane and is at the location of the airplane.   \nA truck can be driven from one location to another if the truck is at the from-location and both from-location and to-location are locations in the same city.\nOnce a truck is driven from one location to another, it is not at the from-location and is at the to-location.\nAn airplane can be flown from one city to another if the from-location and the to-location are airports and the airplane is at the from-location.\nOnce an airplane is flown from one city to another the airplane is not at the from-location and is at the to-location."
    context += "\nYou must always specify the location of every loading and unloading action.\n"
    # example = "For example, if I have a blue block on the table and a red block on top of it (and clear), then I may NOT pick up the red block because it is on top of another block. This means I must instead unstack it."
    # context += "\n" + example
    max_parallel_requests = 1000

    tasks = []

    translator = NatLangTranslator()
    novelty = False

    config_name_1 = "explicit_state_action_2_reasoning"
    thinking_1 = False
    engine_1 = Engine(client_vllms, chat=True, model=model, thinking=thinking_1, remove_thinking=True, max_parallel_requests=max_parallel_requests)

    tasks.append(run_tot_tests(engine_1, translator=translator, context=context, config_name=config_name_1, novelty=novelty))

    config_name_2 = "explicit_state_action_2_thinking"
    thinking_2 = True
    engine_2 = Engine(client_vllms, chat=True, model=model, thinking=thinking_2, remove_thinking=True, max_parallel_requests=max_parallel_requests)

    tasks.append(run_tot_tests(engine_2, translator=translator, context=context, config_name=config_name_2, novelty=novelty))

    config_name_3 = "base_simple"
    thinking_3 = False
    engine_3 = Engine(client_vllms, chat=True, model=model, thinking=thinking_3, remove_thinking=True, max_parallel_requests=max_parallel_requests)

    tasks.append(run_tot_tests(engine_3, translator=translator, context=context, config_name=config_name_3, novelty=novelty))
    
    config_name_4 = "base_reasoning"
    thinking_4 = False
    engine_4 = Engine(client_vllms, chat=True, model=model, thinking=thinking_4, remove_thinking=True, max_parallel_requests=max_parallel_requests)

    tasks.append(run_tot_tests(engine_4, translator=translator, context=context, config_name=config_name_4, novelty=novelty))

    config_name_5 = "base_thinking"
    thinking_5 = True
    engine_5 = Engine(client_vllms, chat=True, model=model, thinking=thinking_5, remove_thinking=True, max_parallel_requests=max_parallel_requests)

    tasks.append(run_tot_tests(engine_5, translator=translator, context=context, config_name=config_name_5, novelty=novelty))

    # run with novelty

    novelty_2 = True

    config_name_novelty_6 = "explicit_state_action_2_reasoning"
    thinking_6 = False
    engine_6 = Engine(client_vllms, chat=True, model=model, thinking=thinking_6, remove_thinking=True, max_parallel_requests=max_parallel_requests)

    tasks.append(run_tot_tests(engine_6, translator=translator, context=context, config_name=config_name_novelty_6, novelty=novelty_2))

    config_name_novelty_7 = "explicit_state_action_2_thinking"
    thinking_7 = True
    engine_7 = Engine(client_vllms, chat=True, model=model, thinking=thinking_7, remove_thinking=True, max_parallel_requests=max_parallel_requests)

    tasks.append(run_tot_tests(engine_7, translator=translator, context=context, config_name=config_name_novelty_7, novelty=novelty_2))

    config_name_novelty_8 = "base_simple"
    thinking_8 = False
    engine_8 = Engine(client_vllms, chat=True, model=model, thinking=thinking_8, remove_thinking=True, max_parallel_requests=max_parallel_requests)

    tasks.append(run_tot_tests(engine_8, translator=translator, context=context, config_name=config_name_novelty_8, novelty=novelty_2))

    config_name_novelty_9 = "base_reasoning"
    thinking_9 = False
    engine_9 = Engine(client_vllms, chat=True, model=model, thinking=thinking_9, remove_thinking=True, max_parallel_requests=max_parallel_requests)

    tasks.append(run_tot_tests(engine_9, translator=translator, context=context, config_name=config_name_novelty_9, novelty=novelty_2))

    config_name_novelty_10 = "base_thinking"
    thinking_10 = True
    engine_10 = Engine(client_vllms, chat=True, model=model, thinking=thinking_10, remove_thinking=True, max_parallel_requests=max_parallel_requests)

    tasks.append(run_tot_tests(engine_10, translator=translator, context=context, config_name=config_name_novelty_10, novelty=novelty_2))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
