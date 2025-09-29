import asyncio
import os

from llm_engine import LlmApiClient, Engine

def setup_vllms():
    vllm_hosts = os.getenv("VLLM_NODES")

    vllm_hosts = vllm_hosts.split(",") if vllm_hosts else []
    if not vllm_hosts:
        print("No VLLM nodes found. Exiting.")
        return []

    vllm_urls = [f"http://{host}:8000/v1" for host in vllm_hosts]

    client_vllms = [
        LlmApiClient(
            api_key="None",
            base_url=url,
            model="Qwen/Qwen3-14B",
            max_parallel_requests=250,
        ) for url in vllm_urls
    ]
    return client_vllms

async def main():
    # start interactive chat with vllm model
    client_vllms: list[LlmApiClient] = setup_vllms()

    print("Starting interactive chat with VLLM models. Type 'exit' or 'quit' to stop.")

    thinking_input = input("Enable thinking mode? (y/n): ").strip().lower()
    if thinking_input in ["exit", "quit"]:
        print("Exiting.")
        return
    if thinking_input not in ['y', 'n']:
        print("Invalid input. Please enter 'y' or 'n'. Exiting.")
        return
    thinking_mode = thinking_input == 'y'
    print(f"Thinking mode {'enabled' if thinking_mode else 'disabled'}.")

    if thinking_mode:
        remove_thinking = input("Remove thinking from final output? (y/n): ").strip().lower()
        if remove_thinking in ["exit", "quit"]:
            print("Exiting.")
            return
        if remove_thinking not in ['y', 'n']:
            print("Invalid input. Please enter 'y' or 'n'. Exiting.")
            return
        remove_thinking_mode = remove_thinking == 'y'
        print(f"Remove thinking from final output {'enabled' if remove_thinking_mode else 'disabled'}.")
    else:
        remove_thinking_mode = False

    engine = Engine(client=client_vllms, chat=True, model="Qwen/Qwen3-14B", thinking=thinking_mode, remove_thinking=remove_thinking_mode)

    while True:
        user_input = input("Input:\n").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        response = await engine.query(user_input)
        print(f"Response:\n{response}")

if __name__ == "__main__":
    asyncio.run(main())
