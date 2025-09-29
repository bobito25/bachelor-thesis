import random
import asyncio
from openai import AsyncOpenAI, BadRequestError, APITimeoutError
from openai.types.chat import ChatCompletion


class EngineResponse(str):
    """
    Class for LLM response.
    """
    def __new__(cls, value: str):
        str_obj =  super().__new__(cls, value)
        str_obj._token_usage = None
        str_obj._finish_reason = None
        str_obj._thinking = None
        return str_obj

    @property
    def token_usage(self) -> int:
        """
        Get token usage of response.
        """
        if self._token_usage is not None:
            return self._token_usage
        else:
            raise AttributeError("Token usage not set.")

    @token_usage.setter
    def token_usage(self, value: int) -> None:
        """
        Set token usage of response.
        """
        self._token_usage = value

    @property
    def finish_reason(self) -> str:
        """
        Get finish reason of response.
        """
        if self._finish_reason is not None:
            return self._finish_reason
        else:
            raise AttributeError("Finish reason not set.")
    
    @finish_reason.setter
    def finish_reason(self, value: str) -> None:
        """
        Set finish reason of response.
        """
        self._finish_reason = value
    
    @property
    def thinking(self) -> str:
        """
        Get thinking part of response.
        """
        if self._thinking is not None:
            return self._thinking
        else:
            raise AttributeError("Thinking not set.")
    
    @thinking.setter
    def thinking(self, value: str) -> None:
        """
        Set thinking part of response.
        """
        self._thinking = value


class LlmApiClient(AsyncOpenAI):
    """
    Class for LLM API.
    """
    def __init__(self, *args, max_parallel_requests: int = 240, model: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_parallel_requests = max_parallel_requests
        if model is not None:
            self.model = model
        else:
            raise ValueError("Model must be specified.")
        self.semaphore = asyncio.Semaphore(self.max_parallel_requests)

    async def aquery(self, query: str, max_completion_tokens: int = 1000, temperature: float = 1.0, thinking: bool = False, frequency_penalty: float = 0.0) -> ChatCompletion:
        """
        Asynchronous query to the LLM API.
        """
        async with self.semaphore:
            response = await self.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                extra_body= {"chat_template_kwargs": {"enable_thinking": False}} if not thinking else None,
                frequency_penalty=frequency_penalty
            )
            return response


def has_repeated_pattern(s: str) -> bool:
    return s in (s + s)[1:-1]


def repeating_suffix(s: str, min_length: int = 50):
    n = len(s)
    for size in range(min_length, n // 2 + 1):
        suffix = s[-size:]
        if has_repeated_pattern(suffix):
            return suffix
    return None


def remove_repeating_suffix(s: str, suffix: str, keep_last: bool = False) -> str:
    if suffix is None:
        return s
    max_removals = 200
    while s.endswith(suffix) and max_removals > 0:
        s = s[:-len(suffix)]
        max_removals -= 1
    if keep_last:
        s += suffix
    return s


class Engine():
    """
    Class for interacting with LLM.
    """
    def __init__(self, client: LlmApiClient | list[LlmApiClient], model: str = None, max_tokens: int = 22000, chat: bool = False, thinking: bool = False, remove_thinking: bool = True, max_parallel_requests: int = 10, temperature: float = 0.6):
        """
        Initialize Engine object.
        """
        self.max_tokens = max_tokens
        if model is None:
            if chat:
                self.model = "gpt-4o-mini"
            else:
                self.model = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
        else:
            self.model = model
        self.chat = chat
        if not isinstance(client, list):
            client = [client]
        else:
            if not all(isinstance(c, LlmApiClient) for c in client):
                raise ValueError("All clients must be instances of LlmApiClient.")
        if len(client) == 0:
            raise ValueError("At least one client must be provided.")
        self.clients = client
        self.thinking = thinking
        self.remove_thinking = remove_thinking
        self.temperature = temperature
        
        # Keep instance-level semaphore for backward compatibility
        self.semaphore = asyncio.Semaphore(max_parallel_requests)

        self._stats_lock = asyncio.Lock()
        self.total_token_usage = 0
        self.num_early_stops = 0
        self.num_max_context_exceeded = 0
        self.num_max_retries_exceeded = 0
        self.num_retries = 0
        self.num_truncated_queries = 0

    async def query(self, query: str, max_tokens: int = None, retries: int = 2, thinking_override: bool | None = None, frequency_penalty: float = 0.0) -> EngineResponse:
        """
        Get solution using Engine.
        """
        if retries < 0:
            raise ValueError("No more retries left.")
        if thinking_override is not None:
            if thinking_override:
                thinking = True
            else:
                thinking = False
        else:
            thinking = self.thinking
        if max_tokens is None:
            max_tokens = self.max_tokens
        max_query_length_chars = 40000
        if len(query) > max_query_length_chars:
            print(f"Query is too long, truncating to ~{max_query_length_chars} characters.")
            print(f"Query length: {len(query)}")
            print(f"Query:\n{query[:200]}...{query[-200:]}")
            chars_to_keep = max_query_length_chars - 9  # subtract length of " [...] "
            half_keep = chars_to_keep // 2
            query = query[:half_keep] + " [...] " + query[-half_keep:]
            async with self._stats_lock:
                self.num_truncated_queries += 1
        # Use both client and instance semaphores
        async with self.semaphore:
            # try to get non blocked client
            free_clients = [c for c in self.clients if not c.semaphore.locked()]
            if not free_clients:
                free_clients = self.clients
            client = random.choice(free_clients)
            try:
                if self.chat:
                    response = await client.aquery(
                        query=query,
                        max_completion_tokens=max_tokens,
                        temperature=self.temperature,
                        thinking=thinking,
                        frequency_penalty=frequency_penalty
                    )
                    response_text = response.choices[0].message.content.strip()
                else:
                    raise NotImplementedError("Non-chat mode is not implemented in this version.")
            except APITimeoutError as e:
                # retry on timeout
                print("Request timed out (APITimeoutError). Retrying...")
                async with self._stats_lock:
                    self.num_retries += 1
                if retries > 0:
                    await asyncio.sleep(2)  # wait before retrying
                    return await self.query(query, max_tokens, retries - 1, thinking_override, frequency_penalty)
                else:
                    print("No more retries left after timeout.")
                    async with self._stats_lock:
                        self.num_max_retries_exceeded += 1
                    raise e
            except BadRequestError as e:
                if "maximum context length" in str(e):
                    print("maximum context length exceeded!")
                    response_text = "[context length exceeded]"
                    async with self._stats_lock:
                        self.num_max_context_exceeded += 1
                else:
                    raise e
            #print(f"LLM query: {query}")
            #print(f"LLM response: {response}")
            if not thinking and ("<think>" in response_text or "</think>" in response_text):
                if retries > 0:
                    print("Thinking is not enabled, but response contains thinking tags. Retrying...")
                    return await self.query(query, max_tokens, retries - 1, thinking_override, frequency_penalty)
                print("-> query:\n", query)
                print("-> response:\n", response_text)
                raise ValueError("Thinking is not enabled, but response contains thinking tags.")
            thinking_part = None
            if thinking and "<think>" in response_text:
                # check if thinking was not completed because of token_limit
                if "<think>" in response_text and "</think>" not in response_text:
                    response_text += "</think>"
                thinking_part = response_text.split("<think>")[1].split("</think>")[0].strip()
                if self.remove_thinking:
                    # remove everything until </think>
                    response_text = response_text.split("</think>")[-1].strip()
            if response_text == "":
                if retries > 0:
                    print("Empty response received, retrying...")
                    async with self._stats_lock:
                        self.num_retries += 1
                    return await self.query(query, max_tokens, retries - 1, thinking_override, frequency_penalty)
                else:
                    print("Empty response received, no more retries left.")
                    response_text = "[empty response]"
                    async with self._stats_lock:
                        self.num_max_retries_exceeded += 1
            self.total_token_usage += response.usage.total_tokens
            finish_reason = response.choices[0].finish_reason
            if finish_reason != "stop":
                print(f"Completion stopped early (finish_reason: {finish_reason})")
                print("-> query:\n", query)
                print("-> response:\n", response_text)
                async with self._stats_lock:
                    self.num_early_stops += 1
                if finish_reason == "length":
                    # check for repititive patterns at the end of the response
                    suffix = repeating_suffix(response_text, min_length=10)
                    if suffix is not None:
                        print(f"Response ends with a repeating pattern: '{suffix}'. Retrying...")
                        if retries > 0:
                            async with self._stats_lock:
                                self.num_retries += 1
                            return await self.query(query, max_tokens, retries - 1, thinking_override, frequency_penalty=frequency_penalty + 0.1)
                        else:
                            print("No more retries left.")
                            response_text = remove_repeating_suffix(response_text, suffix, keep_last=True)
            engine_response = EngineResponse(response_text)
            engine_response.token_usage = response.usage.total_tokens
            engine_response.finish_reason = finish_reason
            if thinking and thinking_part is not None:
                engine_response.thinking = thinking_part
            return engine_response
