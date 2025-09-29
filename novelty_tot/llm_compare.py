import re
import logging


llm_compare_logger = logging.getLogger("nov_tot." + __name__)

def is_yes(response: str) -> bool:
    """
    Check if the response is 'yes'.
    """
    response = response.lower()
    if "<think>" in response:
        # remove everything between <think> and </think>
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    # remove the new lines
    response = response.replace('\n', '').strip()
    if "[response]" in response:
        # remove the [RESPONSE] prefix and anything before it
        response = response.split("[response]", 1)[-1].strip()
    yes_strings = ["yes", "yes."]
    no_strings = ["no", "no."]
    if response in yes_strings:
        return True
    elif response in no_strings:
        return False
    else:
        llm_compare_logger.warning(f"Unexpected response: {response}")
        if "yes" in response:
            return True
        return False

async def check_answer_simple(query_fn: callable, answer: str, ground_truth: str, **query_kwargs) -> bool:
    """
    Check if the answer is the same as the ground truth using an LLM.
    """
    query = f"Is the answer -START-\n'{answer}'\n-END- correct for the ground truth -START-\n'{ground_truth}'\n-END-? Respond with 'yes' or 'no'. Do not add any other text."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Answer check query: {query}")
    llm_compare_logger.debug(f"Answer check response: {response}")
    return is_yes(response)


async def check_answer(query_fn: callable, answer: str, ground_truth: str, **query_kwargs) -> bool:
    """
    Check if the answer is the same as the ground truth using an LLM.
    """
    query = f"Is the answer -START-\n'{answer}'\n-END- correct for the ground truth -START-\n'{ground_truth}'\n-END-? Redundant, uncessary and additional information is fine as long as it is not incorrect. The phrasing does not matter as long as the meaning is correct. Output your reasoning and then [OUTPUT] followed by yes or no. Do not add any other text."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Answer check query: {query}")
    llm_compare_logger.debug(f"Answer check response: {response}")
    response = response.lower().split("[output]")[-1].strip()  # remove everything before [OUTPUT]
    return is_yes(response)


async def check_answer_action_simple(query_fn: callable, answer: str, ground_truth: str, **query_kwargs) -> bool:
    """
    Check if the answer is the same as the ground truth using an LLM.
    """
    query = f"Is the answer -START-\n'{answer}'\n-END- correct for the ground truth -START-\n'{ground_truth}'\n-END-? Redundant, unnecessary and additional information is fine as long as it is not incorrect. The phrasing does not matter as long as the meaning is correct. However, picking up and unstacking is not the same. Order also matters. Respond with 'yes' or 'no'. Do not add any other text."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Answer check query: {query}")
    llm_compare_logger.debug(f"Answer check response: {response}")
    return is_yes(response)


async def check_answer_action(query_fn: callable, answer: str, ground_truth: str, **query_kwargs) -> bool:
    """
    Check if the answer is the same as the ground truth using an LLM.
    """
    query = f"Is the answer -START-\n'{answer}'\n-END- correct for the ground truth -START-\n'{ground_truth}'\n-END-? Redundant, unnecessary and additional information is fine as long as it is not incorrect. The phrasing does not matter as long as the meaning is correct. However, picking up and unstacking is not the same. Order also matters. Output your reasoning and then [OUTPUT] followed by yes or no. Do not add any other text."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Answer check query: {query}")
    llm_compare_logger.debug(f"Answer check response: {response}")
    response = response.lower().split("[output]")[-1].strip()  # remove everything before [OUTPUT]
    return is_yes(response)

async def check_answer_for_question_simple(query_fn: callable, answer: str, ground_truth: str, question: str, **query_kwargs) -> bool:
    """
    Check if the answer is the same as the ground truth using an LLM.
    """
    query = f"Is the answer -START-\n'{answer}'\n-END- correct for the question -START-\n'{question}'\n-END- and the ground truth -START-\n'{ground_truth}'\n-END-? Redundant, unnecessary and additional information is fine as long as it is not incorrect. The phrasing does not matter as long as the meaning is correct. Respond with 'yes' or 'no'. Do not add any other text."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Answer check query: {query}")
    llm_compare_logger.debug(f"Answer check response: {response}")
    return is_yes(response)

async def check_answer_for_question(query_fn: callable, answer: str, ground_truth: str, question: str, **query_kwargs) -> bool:
    """
    Check if the answer is the same as the ground truth using an LLM.
    """
    query = f"Is the answer -START-\n'{answer}'\n-END- correct for the question -START-\n'{question}'\n-END- and the ground truth -START-\n'{ground_truth}'\n-END-? Redundant, unnecessary and additional information is fine as long as it is not incorrect. The phrasing does not matter as long as the meaning is correct. Output your reasoning and then [OUTPUT] followed by yes or no. Do not add any other text."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Answer check query: {query}")
    llm_compare_logger.debug(f"Answer check response: {response}")
    response = response.lower().split("[output]")[-1].strip()  # remove everything before [OUTPUT]
    return is_yes(response)

async def check_answer_for_question_action_simple(query_fn: callable, answer: str, ground_truth: str, question: str, **query_kwargs) -> bool:
    """
    Check if the answer is the same as the ground truth using an LLM.
    """
    query = f"Is the answer -START-\n'{answer}'\n-END- correct for the question -START-\n'{question}'\n-END- and the ground truth -START-\n'{ground_truth}'\n-END-? Redundant, unnecessary and additional information is fine as long as it is not incorrect. The phrasing does not matter as long as the meaning is correct. However, picking up and unstacking is not the same. 'from' and 'from on top of' are interchangeable. Example: 'unstack b from a' is equivalent to '(unstack b a)' and 'unstack block b from on top of block a' but not '(unstack a b)'. Respond with 'yes' or 'no'. Do not add any other text."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Answer check query: {query}")
    llm_compare_logger.debug(f"Answer check response: {response}")
    return is_yes(response)

async def check_answer_for_question_action(query_fn: callable, answer: str, ground_truth: str, question: str, **query_kwargs) -> bool:
    """
    Check if the answer is the same as the ground truth using an LLM.
    """
    query = f"Is the answer -START-\n'{answer}'\n-END- correct for the question -START-\n'{question}'\n-END- and the ground truth -START-\n'{ground_truth}'\n-END-? Redundant, unnecessary and additional information is fine as long as it is not incorrect. The phrasing does not matter as long as the meaning is correct. However, picking up and unstacking is not the same. 'from' and 'from on top of' are interchangeable. Example: 'Unstack b from a' is equivalent to '(unstack b a)' and 'unstack block b from on top of block a' but not '(unstack a b)'. Output your reasoning and then [OUTPUT] followed by yes or no. Do not add any other text."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Answer check query: {query}")
    llm_compare_logger.debug(f"Answer check response: {response}")
    response = response.lower().split("[output]")[-1].strip()  # remove everything before [OUTPUT]
    return is_yes(response)

async def extract_answer_for_question(query_fn: callable, answer: str, question: str, **query_kwargs) -> str:
    """
    Extract the answer for the question from the LLM response.
    """
    query = f"Extract the answer for the question -START-\n'{question}'\n-END- from the answer -START-\n'{answer}'\n-END-. Remove all reasoning and unnecessary text. Do not add any other text."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Answer extraction query: {query}")
    llm_compare_logger.debug(f"Answer extraction response: {response}")
    return response.strip()

async def check_answer_is_one_of_simple(query_fn: callable, answer: str, options: list[str], **query_kwargs) -> bool:
    """
    Check if the answer is one of the options using an LLM.
    """
    options_str = ", ".join(f"'{opt}'" for opt in options)
    query = f"Is the answer -START-\n'{answer}'\n-END- one of the following options: {options_str}? Respond with 'yes' or 'no'. Do not add any other text."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Answer check query: {query}")
    llm_compare_logger.debug(f"Answer check response: {response}")
    return is_yes(response)

async def check_answer_is_one_of(query_fn: callable, answer: str, options: list[str], **query_kwargs) -> bool:
    """
    Check if the answer is one of the options using an LLM.
    """
    options_str = ", ".join(f"'{opt}'" for opt in options)
    query = f"Is the answer -START-\n'{answer}'\n-END- one of the following options: -START-\n'{options_str}'\n-END-? Redundant, unnecessary and additional information is fine as long as it is not incorrect. The phrasing does not matter as long as the meaning is correct. Output your reasoning and then [OUTPUT] followed by yes or no. Do not add any other text."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Answer check query: {query}")
    llm_compare_logger.debug(f"Answer check response: {response}")
    response = response.lower().split("[output]")[-1].strip()  # remove everything before [OUTPUT]
    return is_yes(response)

async def check_answer_is_one_of_action(query_fn: callable, answer: str, options: list[str], **query_kwargs) -> bool:
    """
    Check if the answer is one of the options using an LLM.
    """
    options_str = ", ".join(f"'{opt}'" for opt in options)
    query = f"Is the answer -START-\n'{answer}'\n-END- one of the following options: -START-\n'{options_str}'\n-END-? Redundant, unnecessary and additional information is fine as long as it is not incorrect. The phrasing does not matter as long as the meaning is correct. However, picking up and unstacking is not the same. 'from' and 'from on top of' are interchangeable. Example: 'unstack b from a' is equivalent to '(unstack b a)' and 'unstack block b from on top of block a' but not '(unstack a b)'. Output your reasoning and then [OUTPUT] followed by yes or no. Do not add any other text."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Answer check query: {query}")
    llm_compare_logger.debug(f"Answer check response: {response}")
    response = response.lower().split("[output]")[-1].strip()  # remove everything before [OUTPUT]
    return is_yes(response)

async def extract_plan_for_question(query_fn: callable, answer: str, question: str, **query_kwargs) -> str:
    """
    Extract the plan for the question from the LLM response.
    """
    query = f"Extract the plan for the question -START-\n'{question}'\n-END- from the answer -START-\n'{answer}'\n-END-. Remove all reasoning and unnecessary text. Do not add any other text."
    query += " The plan should be in the form of a list of actions, each action on a new line. Here is an example of such a plan that is unrelated to the current one:\n" \
              "unstack the black block from on top of the red block,\n" \
              "put down the yellow block,\n" \
              "pick up the green block,\n" \
              "stack the grey block on top of the white block,\n" \
              "Now, extract the plan from the answer."
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Plan extraction query: {query}")
    llm_compare_logger.debug(f"Plan extraction response: {response}")
    return response.strip()

async def does_answer_solve_problem(query_fn: callable, answer: str, problem_statement: str, **query_kwargs) -> bool:
    """
    Check if the answer solves the problem statement using an LLM.
    """
    query = f"Does the following answer correctly and completely solve the problem statement? Your response must end with '[RESPONSE] yes' or '[RESPONSE] no'. Problem statement: {problem_statement}\nAnswer: {answer}"
    response = await query_fn(query, **query_kwargs)
    llm_compare_logger.debug(f"Problem solving check query: {query}")
    llm_compare_logger.debug(f"Problem solving check response: {response}")
    return is_yes(response)
