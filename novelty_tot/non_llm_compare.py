from pddl_translation.translator import Translator
from pddl_translation.pddl_translation import action_string_to_pddl_tuple_blocksworld


def check_action_one_of(action: str, options: list[str], translator: Translator, verbose: bool = False) -> bool:
    """
    Check if the action is one of the options.
    """
    action_tuple = action_string_to_pddl_tuple_blocksworld(action, translator, verbose)
    if action_tuple is None:
        return False
    for option in options:
        option_tuple = action_string_to_pddl_tuple_blocksworld(option, translator, verbose)
        if option_tuple is not None and action_tuple == option_tuple:
            return True
    return False


def check_action_equals(action: str, option: str, translator: Translator, verbose: bool = False) -> bool:
    """
    Check if the action is equal to the option.
    """
    action_tuple = action_string_to_pddl_tuple_blocksworld(action, translator, verbose)
    if verbose:
        print(f"Action tuple: {action_tuple}")
    option_tuple = action_string_to_pddl_tuple_blocksworld(option, translator, verbose)
    if verbose:
        print(f"Option tuple: {option_tuple}")
    if action_tuple is None or option_tuple is None:
        return False
    return action_tuple == option_tuple
