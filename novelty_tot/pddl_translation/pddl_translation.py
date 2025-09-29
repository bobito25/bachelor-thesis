from pydantic import BaseModel


class PDDLTranslationConfig(BaseModel):
    """
    Configuration for PDDL translation.
    """
    raw_actions: list[str]  # list of raw action names
    num_obj_action: dict[str, int]  # mapping from action names to number of objects / params
    actions: dict[str, str]  # pddl action names to formattable action strings
    predicates: dict[str, str]  # pddl predicate names to formattable predicate strings
    predicate_mapping: dict[str, str]  # mapping from predicate names to their string representations
    encoded_objects: dict[int, str]  # mapping from lifted object indices to pddl object names
    encoded_objects_str: dict[str, str]  # mapping from pddl object names to string object representations


def decode_object_idx(obj: str, config: PDDLTranslationConfig) -> str:
    """
    Decode an lifted object idx to its string representation.
    """
    if obj not in config.encoded_objects:
        raise ValueError(f"Object {obj} not found as encoded object in config.")
    if config.encoded_objects[obj] not in config.encoded_objects_str:
        raise ValueError(f"Object {config.encoded_objects[obj]} not found as encoded object string in config.")
    return config.encoded_objects_str[config.encoded_objects[obj]]


def action_to_string(config: PDDLTranslationConfig, action: tuple[str, tuple]) -> str:
    """
    Convert a pddl action to a string representation.
    """
    action_name = action[0]
    params = action[1]
    if action_name not in config.actions:
        raise ValueError(f"Action '{action_name}' not found in config. Available actions: {list(config.actions.keys())}")
    action_str = config.actions[action_name]
    objects = [decode_object_idx(obj, config) for obj in params]
    action_str = action_str.format(*objects)
    return action_str


def atom_to_string(config: PDDLTranslationConfig, atom: tuple[str, tuple]) -> str:
    """
    Convert a pddl atom to a string representation.
    """
    predicate, params = atom
    if predicate not in config.predicates:
        raise ValueError(f"Predicate {predicate} not found in config.")
    predicate_str = config.predicates[predicate]
    objects = [decode_object_idx(obj, config) for obj in params]
    return predicate_str.format(*objects)


def action_pddl_string_to_tuple(action_str: str, config: PDDLTranslationConfig) -> tuple[str, tuple]:
    """
    Convert a string representation of a pddl action to a tuple.
    """
    # Extract action name and parameters from the string
    action_name, *params = action_str.strip("()").split(" ")
    if action_name not in config.actions:
        raise ValueError(f"Action {action_name} not found in config.")
    # Map parameter strings to their encoded object representations and then to indices
    params = [config.encoded_objects_str.get(param, param) for param in params]
    params = [config.encoded_objects.get(param, param) for param in params]
    return action_name, tuple(params)


def decode_action_string(action_str: str, config: PDDLTranslationConfig) -> str:
    """
    Decode encoded objects in an action string to their string representations.
    """
    action_name, *params = action_str.strip("()").split(" ")
    if action_name not in config.actions:
        raise ValueError(f"Action {action_name} not found in config.")
    action_str = config.actions[action_name]
    for i, param in enumerate(params):
        if param not in config.encoded_objects_str:
            raise ValueError(f"Object {param} not found as encoded object in config.")
        params[i] = config.encoded_objects_str[param]
    action_str = action_str.format(*params)
    return action_str


def remove_non_letters(s: str, exceptions: list = [], allow_digits: bool = False) -> str:
    """
    Remove all non-letter characters from a string.
    """
    return ''.join(c for c in s if c.isalpha() or c.isspace() or c in exceptions or (allow_digits and c.isdigit())).strip()


def action_string_to_pddl_tuple_blocksworld(action: str, translator, verbose: bool = False) -> tuple[str, tuple]:
    """
    Convert an action string of plan text to a tuple in pddl form.
    """
    action = action.lower().strip()
    # get action predicate
    actions = [["pick-up", "pick up", "pickup"], ["put-down", "put down", "putdown"], ["unstack", "un-stack"], ["stack"]]
    action_pred = None
    for action_synonyms in actions:
        if action_pred is not None:
            break
        for action_synonym in action_synonyms:
            if action_synonym in action:
                action_pred = action_synonyms[0]
                break
    if verbose:
        print(f"Action predicate: {action_pred}")
    if action_pred is None:
        return None
    # get args
    num_args = 2 if action_pred in ["stack", "unstack"] else 1
    if translator.type == "nat_lang":
        possible_args = list(translator.config.encoded_objects_str.values())
        possible_args = [[arg, arg.replace("block", "").strip()] for arg in possible_args]
    else:
        possible_args = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
        possible_args = [[arg] for arg in possible_args]
    args = []
    # replace parentheses with spaces
    action = action.replace("(", " ").replace(")", " ")
    # split by spaces
    words = action.split(" ")
    if verbose:
        print(f"Words: {words}")
    # remove all delimiters and unnecessary chars
    words = [remove_non_letters(word) for word in words]
    if verbose:
        print(f"Words after removing delimiters: {words}")
    for idx, word in enumerate(words):
        for arg_idx, possible_arg_synonyms in enumerate(possible_args):
            for possible_arg_synonym in possible_arg_synonyms:
                if possible_arg_synonym == word:
                    position = idx
                    arg_name = chr(97 + arg_idx)  # 97 is 'a'
                    args.append((arg_name, position))
    if verbose:
        print(f"Args: {args}")
    if len(args) != num_args:
        return None
    sorted_args = sorted(args, key=lambda x: x[1])
    sorted_args = [arg[0] for arg in sorted_args]
    return (action_pred, tuple(sorted_args))


def text_to_pddl_plan_blocksworld(text: str, translator, verbose: bool = False) -> str:
    """
    Converts blocksworld plan in plain text to PDDL plan format.
    """
    plan = []
    text = text.lower().strip()
    splitters = [".", ";"]
    for splitter in splitters:
        text = text.replace(splitter, "\n")
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if verbose:
            print(f"Processing line: {line}")
        action = action_string_to_pddl_tuple_blocksworld(line, translator, verbose)
        if action is None:
            continue
        action_str = f"({action[0]} {' '.join(action[1])})"
        plan.append(action_str)
    return "\n".join(plan)


def action_string_to_pddl_tuple_logistics(action: str, translator, verbose: bool = False) -> tuple[str, tuple]:
    """
    Convert an action string of plan text to a tuple in pddl form for logistics domain.
    Only works for nat lang for now.
    """
    action = action.lower().strip()
    # get action predicate
    action_pred = None
    # actions = ["load-truck", "load-airplane", "unload-truck", "unload-airplane", "drive-truck", "fly-airplane"]
    if "truck" in action and "airplane" in action:
        return None
    elif "truck" in action:
        # truck action
        if "unload" in action:
            action_pred = "unload-truck"
        elif "load" in action:
            action_pred = "load-truck"
        elif "drive" in action:
            action_pred = "drive-truck"
        else:
            return None
    elif "airplane" in action:
        # airplane action
        if "unload" in action:
            action_pred = "unload-airplane"
        elif "load" in action:
            action_pred = "load-airplane"
        elif "fly" in action:
            action_pred = "fly-airplane"
        else:
            return None
    elif "drive" in action:
        action_pred = "drive-truck"
    elif "fly" in action:
        action_pred = "fly-airplane"
    else:
        return None
    if verbose:
        print(f"Action predicate: {action_pred}")
    # get args
    num_args = 4 if action_pred == "drive-truck" else 3
    if translator.type == "nat_lang":
        possible_args = list(translator.config.encoded_objects_str.values())
    else:
        raise NotImplementedError("Only nat_lang translator is supported for logistics domain.")
    args = []
    words = action.split(" ")
    if verbose:
        print(f"Words: {words}")
    # remove all delimiters and unnecessary chars
    words = [remove_non_letters(word, exceptions=["_"], allow_digits=True) for word in words]
    if verbose:
        print(f"Words after removing delimiters: {words}")
    arg_names = {text_name: pddl_name for pddl_name, text_name in translator.config.encoded_objects_str.items()}
    for idx, word in enumerate(words):
        for arg_idx, possible_arg_synonym in enumerate(possible_args):
            if possible_arg_synonym == word:
                position = idx
                arg_name = arg_names.get(word)
                args.append((arg_name, position))

    sorted_args = sorted(args, key=lambda x: x[1])
    sorted_args = [arg[0] for arg in sorted_args]

    # make sure package arg comes first and location last
    if action_pred in ["load-truck", "load-airplane", "unload-truck", "unload-airplane"]:
        package_args = [arg for arg in sorted_args if "p" in arg]
        if len(package_args) != 1:
            return None
        non_package_args = [arg for arg in sorted_args if "p" not in arg]
        sorted_args = package_args + non_package_args

        location_args = [arg for arg in sorted_args if "l" in arg]
        if len(location_args) != 1:
            return None
        non_location_args = [arg for arg in sorted_args if "l" not in arg]
        sorted_args = non_location_args + location_args

    # make sure truck arg comes first and city last
    if action_pred == "drive-truck":
        truck_args = [arg for arg in sorted_args if "t" in arg]
        if len(truck_args) != 1:
            return None
        non_truck_args = [arg for arg in sorted_args if "t" not in arg]
        sorted_args = truck_args + non_truck_args

        city_args = [arg for arg in sorted_args if "c" in arg]
        if len(city_args) > 1:
            return None
        non_city_args = [arg for arg in sorted_args if "c" not in arg]
        sorted_args = non_city_args + city_args

        # autofill city if missing (assumes truck number is city number)
        if len(city_args) == 0:
            truck_arg = truck_args[0]
            city_number = truck_arg[-1]  # last char of truck arg
            sorted_args.append("c" + city_number)
            
    # make sure airplane arg comes first
    if action_pred == "fly-airplane":
        airplane_args = [arg for arg in sorted_args if "a" in arg]
        if len(airplane_args) != 1:
            return None
        non_airplane_args = [arg for arg in sorted_args if "a" not in arg]
        sorted_args = airplane_args + non_airplane_args

    if verbose:
        print(f"Args: {sorted_args}")

    if len(sorted_args) != num_args:
        return None

    return (action_pred, tuple(sorted_args))


def text_to_pddl_plan_logistics(text: str, translator) -> str:
    """
    Converts logistics plan in plain text to PDDL plan format.
    """
    plan = []
    text = text.lower().strip()
    splitters = [".", ",", ";", ":"]
    for splitter in splitters:
        text = text.replace(splitter, "\n")
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        action = action_string_to_pddl_tuple_logistics(line, translator)
        if action is None:
            continue
        action_str = f"({action[0]} {' '.join(action[1])})"
        plan.append(action_str)
    return "\n".join(plan)
