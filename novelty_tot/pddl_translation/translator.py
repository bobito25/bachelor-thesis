from abc import ABC, abstractmethod

import yaml

from pddl_translation.pddl_translation import PDDLTranslationConfig, action_to_string as nat_lang_action_to_string, atom_to_string as nat_lang_atom_to_string, decode_action_string


class Translator(ABC):

    @abstractmethod
    def action_to_string(self, action: tuple[str, tuple]) -> str:
        """
        Convert a PDDL action and its parameters to a string representation.
        """
        pass

    @abstractmethod
    def atom_to_string(self, atom: tuple[str, tuple]) -> str:
        """
        Convert a PDDL atom to a string representation.
        """
        pass

    def state_to_string(self, state: set[tuple[str, tuple]]) -> str:
        """
        Convert a set of PDDL atoms to a string representation.
        """
        atoms = [self.atom_to_string(atom) for atom in state]
        atoms = [atom for atom in atoms if atom]  # Filter out empty strings
        return ", ".join(atoms)

    @abstractmethod
    def plan_to_string(self, plan: list[tuple[str, tuple]] | list[str]) -> str:
        """
        Convert a PDDL plan (list of actions) to a string representation.
        """
        pass

    def goal_atoms_to_string(self, goal_atoms: set[tuple[bool, str, tuple]]) -> str:
        """
        Convert goal atoms to a string representation.
        """
        goal_strs = []
        for goal_atom in goal_atoms:
            is_true, name, params = goal_atom
            string = self.atom_to_string((name, params))
            if is_true:
                goal_str = string
            else:
                goal_str = "(not " + string + ")"
            goal_strs.append(goal_str)
        return " and ".join(goal_strs)


class StandardTranslator(Translator):
    def __init__(self):
        self.type = "standard"

    def atom_to_string(self, atom: tuple[str, tuple]) -> str:
        name, args = atom
        if not args:
            return f"({name})"
        args = tuple(chr(ord('a') + i) for i in args)
        return f"({name} {' '.join(map(str, args))})"

    def action_to_string(self, action: tuple[str, tuple]) -> str:
        name, args = action
        if not args:
            return f"({name})"
        args = tuple(chr(ord('a') + i) for i in args)
        return f"({name} {' '.join(map(str, args))})"

    def plan_to_string(self, plan: list[tuple[str, tuple]] | list[str]) -> str:
        if not plan:
            return "[empty plan]"
        if isinstance(plan[0], str):
            return ", ".join(plan)
        return ", ".join(self.action_to_string(action) for action in plan)

class NatLangTranslator(Translator):
    def __init__(self, config: PDDLTranslationConfig | None = None):
        if config is None:
            config_dict = yaml.safe_load(open('pddl_translation/domains/blockworld.yaml', 'r'))
            config = PDDLTranslationConfig.model_validate(config_dict)
        self.config = config
        self.type = "nat_lang"

    def atom_to_string(self, atom: tuple[str, tuple]) -> str:
        return nat_lang_atom_to_string(self.config, atom)

    def action_to_string(self, action: tuple[str, tuple]) -> str:
        return nat_lang_action_to_string(self.config, action)

    def plan_to_string(self, plan: list[tuple[str, tuple]] | list[str]) -> str:
        if not plan:
            return "[empty plan]"
        if isinstance(plan[0], str):
            plan = [decode_action_string(action, self.config) for action in plan]
        elif isinstance(plan[0], tuple):
            plan = [self.action_to_string(action) for action in plan]
        else:
            raise ValueError("Plan must be a list of tuples or strings.")
        return ", ".join(plan)
