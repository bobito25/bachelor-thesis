import os
import yaml
from pydantic import BaseModel, model_validator
from typing import Optional, Any
from typing_extensions import Self

CONFIG_DIR = "configs/"

MODES = {
    "stream": {
        "attributes": ["stream_prompt"],
    },
    "base": {
        "attributes": ["succ_prompt", "gen_succ_prompt", "progress_prompt"],
    },
    "explicit_state_action": {
        "attributes": ["succ_prompt", "gen_action_prompt", "map_succ_prompt", "current_state_prompt", "progress_prompt"],
    }
}

class Configuration(BaseModel):
    mode: str
    succ_prompt: Optional[str] = None
    gen_succ_prompt: Optional[str] = None
    stream_prompt: Optional[str] = None
    gen_action_prompt: Optional[str] = None
    current_state_prompt: Optional[str] = None
    map_succ_prompt: Optional[str] = None
    progress_prompt: Optional[str] = None
    novelty: bool = False
    novelty_estimation: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def infer_mode(cls, data: Any) -> Any:
        if not "mode" in data:
            possible_modes = []
            for mode in MODES:
                if all(attr in data for attr in MODES[mode]["attributes"]):
                    possible_modes.append(mode)
            if len(possible_modes) == 1:
                data["mode"] = possible_modes[0]
        return data
    
    @model_validator(mode="after")
    def val_mode(self) -> Self:
        if self.mode not in MODES:
            raise ValueError(f"Invalid mode: {self.mode}. Valid modes are: {', '.join(MODES.keys())}")
        else:
            for attr in MODES[self.mode]["attributes"]:
                if getattr(self, attr) is None:
                    raise ValueError(f"Missing attribute '{attr}' for mode '{self.mode}'")
        return self

    @model_validator(mode="after")
    def val_novelty_estimation(self) -> Self:
        if self.novelty_estimation and not self.novelty:
            raise ValueError("novelty_estimation can only be True if novelty is True")
        if self.novelty_estimation and not self.mode == "explicit_state_action":
            raise ValueError("novelty_estimation can only be True if mode is 'explicit_state_action'")
        return self

    @classmethod
    def from_file(cls, name: str):
        path = os.path.join(CONFIG_DIR, name) + ".yaml"
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    def to_file(self, name: str):
        path = os.path.join(CONFIG_DIR, name) + ".yaml"
        with open(path, "w") as f:
            yaml.dump(self.model_dump(exclude_unset=True), f, sort_keys=False)