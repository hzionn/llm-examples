import json
import os
from typing import List, Literal

import google.genai as genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

ALLOWED_ACTIONS = [
    "move",
    "pick",
    "place",
    "grab",
    "pour",
    "activate",
    "wait",
    "present",
]


# Define the structure of a robot action using Pydantic
class RobotAction(BaseModel):
    action_type: Literal[
        "move", "pick", "place", "grab", "pour", "activate", "wait", "present"
    ] = Field(..., description="Type of action")
    parameters: dict = Field(..., description="Parameters for the action")


class ActionPlan(BaseModel):
    actions: List[RobotAction]


def parse_actions(llm_output: str) -> ActionPlan:

    actions_data = json.loads(llm_output)
    # Validate each action_type
    for a in actions_data:
        if a.get("action_type") not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action_type: {a.get('action_type')}")
    return ActionPlan(actions=[RobotAction(**a) for a in actions_data])


def call_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return the response text."""
    api_key = os.getenv("GEMINI_API_KEY")
    model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
    )
    return response.text


def main():
    user_input = input("Describe the task for the robot: ")

    allowed_actions_str = ", ".join(f'"{a}"' for a in ALLOWED_ACTIONS)
    prompt = (
        f"You are controlling a robot that can only perform the following actions: [{allowed_actions_str}].\n"
        "Given the user's instruction, generate a JSON list of robot actions. "
        "Each action must have 'action_type' and 'parameters'.\n"
        "If the instruction requires actions outside of the allowed list (such as running a marathon, swimming, flying, or any impossible or unsupported task), respond with an empty JSON array [].\n"
        "Respond ONLY with a JSON array, no explanation or extra text. Do not wrap in markdown or code blocks.\n"
        f"Instruction: {user_input}\n"
        "Example output: "
        '[{"action_type": "move", "parameters": {"direction": "forward", "distance": 2}}]'
    )

    llm_output = call_gemini(prompt)

    try:
        cleaned = llm_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()
        if not cleaned.startswith("["):
            raise ValueError("LLM output is not valid JSON list.")
        action_plan = parse_actions(cleaned)
        print("Generated action plan:")
        print(action_plan.model_dump_json(indent=2))

    except (ValidationError, Exception) as e:
        print("Failed to parse LLM output:", e)
        print("Raw output:", llm_output)


if __name__ == "__main__":
    main()
