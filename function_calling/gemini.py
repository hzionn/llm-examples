import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_ID = os.getenv("GEMINI_MODEL_ID")
user_prompt = input(">>> ")


# Define the function with type hints and docstring
def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    """Calculates the Body Mass Index (BMI) given weight and height.

    Args:
        weight_kg: Weight in kilograms.
        height_cm: Height in centimeters.

    Returns:
        A dictionary containing the BMI value and a category string.
    """
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m**2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obesity"
    return {"bmi": round(bmi, 2), "category": category}


# Configure the client and model
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
config = types.GenerateContentConfig(tools=[calculate_bmi])

response = client.models.generate_content(
    model=MODEL_ID,
    contents=user_prompt,
    config=config,
)

print(response.text)  # The SDK handles the function call and returns the final text
