import os
import re
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY not found — check your .env file")

client = genai.Client(api_key=API_KEY)

MODELS_TO_TRY = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
]

# Turbofan report prompt
PROMPT_TEMPLATE = """You are a maintenance engineer assistant writing a brief maintenance interpretation for a turbofan engine monitoring system.

The predictive model has analysed sensor data from one engine and produced the following output:

PREDICTION
- Engine ID: {engine_id}
- Estimated remaining useful life: {predicted_rul:.0f} operating cycles
- Fleet median remaining life for reference: {baseline:.0f} operating cycles

WHAT INFLUENCED THE PREDICTION
The following sensor readings increased the remaining life estimate (the model considers the engine healthier based on these):
{positive_drivers}

The following sensor readings decreased the remaining life estimate (the model considers the engine more worn based on these):
{negative_drivers}

Return ONLY a JSON object (no markdown fences, no prose before or after) with exactly these keys:

{{
  "verdict": "<one of: OPTIMAL, CAUTION, CRITICAL>",
  "summary": "<2-3 sentences in plain English. Mention the estimated remaining life in operating cycles. Say whether it is above or below the fleet median. Write as an interpretation, not a certainty — use phrases like 'the model estimates' or 'the analysis suggests'. Do not claim the engine has definitely failed.>",
  "drivers_positive": "<1 sentence naming the sensor readings that increased the remaining life estimate. Use plain sensor descriptions. Do not use abbreviations like RUL, SHAP, FFT, or model names.>",
  "drivers_negative": "<1 sentence naming the sensor readings that decreased the remaining life estimate. Use plain sensor descriptions. If none, write 'No significant factors decreased the estimate.'>",
  "recommendation": "<ONE full sentence (10-25 words). Start with an action verb. State what to do AND on what component. Frame as advisory — use 'consider', 'schedule', or 'monitor'. Do NOT claim failure has occurred. Example: 'Consider scheduling a compressor inspection within the next maintenance window to verify component condition.'>",
  "urgency": "<one of: Routine, Elevated, Immediate>"
}}

Rules:
- Do not mention SHAP, TreeSHAP, XGBoost, LSTM, FFT, or any technical model names or methods.
- Do not use the abbreviation "RUL" — always write "remaining life" or "remaining useful life" in full.
- Do not invent sensor names or values not given above.
- If the estimated remaining life is below 30, verdict MUST be CRITICAL and urgency MUST be Immediate.
- If the estimated remaining life is between 30 and 80, verdict MUST be CAUTION.
- If the estimated remaining life is above 80, verdict MUST be OPTIMAL.
- Every field MUST be a complete sentence or phrase. Never a single word.
- Treat all predictions as estimates, not certainties. Recommendations are advisory.
- Return valid JSON only. No trailing commas. No markdown.
"""

# Bearing classification prompt
CLASSIFICATION_PROMPT = """You are a maintenance engineer assistant writing a brief maintenance interpretation for a bearing monitoring system.

The predictive model analysed vibration data from one bearing and produced:

PREDICTION
- Bearing signal: {bearing_id}
- Detected condition: {predicted_class}
- Model confidence: {confidence:.0%}

WHAT INFLUENCED THE CLASSIFICATION
Top vibration pattern contributions that supported or opposed the detected condition:
{drivers}

Return ONLY a JSON object (no markdown fences, no prose) with exactly these keys:

{{
  "verdict": "<one of: OPTIMAL, CAUTION, CRITICAL>",
  "summary": "<2-3 sentences in plain English. State the detected condition and model confidence. Write as an interpretation — use phrases like 'the analysis suggests' or 'the model detected'. Do not claim the bearing has definitely failed.>",
  "drivers_positive": "<1 sentence naming the vibration patterns that most supported the detected condition. Use plain descriptions, not technical abbreviations.>",
  "drivers_negative": "<1 sentence naming vibration patterns that argued against it. If none, write 'No significant opposing patterns were detected.'>",
  "recommendation": "<ONE full sentence (10-25 words). Start with an action verb. Frame as advisory. Example: 'Consider scheduling a bearing inspection to verify the detected condition and assess replacement needs.'>",
  "urgency": "<one of: Routine, Elevated, Immediate>"
}}

Rules:
- 'Normal' -> OPTIMAL. 'Ball' or 'Outer Race' -> CRITICAL. 'Inner Race' -> CAUTION (or CRITICAL if confidence > 90%).
- Do not mention SHAP, TreeSHAP, XGBoost, FFT, or any technical model names or methods.
- Treat predictions as estimates, not certainties. Recommendations are advisory only.
- Return valid JSON only. No trailing commas. No markdown.
"""


def _try_generate(prompt, model_name):
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text.strip()


def _fallback(reason: str) -> dict:
    return {
        "verdict": "UNKNOWN",
        "summary": (
            f"AI interpretation temporarily unavailable ({reason}). "
            "The prediction and feature contribution chart above are still valid — "
            "only the written interpretation could not be generated. Try again in a minute."
        ),
        "drivers_positive": "—",
        "drivers_negative": "—",
        "recommendation": "Retry in a minute.",
        "urgency": "Routine",
        "_fallback": True,
    }


def _parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "verdict": "UNKNOWN",
        "summary": raw[:400],
        "drivers_positive": "—",
        "drivers_negative": "—",
        "recommendation": "See summary.",
        "urgency": "Routine",
        "_raw": raw,
    }


def generate_report(engine_id, predicted_rul, baseline, top_features):
    pos = [(f, v) for f, v in top_features if v > 0][:5]
    neg = [(f, v) for f, v in top_features if v < 0][:5]

    pos_text = "\n".join(f"- {f}: +{v:.2f}" for f, v in pos) or "- (none)"
    neg_text = "\n".join(f"- {f}: {v:.2f}"  for f, v in neg) or "- (none)"

    prompt = PROMPT_TEMPLATE.format(
        engine_id=engine_id,
        predicted_rul=predicted_rul,
        baseline=baseline,
        positive_drivers=pos_text,
        negative_drivers=neg_text,
    )

    last_err = None
    raw = None

    for model_name in MODELS_TO_TRY:
        for attempt in range(3):
            try:
                raw = _try_generate(prompt, model_name)
                break
            except genai_errors.ServerError as e:
                last_err = e
                time.sleep(2 ** attempt)
            except genai_errors.ClientError as e:
                last_err = e
                break
        if raw:
            break

    if not raw:
        return _fallback(type(last_err).__name__ if last_err else "UnknownError")

    return _parse_json(raw)


def generate_classification_report(bearing_id, predicted_class, confidence, top_features):
    pos = [(f, v) for f, v in top_features if v > 0][:5]
    neg = [(f, v) for f, v in top_features if v < 0][:5]

    pos_text = "\n".join(f"- {f}: +{v:.3f}" for f, v in pos) or "- (none)"
    neg_text = "\n".join(f"- {f}: {v:.3f}"  for f, v in neg) or "- (none)"
    drivers = pos_text + "\n" + neg_text

    prompt = CLASSIFICATION_PROMPT.format(
        bearing_id=bearing_id,
        predicted_class=predicted_class,
        confidence=confidence,
        drivers=drivers,
    )

    last_err = None
    raw = None
    for model_name in MODELS_TO_TRY:
        for attempt in range(3):
            try:
                raw = _try_generate(prompt, model_name)
                break
            except genai_errors.ServerError as e:
                last_err = e
                time.sleep(2 ** attempt)
            except genai_errors.ClientError as e:
                last_err = e
                break
        if raw:
            break

    if not raw:
        return _fallback(type(last_err).__name__ if last_err else "UnknownError")

    return _parse_json(raw)


if __name__ == "__main__":
    test_top = [
        ("s7_delta", 9.65),
        ("s15_delta", 6.17),
        ("s8_delta", -5.63),
        ("s12_delta", 5.01),
        ("s9_delta", -4.11),
        ("s4_delta", 3.63),
        ("s14_delta", 3.40),
        ("s11_delta", 3.39),
    ]
    test_report = generate_report(
        engine_id=1,
        predicted_rul=82.4,
        baseline=63.1,
        top_features=test_top,
    )
    print(json.dumps(test_report, indent=2))
