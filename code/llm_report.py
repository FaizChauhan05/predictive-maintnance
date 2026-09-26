"""
llm_report.py
Turns (engine_id, predicted_rul, top SHAP features) into a structured
maintenance report using Gemini. Returns a dict with a fixed schema:

    {
        "verdict":            "OPTIMAL" | "CAUTION" | "CRITICAL" | "UNKNOWN",
        "summary":            str,
        "drivers_positive":   str,
        "drivers_negative":   str,
        "recommendation":     str,
        "urgency":            "Routine" | "Elevated" | "Immediate"
    }
"""

import os
import re
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from pathlib import Path
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY not found — check your .env file")

client = genai.Client(api_key=API_KEY)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a maintenance engineer assistant for a turbofan engine monitoring system.

The AI model has analysed the sensor data from one engine and produced the following output:

PREDICTION
- Engine ID: {engine_id}
- Predicted Remaining Useful Life (RUL): {predicted_rul:.0f} cycles
- Population-average RUL for reference: {baseline:.0f} cycles

WHY THE MODEL PREDICTED THIS
The following sensor features pushed the prediction UP (model thinks the engine has MORE life left):
{positive_drivers}

The following sensor features pushed the prediction DOWN (model thinks the engine has LESS life left):
{negative_drivers}

Return ONLY a JSON object (no markdown fences, no prose before or after) with exactly these keys:

{{
  "verdict": "<one of: OPTIMAL, CAUTION, CRITICAL>",
  "summary": "<2-3 sentences. Plain English. Mention the predicted RUL in cycles. Say whether it's above or below the fleet average.>",
  "drivers_positive": "<1 sentence naming the sensors that pushed RUL up. Use plain descriptions, not 'SHAP'.>",
  "drivers_negative": "<1 sentence naming the sensors that pushed RUL down.>",
  "recommendation": "<ONE full sentence (10-25 words). Start with an action verb. State what to do AND on what component. Example: 'Schedule a bearing inspection within the next maintenance window to verify raceway condition.' Do NOT return a single word.>",,
  "urgency": "<one of: Routine, Elevated, Immediate>"
}}

Rules:
- Do not mention SHAP, XGBoost, LSTM, or any technical model detail.
- Do not invent sensor names or values not given above.
- If the predicted RUL is below 30, verdict MUST be CRITICAL and urgency MUST be Immediate.
- If the predicted RUL is between 30 and 80, verdict MUST be CAUTION.
- If the predicted RUL is above 80, verdict MUST be OPTIMAL.
- Every field MUST be a complete sentence or phrase. Never a single word.
- Return valid JSON only. No trailing commas. No markdown.
"""


# ---------------------------------------------------------------------------
# Models to try, in order. Fall back if one is retired or rate-limited.
# ---------------------------------------------------------------------------
MODELS_TO_TRY = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
]


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
            f"AI report temporarily unavailable ({reason}). "
            "The prediction and SHAP explanation above are still valid — "
            "only the natural-language summary failed. Try again in a minute."
        ),
        "drivers_positive": "—",
        "drivers_negative": "—",
        "recommendation": "Retry in a minute.",
        "urgency": "Routine",
        "_fallback": True,
    }


def _parse_json(raw: str) -> dict:
    """Parse Gemini's output into a dict, stripping markdown fences if present."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # last-ditch: find the first {...} block
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # give up — return raw text as summary
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
    """
    engine_id       : int
    predicted_rul   : float
    baseline        : float
    top_features    : list of (feature_name, shap_value), sorted by |shap|
    Returns a dict (see schema at top of file).
    """
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
                time.sleep(2 ** attempt)          # 1s, 2s, 4s
            except genai_errors.ClientError as e:
                last_err = e
                break                              # 404 / 429 → next model
        if raw:
            break

    if not raw:
        print("=== Gemini error ===")
        if last_err is not None:
            print("Type:   ", type(last_err).__name__)
            print("Status: ", getattr(last_err, "code", "no code"))
            print("Message:", str(last_err)[:500])
        print("===================")
        return _fallback(type(last_err).__name__ if last_err else "UnknownError")

    return _parse_json(raw)


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    top = [
        ("s7_delta",  9.65),
        ("s15_delta", 6.17),
        ("s8_delta", -5.63),
        ("s12_delta", 5.01),
        ("s9_delta", -4.11),
        ("s4_delta",  3.63),
        ("s14_delta", 3.40),
        ("s11_delta", 3.39),
    ]
    report = generate_report(
        engine_id=1,
        predicted_rul=82.4,
        baseline=63.1,
        top_features=top,
    )
    print(json.dumps(report, indent=2))


    # ---------------------------------------------------------------------------
# Bearing fault classification report
# ---------------------------------------------------------------------------
CLASSIFICATION_PROMPT = """You are a maintenance engineer assistant for a bearing monitoring system.

The AI model analysed vibration data from one bearing and produced:

PREDICTION
- Bearing ID: {bearing_id}
- Predicted condition: {predicted_class}
- Confidence: {confidence:.0%}

WHY THE MODEL PREDICTED THIS
Top vibration features driving the prediction (signed SHAP contributions):
{drivers}

Return ONLY a JSON object (no markdown fences, no prose) with exactly these keys:

{{
  "verdict": "<one of: OPTIMAL, CAUTION, CRITICAL>",
  "summary": "<2-3 sentences. Plain English. State the fault type and confidence.>",
  "drivers_positive": "<1 sentence naming the features that most supported the prediction.>",
  "drivers_negative": "<1 sentence naming features that argued against it. If none, write 'No significant opposing signals.'>",
  "recommendation": "<one concrete action: monitor / inspect / replace / schedule.>",
  "urgency": "<one of: Routine, Elevated, Immediate>"
}}

Rules:
- 'Normal' -> OPTIMAL. 'Ball' or 'Outer Race' -> CRITICAL. 'Inner Race' -> CAUTION (or CRITICAL if confidence > 90%).
- Do not mention SHAP, XGBoost, or any technical model detail.
- Return valid JSON only. No trailing commas. No markdown.
"""


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
        print("=== Gemini error (classification) ===")
        if last_err is not None:
            print("Type:   ", type(last_err).__name__)
            print("Status: ", getattr(last_err, "code", "no code"))
            print("Message:", str(last_err)[:500])
        print("====================================")
        return _fallback(type(last_err).__name__ if last_err else "UnknownError")

    return _parse_json(raw)