
import re
import json
import logging
import os
from typing import Literal, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)

Category = Literal["data_request","help_request","info_request","nonsense","offensive"]

class Classifier:
    """
    Low-cost LLM gate. Classifies a user message.
    1) Heuristics for greetings/help/info/nonsense
    2) Otherwise call the model (default gpt-3.5) to decide.
    """
    def __init__(self, api_key: str, model: str | None = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("LLM_CLASSIFIER_MODEL","gpt-3.5-turbo")
        # fallback if 3.5 not available
        if not self.model:
            self.model = "gpt-4o-mini"

        # simple profanity list (extend as needed)
        self.bad_words = {"vaffanculo","stronzo","merda","cazzo","bastardo","cretino","imbecille","troia","puttana"}

        self.help_triggers = [
            "aiuto","help","come funziona","come funzioni","cosa puoi fare","istruzioni","manuale","spiegami",
            "tutorial","guida","come ti uso"
        ]
        self.info_triggers = [
            "chi sei","cosa sei","chi ti ha creato","presentati","info","informazioni","identità","identita"
        ]
        self.greetings = ["ciao","salve","buongiorno","buonasera","hello","hi","ehi","hey"]

    # ---------- Public API ----------
    def classify(self, text: str) -> Dict[str, Any]:
        text_norm = (text or "").strip().lower()

        # 0) empty or too short
        if len(text_norm) < 2:
            return {"category":"nonsense","reason":"too_short"}

        # 1) offensive check
        if any(bw in text_norm for bw in self.bad_words):
            return {"category":"offensive","reason":"profanity"}

        # 2) greetings/help/info heuristics
        if any(g in text_norm for g in self.greetings):
            # greet -> usually a help intent
            return {"category":"help_request","reason":"greeting_detected"}
        if any(h in text_norm for h in self.help_triggers):
            return {"category":"help_request","reason":"help_trigger"}
        if any(i in text_norm for i in self.info_triggers):
            return {"category":"info_request","reason":"info_trigger"}

        # 3) nonsense heuristic: only symbols or random letters w/out vowels
        if re.fullmatch(r"[^\w\s]+", text_norm) or (re.fullmatch(r"[a-z]{10,}", text_norm) and not any(v in text_norm for v in "aeiou")):
            return {"category":"nonsense","reason":"noise_pattern"}

        # 4) LLM lightweight classification
        try:
            system = (
                "You are a concise classifier. Output ONLY a compact JSON with keys: "
                "'category' in ['data_request','help_request','info_request','nonsense','offensive'] "
                "and optional 'reason'. If the user asks about data, charts, metrics, population, GDP, income, etc., it's 'data_request'. "
                "If the user asks who you are or how to use, choose 'info_request' or 'help_request'. "
                "If the message is insults or profanity, choose 'offensive'. "
                "If it's random text lacking meaning, choose 'nonsense'."
            )
            user = f'Classify the following message: "{text_norm}"'
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=0
            )
            out = resp.choices[0].message.content.strip()
            if "```json" in out:
                out = out[out.find("```json")+7 : out.rfind("```")]
            data = json.loads(out)
            cat = data.get("category","data_request")
            if cat not in ["data_request","help_request","info_request","nonsense","offensive"]:
                cat = "data_request"
            data["category"] = cat
            return data
        except Exception as e:
            logger.warning(f"classifier fallback due to error: {e}")
            # default to data_request in doubt
            return {"category":"data_request","reason":"fallback"}
