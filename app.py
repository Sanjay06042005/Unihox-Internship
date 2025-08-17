import streamlit as st
import json, re, textwrap
from typing import List, Dict, Any, Tuple
from mistralai import Mistral
import spacy, torch
from transformers import pipeline

st.set_page_config(page_title="AI Wellness Assistant", page_icon="🩺", layout="centered")

MISTRAL_API_KEY = "ktCJ5k5acfiU8Y7p4a219N3ghA3oDmIi"        
MODEL = "mistral-large-latest"
USE_MISTRAL = bool(MISTRAL_API_KEY and MISTRAL_API_KEY != "REPLACE_ME")

@st.cache_resource
def load_nlp_pipeline():
    nlp = spacy.load("en_core_web_sm")
    intent_classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased",
        device=0 if torch.cuda.is_available() else -1
    )
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return nlp, intent_classifier, sentiment_analyzer

nlp, intent_classifier, sentiment_analyzer = load_nlp_pipeline()

def ensure_list(v):
    return list(v) if isinstance(v, list) else [v] if v else []

def normalize_list(v):
    v = ensure_list(v)
    return [str(item) if not isinstance(item, dict) else json.dumps(item) for item in v]

def shorten(text: str, max_chars: int = 360):
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars-1].rstrip() + "…"

def shorten_join(items: List[str], max_chars: int = 360):
    return shorten(" • ".join([s.strip(" -•") for s in items if s]), max_chars=max_chars)

def tolerant_json_loads(raw: str):
    """
    Try very hard to get a JSON object out of an LLM response.
    Returns dict or None.
    """
    if not raw:
        return None
 
    raw = re.sub(r"^```.*?\n|```$", "", raw.strip(), flags=re.DOTALL)
  
    try:
        return json.loads(raw)
    except Exception:
        pass
    
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        block = m.group(0)
       
        try:
            return json.loads(block)
        except Exception:
           
            repaired = block.replace("'", '"')
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            try:
                return json.loads(repaired)
            except Exception:
                return None
    return None

def get_intent(text: str) -> str:
    
    try:
        label = intent_classifier(text, truncation=True)[0]['label']
    except Exception:
        label = "General"
    tokens = text.lower()
    keywords = ["pain", "ache", "fever", "cough", "dizzy", "nausea", "vomit", "tired", "fatigue",
                "headache", "sore throat", "diarrhea", "rash", "cramp"]
    if any(k in tokens for k in keywords):
        return "Symptom Check"
    return "Symptom Check" if label.lower() in ["symptom", "symptom check", "medical", "health"] else "General"

def extract_entities_and_cuis(text: str) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    
    symptom_like = [t.text.lower() for t in doc if t.pos_ in {"ADJ", "NOUN"}]
  
    symptoms = sorted(set(symptom_like)) or [text.strip().lower()]
    return symptoms, [], entities

def detect_emotional_state(text: str) -> str:
    try:
        result = sentiment_analyzer(text)[0]
        return f"{result['label']} ({result['score']:.2f})"
    except Exception:
        return "Unknown"

FALLBACK_HINTS = {
    "fever": {
        "precautions": ["Rest and avoid strenuous activity", "Monitor temperature twice daily", "Avoid aspirin for children"],
        "diet": ["Fluids: water, oral rehydration, broths", "Light meals: rice, bananas, toast"],
        "yoga": ["Gentle breathing (5–10 min)"],
        "follow_up": ["Seek care if fever > 102°F (38.9°C), >3 days, or with severe symptoms"]
    },
    "headache": {
        "precautions": ["Dim lights and reduce screen time", "Hydrate", "Avoid alcohol"],
        "diet": ["Small frequent water sips", "Magnesium-rich foods (leafy greens, nuts)"],
        "yoga": ["Neck/shoulder rolls, gentle stretches"],
        "follow_up": ["Urgent care for worst-ever headache, neuro symptoms, head injury"]
    },
    "cough": {
        "precautions": ["Avoid smoke/irritants", "Use a cool-mist humidifier"],
        "diet": ["Warm fluids (tea with honey if not <1yo)", "Soups and broths"],
        "yoga": ["Diaphragmatic breathing"],
        "follow_up": ["Seek care if cough >3 weeks, high fever, chest pain, or shortness of breath"]
    },
    "dizzy": {
        "precautions": ["Rise slowly", "Sit/lie if spinning sensation"],
        "diet": ["Hydrate; consider oral rehydration salts"],
        "yoga": ["Seated forward folds, legs-up-the-wall"],
        "follow_up": ["Immediate care if fainting, chest pain, neuro symptoms"]
    },
    "tired": {
        "precautions": ["Set consistent sleep schedule", "Limit caffeine after noon"],
        "diet": ["Iron-rich foods (beans, greens) + vitamin C", "Balanced meals, avoid heavy late dinners"],
        "yoga": ["Gentle stretches before bed"],
        "follow_up": ["Discuss labs (iron, thyroid, B12) if persistent"]
    },
    "sore throat": {
        "precautions": ["Avoid shouting; rest voice"],
        "diet": ["Warm salt-water gargles", "Warm soups, soothing teas"],
        "yoga": ["Gentle breathing only"],
        "follow_up": ["Care for severe pain, trouble swallowing/breathing, or >1 week"]
    }
}

def fallback_from_text(text: str):
    txt = text.lower()
    out = {"precautions": [], "diet": [], "yoga": [], "follow_up": []}
    for key, recs in FALLBACK_HINTS.items():
        if key in txt:
            for k in out:
                out[k].extend(recs.get(k, []))
    
    if not any(out.values()):
        out = {
            "precautions": ["Avoid strenuous activity for now", "Note any new/worsening symptoms"],
            "diet": ["Hydrate regularly", "Simple, balanced meals"],
            "yoga": ["Gentle breathing 5 minutes"],
            "follow_up": ["Seek care if symptoms worsen or persist >3 days"]
        }
    
    for k in out:
        seen, deduped = set(), []
        for item in out[k]:
            if item not in seen:
                deduped.append(item); seen.add(item)
        out[k] = deduped
    return out

def call_mistral(symptoms, cuis, ranked_conditions, recs, free_text: str):
    client = Mistral(api_key=MISTRAL_API_KEY)
    system_msg = (
        "You are an AI wellness assistant. "
        "Always respond with STRICT JSON only (no prose, no markdown). "
        "Schema: {"
        '"summary": string (<= 3 sentences), '
        '"precautions": string[], '
        '"yoga": string[], '
        '"diet": string[], '
        '"follow_up": string[], '
        '"disclaimer": string'
        "}. Keep list items short."
    )
    payload = {
        "symptoms": symptoms,
        "umls_cuis": cuis,
        "ranked_conditions": ranked_conditions,
        "rule_recommendations": recs,
        "free_text": free_text
    }

    resp = client.chat.complete(
        model=MODEL,
        temperature=0, 
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(payload)}
        ]
    )

    raw = resp.choices[0].message.content if resp and resp.choices else ""
    data = tolerant_json_loads(raw)
    return data

class Plan:
    def __init__(self, symptoms, cuis, ranked_conditions, summary, precautions, yoga, diet, follow_up, disclaimer):
        self.symptoms = symptoms
        self.cuis = cuis
        self.ranked_conditions = ranked_conditions
        self.summary = summary
        self.precautions = normalize_list(precautions)
        self.yoga = normalize_list(yoga)
        self.diet = normalize_list(diet)
        self.follow_up = normalize_list(follow_up)
        self.disclaimer = disclaimer

def run_pipeline(text: str) -> Plan:
    intent = get_intent(text)
    if intent != "Symptom Check":
        st.info(f"Detected intent: {intent}.")
    symptoms, cuis, _ = extract_entities_and_cuis(text)
    emotion = detect_emotional_state(text)
    st.caption(f"Detected Emotional State: {emotion}")
    ranked_labels = []  
    recs = {}          

    plan_json = None
    if USE_MISTRAL:
        try:
            plan_json = call_mistral(symptoms, cuis, ranked_labels, recs, free_text=text)
        except Exception as e:
            st.warning(f"Mistral fallback: {e}")

    if not plan_json:
        local = fallback_from_text(text)
        plan_json = {
            "summary": "Here are brief, general steps based on your note.",
            **local,
            "disclaimer": "Informational only. Not a medical diagnosis. Seek care for severe or persistent symptoms."
        }

    return Plan(
        symptoms,
        cuis,
        ranked_labels,
        plan_json.get("summary", ""),
        plan_json.get("precautions", []),
        plan_json.get("yoga", []),
        plan_json.get("diet", []),
        plan_json.get("follow_up", []),
        plan_json.get("disclaimer", "")
    )

st.title("🩺 AI Wellness Assistant")
user_input = st.text_area("Describe your symptoms:", placeholder="e.g., Headache with mild fever since last night, a bit nauseous")

if st.button("Get Wellness Plan") and user_input.strip():
    plan = run_pipeline(user_input)

    st.subheader("🧾 Your Personalized Wellness Plan")

    st.markdown("### 📝 Summary")
    st.success(shorten(plan.summary or "Summary unavailable", 280))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚠️ Precautions")
        st.info(shorten_join(plan.precautions, 380) if plan.precautions else "Not available")

        st.markdown("### 🍎 Diet")
        st.success(shorten_join(plan.diet, 380) if plan.diet else "Not available")

    with col2:
        st.markdown("### 🧘 Yoga")
        st.info(shorten_join(plan.yoga, 380) if plan.yoga else "Not available")

        st.markdown("### 📌 Next Steps")
        st.success(shorten_join(plan.follow_up, 380) if plan.follow_up else "Not available")

    st.caption(plan.disclaimer or "This is AI-generated information and not medical advice.")
