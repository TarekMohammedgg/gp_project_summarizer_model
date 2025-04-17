from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, json
from peft import prepare_model_for_kbit_training
import bitsandbytes as bnb
import os

# -------- CONFIGURATION --------
MODEL_NAME = "CohereForAI/c4ai-command-r7b-arabic-02-2025"
HF_TOKEN = os.getenv("HF_TOKEN", "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")  # Use env var in production

# -------- PYDANTIC SCHEMA --------
class SummaryDetails(BaseModel):
    Patient_symptoms: List[str] = Field(..., min_length=5, max_length=300)
    Symptom_location: str = Field(..., min_length=5, max_length=300)
    Duration: str = Field(..., min_length=5, max_length=300)
    Symptom_progression: str = Field(..., min_length=5, max_length=300)
    Risk_factors: str = Field(..., min_length=5, max_length=300)

class SummarizationInput(BaseModel):
    arabic_text: str

# -------- FASTAPI INIT --------
app = FastAPI()
model = None
tokenizer = None

# -------- MODEL LOADER --------
@app.on_event("startup")
def load_model():
    global model, tokenizer

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=HF_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_auth_token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded successfully.")

# -------- UTILS --------
def extract_translation(messages, tokenizer, model, max_new_tokens=1500):
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
    )
    output_ids = gen_tokens[0][input_ids.shape[-1]:]
    gen_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return gen_text

def extract_details(messages, tokenizer, model, max_new_tokens=800):
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
    )
    output_ids = gen_tokens[0][input_ids.shape[-1]:]
    gen_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    # Clean if wrapped in Markdown
    if "```json" in gen_text:
        gen_text = gen_text.split("```json")[-1].split("```")[0].strip()
    return gen_text

# -------- MAIN ENDPOINT --------
@app.post("/summarize")
def summarize(input_data: SummarizationInput):
    try:
        # Step 1: Translate Arabic to English
        translation_message = [
            {
                "role": "system",
                "content": "\n".join([
                    "You are a professional translator.",
                    "You will be provided by an Arabic text.",
                    "You have to translate the text into the English Language",
                    "Follow the provided Scheme to generate a JSON",
                    "Do not generate any introduction or conclusion."
                ])
            },
            {
                "role": "user",
                "content": input_data.arabic_text.strip()
            }
        ]

        en_text = extract_translation(translation_message, tokenizer, model)

        # Step 2: Extract details based on schema
        details_message = [
            {
                "role": "system",
                "content": "\n".join([
                    "You are an NLP data parser.",
                    "You will be provided by an Arabic text associated with a Pydantic scheme.",
                    "Generate the output as same as language.",
                    "You have to extract JSON details from text according to the Pydantic details.",
                    "Extract details as mentioned in text.",
                    "Do not generate any introduction or conclusion.",
                ])
            },
            {
                "role": "user",
                "content": "\n".join([
                    "## conversation",
                    en_text,
                    "## Pydantic Details",
                    json.dumps(SummaryDetails.model_json_schema(), ensure_ascii=False),
                    "## Summarization : ",
                    "```json"
                ])
            }
        ]

        summary = extract_details(details_message, tokenizer, model)

        return json.loads(summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
