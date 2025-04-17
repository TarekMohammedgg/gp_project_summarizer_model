import gradio as gr
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# احصل على التوكن من Secrets
hf_token = os.environ.get("HF_TOKEN")

# اسم الموديل
model_name = "CohereForAI/c4ai-command-r7b-arabic-02-2025"

# تحميل الموديل والتوكن
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token
)

# تحميل التوكنيزر
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

# دالة الترجمة من النص العربي
def extract_translation(text):
    messages = [
        {
            "role": "system",
            "content": "\n".join([
                "You are a professional translator.",
                "You will be provided by an Arabic text.",
                "You have to translate the text into the English Language.",
                "Follow the provided Scheme to generate a JSON.",
                "Do not generate any introduction or conclusion."
            ])
        },
        {
            "role": "user",
            "content": text.strip()
        }
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(input_ids, max_new_tokens=1500)
    response = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
    return response.strip()

# دالة استخراج التفاصيل من الترجمة
def extract_details(en_text):
    details_extraction_message = [
        {
            "role": "system",
            "content": "\n".join([
                "You are an NLP data parser.",
                "You will be provided by an Arabic text associated with a Pydantic scheme.",
                "Generate the output as same as language.",
                "You have to extract JSON details from text according the Pydantic details.",
                "Do not generate any introduction or conclusion.",
            ])
        },
        {
            "role": "user",
            "content": "\n".join([
                "## conversation",
                en_text.strip(),
                "",
                "## Pydantic Details",
                json.dumps({
                    "title": "SummaryDetails",
                    "type": "object",
                    "properties": {
                        "Patient_symptoms": {"type": "array", "items": {"type": "string"}},
                        "Symptom_location": {"type": "string"},
                        "Duration": {"type": "string"},
                        "Symptom_progression": {"type": "string"},
                        "Risk_factors": {"type": "string"}
                    },
                    "required": [
                        "Patient_symptoms",
                        "Symptom_location",
                        "Duration",
                        "Symptom_progression",
                        "Risk_factors"
                    ]
                }, ensure_ascii=False),
                "",
                "## Summarization : ",
                "```json"
            ])
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        details_extraction_message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(input_ids, max_new_tokens=1500)
    summary = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
    return summary.strip()

# واجهة Gradio
def summarize_api(arabic_text):
    try:
        translated = extract_translation(arabic_text)
        details = extract_details(translated)
        return details
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=summarize_api,
    inputs=gr.Textbox(label="Arabic Medical Conversation", lines=15),
    outputs=gr.Textbox(label="Extracted JSON Summary", lines=15),
    title="Arabic Medical Conversation Summarizer"
)

# شغل التطبيق
demo.launch()
