import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "./question_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        device_map="cpu",        # force CPU
        torch_dtype=torch.float32
    )

    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("Cloud Computing Question Generator")

topic = st.text_input("Enter topic", "Cloud service models")

if st.button("Generate Questions"):

    prompt = f"Generate a university exam question about {topic}:"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=80,
        num_return_sequences=5,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )

    st.subheader("Generated Questions")

    for i, output in enumerate(outputs):
        question = tokenizer.decode(output, skip_special_tokens=True)
        st.write(f"{i+1}. {question}")


