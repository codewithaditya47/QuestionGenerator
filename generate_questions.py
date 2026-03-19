from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "question_model"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

 

prompt = "Generate questions related to cloud service models, deployment models, resource migration,virtualisation:"
inputs = tokenizer(prompt, return_tensors="pt")


outputs = model.generate(
    inputs["input_ids"],
    max_length=80,
    num_return_sequences=5,
    do_sample=True,
    temperature=0.9,
    top_k=50,
    top_p=0.95
)




for i, output in enumerate(outputs):
    print(f"Question {i+1}:")
    print(tokenizer.decode(output, skip_special_tokens=True))
