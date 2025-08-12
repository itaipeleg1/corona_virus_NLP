from models.model_config import model_configs

def get_model_size_and_tokenizer(model_key):
    config = model_configs[model_key]
    model = config["model_class"].from_pretrained(config["model_name"])
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / 1024**2
    
    tokenizer = config["tokenizer_class"].from_pretrained(config["model_name"])
    print(f"🧮 Model size of {model_key} is {total_size_mb:.2f} MB")
    print("Teacher tokenizer vocab size:", tokenizer.vocab_size)


    return total_size_mb

# bertweet_size = get_model_size_in_mb("bertweet")
# student_bertweet_size = get_model_size_in_mb("student_bertweet_roberta")
covidbert_size = get_model_size_in_mb("covidbert")
student_covidbert_size = get_model_size_in_mb("student_covidbert_bert")