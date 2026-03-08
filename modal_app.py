import modal

volume = modal.Volume.from_name("job-predictor-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "huggingface_hub",
        "sentencepiece", "protobuf", "numpy",
    )
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("config.py", "/root/config.py")
)

app = modal.App("job-predictor", image=image)

# Run this command first: `modal secret create huggingface HF_TOKEN=hf_your_token_here`
# Deploy with: `modal deploy modal_app.py` 
@app.cls(
    gpu="T4",
    memory=8192,
    timeout=300,
    volumes={"/weights": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
class JobPredictor:

    @modal.enter()
    def load_model(self):
        import os
        os.environ["HF_HOME"] = "/weights/hf_cache"
        from model import load_predictor
        self.model, self.tokenizer, self.norm_stats = load_predictor()

    @modal.method()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        from model import predict_all
        return predict_all(texts, self.model, self.tokenizer, self.norm_stats)


# Call with `modal run modal_app.py`
@app.local_entrypoint()
def main():
    sample_texts = [
        "[LOCATION]: United States California [TITLE]: Senior Software Engineer [DESC]: 5+ years Python required.",
    ]
    predictor = JobPredictor()
    results = predictor.predict_batch.remote(sample_texts)
    print("Prediction:", results[0])
