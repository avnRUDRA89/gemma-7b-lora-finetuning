# 🔧 Fine-Tuning Gemma 7B with LoRA & TRL

This project demonstrates how to fine-tune **Google’s Gemma-7B** language model using **LoRA (Low-Rank Adaptation)** and **Hugging Face’s TRL (SFTTrainer)**. It supports **4-bit quantization** for efficient GPU memory usage and includes an interactive CLI chatbot built from the fine-tuned model.

---

## 📌 Key Features

- ⚡ 4-bit quantized loading with `bitsandbytes`
- 🔁 LoRA-based fine-tuning with `peft`
- 📚 Supervised Fine-Tuning with `trl`'s `SFTTrainer`
- 🧠 Prompt formatting using paired `input`/`output` JSON
- 💬 Inference-ready merged model with interactive CLI
- 🧰 Fully GPU-accelerated with `transformers`, `datasets`, and `accelerate`

---

## 🧠 Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
