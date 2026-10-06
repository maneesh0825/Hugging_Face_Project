# Technova AI Assistant (Hugging Face Project)

A beginner-friendly, CPU-only walkthrough of the Hugging Face ecosystem using a small fictional company, **Technova**, as the example data.

## What it covers

| Step | Topic | Model / Library |
|---|---|---|
| 1 | Building a dataset | `datasets` |
| 2 | Tokenization (tokens and token IDs) | `AutoTokenizer` |
| 3 | Text generation pipeline | `sshleifer/tiny-gpt2` |
| 4 | Trainer API setup | `Trainer`, `TrainingArguments`, `DataCollatorForLanguageModeling` |
| 5 | PEFT with LoRA (r=4, alpha=8, target `c_attn`) | `peft` |
| 6 | QLoRA (explained in comments, not executed) | `bitsandbytes` (GPU only) |
| 7 | Extractive question answering | `distilbert-base-cased-distilled-squad` |

## Requirements

- Python 3.9+
- No GPU needed (everything runs on CPU)
- Internet on the first run, to download the models (a few hundred MB in total)

## Installation

```bash
git clone https://github.com/<your-username>/technova-ai-assistant.git
cd technova-ai-assistant

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Expected output includes the dataset, tokens and token IDs, generated text, the number of LoRA trainable parameters, and the final answer to *"When was Technova founded?"* (expected: `2018`).

## Notes

- `tiny-gpt2` is a toy model, so the generated text is gibberish. It is used only to demonstrate the workflow quickly on CPU.
- Training (`trainer.train()` / `lora_trainer.train()`) is **intentionally not executed**. Uncomment the line in `main.py` to try it.
- The QLoRA section is documentation only because 4-bit quantization needs a GPU and `bitsandbytes`.
- The question-answering pipeline at the end is a stub. Uncomment and complete it to try the high-level `pipeline("question-answering", ...)` API.

## Project structure

```
technova-ai-assistant/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## License

MIT - see [LICENSE](LICENSE).
