# src/local_model/lora_training.py

import os
import argparse
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import json

# Logging einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger("lora_training")

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA-Training für DeepSeek 7B")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="deepseek-ai/deepseek-coder-7b-base",
        help="Pfad zum Basismodell oder Modellname"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/deepseek-german-lora",
        help="Ausgabeverzeichnis für das trainierte Modell"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Pfad zur Trainingsdatei im JSONL- oder Text-Format"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Anzahl der Trainingsepochen"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch-Größe pro Gerät für Training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Lernrate"
    )
    return parser.parse_args()

def load_training_data(data_path):
    """
    Lädt Trainingsdaten aus einer JSONL- oder Textdatei.
    """
    logger.info(f"Lade Trainingsdaten aus {data_path}")
    
    if data_path.endswith(".jsonl"):
        # JSONL-Format
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = [json.loads(line)["text"] for line in f]
    elif data_path.endswith(".txt"):
        # Einfache Textdatei, eine Zeile pro Text
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Versuche, als HuggingFace-Dataset zu laden
        try:
            dataset = load_dataset(data_path, split="train")
            texts = dataset["text"]
        except Exception as e:
            logger.error(f"Konnte Daten nicht laden: {e}")
            raise ValueError(f"Unbekanntes Datenformat: {data_path}")
    
    logger.info(f"Geladen: {len(texts)} Texte")
    return texts

def main():
    args = parse_args()
    
    # Trainingsdaten laden
    texts = load_training_data(args.data_path)
    
    # Tokenizer laden
    logger.info(f"Lade Tokenizer von {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Texte tokenisieren
    logger.info("Tokenisiere Texte")
    def tokenize_function(examples):
        # DeepSeek-spezifisches Format
        formatted_texts = [f"<｜begin▁of▁sentence｜>{text}<｜end▁of▁sentence｜>" for text in examples]
        return tokenizer(formatted_texts, padding="max_length", truncation=True, max_length=512)
    
    tokenized_texts = tokenize_function(texts)
    
    # Dataset erstellen
    train_dataset = Dataset.from_dict({
        "input_ids": tokenized_texts["input_ids"],
        "attention_mask": tokenized_texts["attention_mask"]
    })
    
    # Aufteilung in Trainings- und Validierungsdaten
    train_val_split = train_dataset.train_test_split(test_size=0.1)
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]
    
    logger.info(f"Trainingsdaten: {len(train_data)} Beispiele")
    logger.info(f"Validierungsdaten: {len(val_data)} Beispiele")
    
    # Modell laden
    logger.info(f"Lade Modell von {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ) if torch.cuda.is_available() else None
    )
    
    # Modell für LoRA-Training vorbereiten
    logger.info("Bereite Modell für LoRA-Training vor")
    model = prepare_model_for_kbit_training(model)
    
    # LoRA-Konfiguration
    logger.info("Konfiguriere LoRA")
    peft_config = LoraConfig(
        r=16,  # Rang der LoRA-Matrix
        lora_alpha=32,  # Skalierungsfaktor
        lora_dropout=0.05,  # Dropout-Rate
        bias="none",  # Keine Bias-Parameter anpassen
        task_type="CAUSAL_LM",  # Aufgabentyp: Kausales Sprachmodell
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Zielmodule für LoRA
    )
    
    # LoRA-Modell erstellen
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Trainingsargumente
    logger.info("Konfiguriere Training")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=10,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to="none",
        evaluation_strategy="steps",
        eval_steps=100,
    )
    
    # Trainer initialisieren
    logger.info("Initialisiere Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Training starten
    logger.info("Starte Training")
    trainer.train()
    
    # Modell speichern
    logger.info(f"Speichere Modell in {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training abgeschlossen")

if __name__ == "__main__":
    main()