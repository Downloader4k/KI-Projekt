# src/local_model/model_quantizer.py

import argparse
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

# Logging einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("quantization.log"), logging.StreamHandler()]
)
logger = logging.getLogger("model_quantizer")

def parse_args():
    parser = argparse.ArgumentParser(description="Quantisieren des DeepSeek-Modells")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Pfad zum trainierten DeepSeek-Modell"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=False,
        help="Pfad zum LoRA-Modell (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/deepseek-german-quantized",
        help="Ausgabeverzeichnis für das quantisierte Modell"
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=8,
        help="Bittiefe für die Quantisierung (4 oder 8)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ausgabeverzeichnis erstellen
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tokenizer laden
    logger.info(f"Lade Tokenizer von {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Quantisierungskonfiguration
    if args.bits == 4:
        logger.info("Konfiguriere 4-bit Quantisierung")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        logger.info("Konfiguriere 8-bit Quantisierung")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Modell laden
    logger.info(f"Lade Modell von {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    # LoRA-Modell laden und mit Basismodell kombinieren, falls angegeben
    if args.lora_path:
        logger.info(f"Lade LoRA-Modell von {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        logger.info("Kombiniere Basis- und LoRA-Modell")
        model = model.merge_and_unload()
    
    # Modell und Tokenizer speichern
    logger.info(f"Speichere quantisiertes Modell in {args.output_dir}")
    model.save_pretrained(
        args.output_dir,
        quantization_config=quantization_config
    )
    tokenizer.save_pretrained(args.output_dir)
    
    # Quantisierungsinformationen speichern
    with open(os.path.join(args.output_dir, "quantization_info.txt"), 'w') as f:
        f.write(f"Quantisierung: {args.bits}-bit\n")
        f.write(f"Basismodell: {args.model_path}\n")
        if args.lora_path:
            f.write(f"LoRA-Modell: {args.lora_path}\n")
    
    logger.info(f"Modell erfolgreich quantisiert und gespeichert in {args.output_dir}")

if __name__ == "__main__":
    main()