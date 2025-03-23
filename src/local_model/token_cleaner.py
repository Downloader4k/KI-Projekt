# src/local_model/token_cleaner.py

import argparse
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os

# Logging einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("token_cleaning.log"), logging.StreamHandler()]
)
logger = logging.getLogger("token_cleaner")

def parse_args():
    parser = argparse.ArgumentParser(description="Entfernen chinesischer Token aus dem DeepSeek-Modell")
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
        default="./models/deepseek-german-cleaned",
        help="Ausgabeverzeichnis für das bereinigte Modell"
    )
    parser.add_argument(
        "--token_threshold",
        type=float,
        default=0.8,
        help="Schwellenwert für die Identifizierung chinesischer Token (0.0-1.0)"
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Nur Token analysieren, kein Modell speichern"
    )
    return parser.parse_args()

def is_chinese_char(c):
    """Überprüft, ob ein Zeichen chinesisch ist."""
    cp = ord(c)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  # CJK Unified Ideographs
        (cp >= 0x3400 and cp <= 0x4DBF) or  # CJK Unified Ideographs Extension A
        (cp >= 0x20000 and cp <= 0x2A6DF) or  # CJK Unified Ideographs Extension B
        (cp >= 0x2A700 and cp <= 0x2B73F) or  # CJK Unified Ideographs Extension C
        (cp >= 0x2B740 and cp <= 0x2B81F) or  # CJK Unified Ideographs Extension D
        (cp >= 0x2B820 and cp <= 0x2CEAF) or  # CJK Unified Ideographs Extension E
        (cp >= 0xF900 and cp <= 0xFAFF) or  # CJK Compatibility Ideographs
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  # CJK Compatibility Ideographs Supplement
        return True
    return False

def analyze_tokenizer(tokenizer, threshold=0.8):
    """
    Analysiert das Vokabular und identifiziert chinesische Token.
    """
    token_counts = {
        "chinese": 0,
        "non_chinese": 0,
        "mixed": 0
    }
    
    # Token-Informationen für die spätere Verwendung speichern
    token_info = []
    
    logger.info(f"Analysiere {len(tokenizer)} Token...")
    
    for i in range(len(tokenizer)):
        token = tokenizer.decode([i])
        chinese_chars = 0
        total_chars = 0
        
        for c in token:
            if c.strip():  # Nur nicht-Leerzeichen zählen
                total_chars += 1
                if is_chinese_char(c):
                    chinese_chars += 1
        
        if total_chars > 0:
            chinese_ratio = chinese_chars / total_chars
            
            if chinese_ratio > threshold:
                category = "chinese"
                token_counts["chinese"] += 1
            elif chinese_ratio > 0:
                category = "mixed"
                token_counts["mixed"] += 1
            else:
                category = "non_chinese"
                token_counts["non_chinese"] += 1
        else:
            category = "special"
            token_counts["non_chinese"] += 1  # Spezielle Token als nicht-chinesisch zählen
        
        token_info.append({
            "id": i,
            "token": token,
            "category": category,
            "chinese_ratio": chinese_ratio if total_chars > 0 else 0,
            "retain": category != "chinese" or i < 1000  # Spezielle und wichtige Token behalten
        })
    
    logger.info("Token-Analyse:")
    logger.info(f"Chinesisch-dominierte Token: {token_counts['chinese']}")
    logger.info(f"Gemischte Token: {token_counts['mixed']}")
    logger.info(f"Nicht-chinesische Token: {token_counts['non_chinese']}")
    
    return token_info

def main():
    args = parse_args()
    
    # Tokenizer laden
    logger.info(f"Lade Tokenizer von {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Token analysieren
    token_info = analyze_tokenizer(tokenizer, args.token_threshold)
    
    # Nur Analyse durchführen, wenn --analyze_only gesetzt ist
    if args.analyze_only:
        # Token-Informationen speichern
        output_file = "token_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(token_info, f, ensure_ascii=False, indent=2)
        logger.info(f"Token-Analyse gespeichert in {output_file}")
        return
    
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
    
    # Token zum Behalten identifizieren
    retained_token_ids = [info["id"] for info in token_info if info["retain"]]
    logger.info(f"Behalte {len(retained_token_ids)} von {len(tokenizer)} Token")
    
    # In dieser einfachen Implementierung erstellen wir ein neues Vokabular nicht.
    # In einer vollständigen Implementierung müssten wir:
    # 1. Ein neues Vokabular erstellen, das nur die zu behaltenden Token enthält
    # 2. Die Embedding-Matrix und den Output-Layer des Modells redimensionieren
    # 3. Das Modell mit dem neuen Vokabular neu initialisieren
    # Dies ist ein komplexer Prozess, der spezifisches Wissen über die Modellarchitektur erfordert.
    
    # Stattdessen speichern wir die Liste der zu behaltenden Token für spätere Verarbeitung
    retained_tokens_file = os.path.join(args.output_dir, "retained_tokens.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(retained_tokens_file, 'w', encoding='utf-8') as f:
        json.dump(retained_token_ids, f)
    
    logger.info(f"Liste der zu behaltenden Token gespeichert in {retained_tokens_file}")
    
    # Modell und Tokenizer speichern
    logger.info(f"Speichere Modell und Tokenizer in {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info(f"Modell gespeichert in {args.output_dir}")
    logger.info("Hinweis: Für die tatsächliche Entfernung der Token ist eine Neuinitialisierung des Tokenizers und eine Anpassung der Modellarchitektur erforderlich.")

if __name__ == "__main__":
    main()