"""
model_manager.py - Verwaltung des lokalen Sprachmodells

Diese Komponente kümmert sich um das Laden, die Initialisierung und die Verwendung
des lokalen DeepSeek-Modells für Inferenz, entweder direkt oder über den Inference-Server.
"""

import os
import sys
import json
import requests
import tempfile
import subprocess
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

# Den Projektpfad zum Python-Pfad hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Verwaltet das lokale Sprachmodell und stellt Schnittstellen für die Inferenz bereit.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 config_path: Optional[str] = None,
                 server_url: Optional[str] = None,
                 server_port: int = 5000):
        """
        Initialisiert den ModelManager.
        
        Args:
            model_path: Pfad zum lokalen Modell (GGUF-Format für DeepSeek)
            config_path: Pfad zur Modellkonfiguration
            server_url: URL des Inference-Servers (wenn vorhanden)
            server_port: Port für den lokalen Inference-Server
        """
        self.model_path = model_path
        self.config_path = config_path
        self.server_url = server_url
        self.server_port = server_port
        self.server_process = None
        self.model = None
        self.initialized = False
        
        # Konfiguration laden oder Standardkonfiguration verwenden
        self.config = self._load_config()
        
        # Modell-Stub (wird später durch das tatsächliche Modell ersetzt)
        self.model_loaded = False
        
        logger.info(f"ModelManager initialisiert. Modell wird bei Bedarf geladen.")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Lädt die Modellkonfiguration oder erstellt Standardwerte.
        
        Returns:
            Dict mit Konfigurationsparametern
        """
        default_config = {
            "model_type": "deepseek-7b",
            "context_size": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 512,
            "stop_sequences": ["Benutzer:", "\n\n"],
            "system_prompt_template": "Du bist ein hilfsbereicher Assistent mit dem Namen {name}. Deine aktuelle Emotionslage ist: {emotion}."
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Standardkonfiguration mit geladenen Werten überschreiben
                    default_config.update(loaded_config)
                    logger.info(f"Modellkonfiguration geladen aus: {self.config_path}")
            except Exception as e:
                logger.error(f"Fehler beim Laden der Konfiguration: {e}")
        
        return default_config
    
    def initialize_model(self) -> bool:
        """
        Initialisiert das Modell oder stellt eine Verbindung zum Inference-Server her.
        
        Returns:
            bool: True, wenn das Modell erfolgreich initialisiert wurde, sonst False
        """
        if self.initialized:
            return True
            
        try:
            # Wenn eine Server-URL angegeben ist, versuchen wir, eine Verbindung herzustellen
            if self.server_url:
                logger.info(f"Verbinde zu Inference-Server: {self.server_url}")
                
                # Gesundheitscheck des Servers
                try:
                    response = requests.get(f"{self.server_url}/health")
                    if response.status_code == 200:
                        logger.info("Verbindung zum Inference-Server hergestellt.")
                        self.initialized = True
                        return True
                    else:
                        logger.warning(f"Inference-Server antwortet nicht korrekt: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    logger.warning(f"Keine Verbindung zum Inference-Server möglich: {self.server_url}")
            
            # Wenn keine Server-URL angegeben ist oder die Verbindung fehlschlägt,
            # prüfen wir, ob die Modelldatei vorhanden ist
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Starte lokalen Inference-Server für Modell: {self.model_path}")
                
                # Lokalen Inference-Server starten
                server_script = os.path.join(os.path.dirname(__file__), 'inference_server.py')
                if os.path.exists(server_script):
                    cmd = [
                        sys.executable,
                        server_script,
                        "--model", self.model_path,
                        "--port", str(self.server_port)
                    ]
                    
                    self.server_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # Kurz warten, um zu sehen, ob der Server startet
                    time.sleep(2)
                    
                    if self.server_process.poll() is not None:
                        # Prozess hat sich bereits beendet
                        stderr = self.server_process.stderr.read()
                        logger.error(f"Inference-Server konnte nicht gestartet werden: {stderr}")
                    else:
                        # Server scheint zu laufen, URL setzen
                        self.server_url = f"http://localhost:{self.server_port}"
                        logger.info(f"Lokaler Inference-Server gestartet: {self.server_url}")
                        
                        # Warten, bis der Server bereit ist
                        for i in range(5):
                            try:
                                response = requests.get(f"{self.server_url}/health")
                                if response.status_code == 200:
                                    logger.info("Inference-Server ist bereit.")
                                    self.initialized = True
                                    return True
                            except requests.exceptions.ConnectionError:
                                time.sleep(1)
                        
                        logger.warning("Timeout beim Warten auf Inference-Server.")
            
            # Wenn kein Server und kein Modell verfügbar ist, fallen wir auf den Stub zurück
            logger.info("Verwende Stub-Implementation für Modell-Inferenz.")
            self.model_loaded = True
            self.initialized = True
            
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Modellinitialisierung: {e}")
            return False
    
    def generate_response(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None,
                          conversation_history: Optional[str] = None,
                          **kwargs) -> str:
        """
        Generiert eine Antwort mit dem lokalen Modell oder über den Inference-Server.
        
        Args:
            prompt: Hauptprompt (Benutzereingabe)
            system_prompt: Systemanweisung (optional)
            conversation_history: Bisheriger Gesprächsverlauf (optional)
            **kwargs: Weitere Parameter für die Inferenz
            
        Returns:
            Generierte Antwort als String
        """
        # Sicherstellen, dass das Modell initialisiert ist
        if not self.initialized and not self.initialize_model():
            return "Entschuldigung, ich kann derzeit keine Antwort generieren, da das lokale Modell nicht verfügbar ist."
        
        # Parameter für die Inferenz sammeln
        params = {
            "temperature": kwargs.get("temperature", self.config["temperature"]),
            "top_p": kwargs.get("top_p", self.config["top_p"]),
            "top_k": kwargs.get("top_k", self.config["top_k"]),
            "max_tokens": kwargs.get("max_tokens", self.config["max_tokens"]),
            "stop": kwargs.get("stop_sequences", self.config["stop_sequences"])
        }
        
        # Systemanweisung verwenden oder Standard aus Konfiguration
        if system_prompt is None:
            system_prompt = self.config["system_prompt_template"]
        
        # Gesamtprompt zusammenbauen
        full_prompt = system_prompt
        
        if conversation_history:
            full_prompt += f"\n\n{conversation_history}"
            
        full_prompt += f"\n\nBenutzer: {prompt}\nAssistent:"
        
        try:
            # Wenn wir eine Server-URL haben, verwenden wir den Inference-Server
            if self.server_url:
                data = {
                    "prompt": full_prompt,
                    **params
                }
                
                response = requests.post(
                    f"{self.server_url}/generate",
                    json=data,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "error" in result:
                        logger.warning(f"Fehler vom Inference-Server: {result['error']}")
                        return self._generate_stub_response(prompt)
                    return result.get("content", "")
                else:
                    logger.warning(f"Fehler bei der Anfrage an den Inference-Server: {response.status_code}")
                    return self._generate_stub_response(prompt)
            
            # Andernfalls verwenden wir den Stub
            return self._generate_stub_response(prompt)
            
        except Exception as e:
            logger.error(f"Fehler bei der Textgenerierung: {e}")
            return f"Entschuldigung, bei der Generierung ist ein Fehler aufgetreten: {str(e)}"
    
    def _generate_stub_response(self, prompt: str) -> str:
        """
        Generiert eine Stub-Antwort basierend auf der Eingabe.
        
        Args:
            prompt: Benutzereingabe
            
        Returns:
            Generierte Stub-Antwort
        """
        prompt_lower = prompt.lower()
        
        if "hallo" in prompt_lower or "hi" in prompt_lower:
            return "Hallo! Wie kann ich dir helfen?"
        elif "wer bist du" in prompt_lower:
            return "Ich bin ein lokales DeepSeek-Modell, das für Deutsch optimiert wurde. Ich kann dir bei verschiedenen Aufgaben helfen."
        elif "danke" in prompt_lower:
            return "Gerne! Ich freue mich, wenn ich helfen kann."
        elif "tschüss" in prompt_lower or "auf wiedersehen" in prompt_lower:
            return "Auf Wiedersehen! Es war schön, mit dir zu sprechen."
        else:
            return f"Ich habe deinen Prompt verstanden: '{prompt}'. Als lokales Modell würde ich hier eine hilfreiche Antwort generieren."
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Erzeugt ein Embedding für den angegebenen Text. In der tatsächlichen 
        Implementierung würde hier ein Embedding-Modell verwendet werden.
        
        Args:
            text: Text, für den ein Embedding erzeugt werden soll
            
        Returns:
            Liste von Float-Werten (Embedding-Vektor)
        """
        # Stub für Embedding-Funktionalität
        logger.info(f"[STUB] Embedding würde erstellt für: {text[:50]}...")
        
        # Dummy-Embedding mit 384 Dimensionen zurückgeben
        import random
        return [random.random() for _ in range(384)]
    
    def cleanup(self):
        """
        Bereinigt Ressourcen, insbesondere den Inference-Server, falls gestartet.
        """
        if self.server_process and self.server_process.poll() is None:
            logger.info("Stoppe Inference-Server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                logger.warning("Inference-Server musste hart beendet werden.")


# Beispielverwendung
if __name__ == "__main__":
    model_manager = ModelManager()
    
    # Beispielanfragen testen
    test_prompts = [
        "Hallo, wie geht's?",
        "Wer bist du eigentlich?",
        "Erkläre mir bitte, wie Photosynthese funktioniert.",
        "Danke für deine Hilfe",
        "Tschüss, bis später!"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = model_manager.generate_response(
            prompt=prompt,
            system_prompt="Du bist ein hilfsberecher Assistent. Deine Antworten sind klar und präzise."
        )
        print(f"Antwort: {response}")
    
    # Ressourcen bereinigen
    model_manager.cleanup()