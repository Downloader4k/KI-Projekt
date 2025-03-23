"""
inference_server.py - Lokaler Inference-Server für das DeepSeek-Modell

Diese Komponente startet einen lokalen Server, der Zugriff auf das 
trainierte und quantisierte DeepSeek-Modell bietet.
"""

import os
import argparse
import json
import time
from typing import Dict, List, Any, Optional
import logging
from flask import Flask, request, jsonify
import threading
import subprocess
import requests

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMInferenceServer:
    """
    Server für die Inferenz mit dem lokalen DeepSeek-Modell über llama.cpp.
    """
    
    def __init__(self, 
                 model_path: str, 
                 server_port: int = 8080,
                 context_size: int = 2048,
                 n_gpu_layers: int = -1):
        """
        Initialisiert den Inference-Server.
        
        Args:
            model_path: Pfad zur GGUF-Modelldatei
            server_port: Port für den Server
            context_size: Kontextfenstergröße
            n_gpu_layers: Anzahl der GPU-Layer (-1 für automatisch)
        """
        self.model_path = model_path
        self.server_port = server_port
        self.context_size = context_size
        self.n_gpu_layers = n_gpu_layers
        self.server_process = None
        self.is_running = False
        
        # Überprüfen, ob die Modelldatei existiert
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelldatei nicht gefunden: {model_path}")
            
        logger.info(f"LLM-Inference-Server initialisiert. Modell: {model_path}")
    
    def start_server(self) -> bool:
        """
        Startet den llama.cpp-Server.
        
        Returns:
            bool: True wenn erfolgreich gestartet, sonst False
        """
        if self.is_running:
            logger.warning("Server läuft bereits.")
            return True
            
        # Pfad zum llama.cpp-Server
        # In einem vollständigen Setup müsste dieser Pfad angepasst werden
        server_cmd = "./llama.cpp/server"
        
        # Wenn der Pfad nicht existiert, suchen wir nach alternativen Orten
        if not os.path.exists(server_cmd):
            alternative_paths = [
                "/usr/local/bin/llama-server",
                os.path.expanduser("~/llama.cpp/server"),
                os.path.expanduser("~/bin/llama-server")
            ]
            
            for path in alternative_paths:
                if os.path.exists(path):
                    server_cmd = path
                    break
            else:
                logger.error("llama.cpp Server nicht gefunden. Bitte installieren Sie llama.cpp.")
                return False
        
        # Server-Befehl zusammenbauen
        cmd = [
            server_cmd,
            "-m", self.model_path,
            "-c", str(self.context_size),
            "--port", str(self.server_port),
            "-ngl", str(self.n_gpu_layers)
        ]
        
        logger.info(f"Starte Server mit Befehl: {' '.join(cmd)}")
        
        try:
            # Server im Hintergrund starten
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
                logger.error(f"Server konnte nicht gestartet werden: {stderr}")
                return False
                
            # Testen, ob der Server erreichbar ist
            retries = 5
            for i in range(retries):
                try:
                    health_check = requests.get(f"http://localhost:{self.server_port}/health")
                    if health_check.status_code == 200:
                        self.is_running = True
                        logger.info(f"Server erfolgreich gestartet und läuft auf Port {self.server_port}")
                        return True
                except requests.exceptions.ConnectionError:
                    if i < retries - 1:
                        logger.info(f"Warte auf Server-Start... ({i+1}/{retries})")
                        time.sleep(2)
                    else:
                        logger.warning("Server läuft möglicherweise, konnte aber nicht erreicht werden.")
                        self.is_running = True  # Wir nehmen an, dass er läuft
                        return True
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Starten des Servers: {e}")
            return False
    
    def stop_server(self) -> bool:
        """
        Stoppt den llama.cpp-Server.
        
        Returns:
            bool: True wenn erfolgreich gestoppt, sonst False
        """
        if not self.is_running or self.server_process is None:
            logger.warning("Server läuft nicht.")
            return True
            
        try:
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
            self.is_running = False
            logger.info("Server erfolgreich gestoppt.")
            return True
        except subprocess.TimeoutExpired:
            self.server_process.kill()
            logger.warning("Server musste hart beendet werden.")
            self.is_running = False
            return True
        except Exception as e:
            logger.error(f"Fehler beim Stoppen des Servers: {e}")
            return False
    
    def generate_completion(self, 
                           prompt: str, 
                           max_tokens: int = 512,
                           temperature: float = 0.7,
                           top_p: float = 0.9,
                           top_k: int = 40,
                           stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generiert eine Vervollständigung mit dem lokalen Modell.
        
        Args:
            prompt: Prompt für die Generierung
            max_tokens: Maximale Anzahl der zu generierenden Token
            temperature: Temperatur für die Generierung (höher = kreativer)
            top_p: Top-p Sampling Parameter
            top_k: Top-k Sampling Parameter
            stop: Stopsequenzen
            
        Returns:
            Dict mit generiertem Text und Metadaten
        """
        if not self.is_running:
            if not self.start_server():
                return {"error": "Server konnte nicht gestartet werden."}
        
        # Anfrage an den llama.cpp-Server
        url = f"http://localhost:{self.server_port}/completion"
        
        data = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop or []
        }
        
        try:
            response = requests.post(
                url, 
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "content": result.get("content", ""),
                    "model": result.get("model", ""),
                    "total_duration": result.get("total_duration", 0),
                    "prompt_eval_duration": result.get("prompt_eval_duration", 0),
                    "eval_duration": result.get("eval_duration", 0),
                    "eval_count": result.get("eval_count", 0)
                }
            else:
                logger.error(f"Fehler bei der Generierung: {response.status_code} - {response.text}")
                return {"error": f"Fehler bei der Generierung: {response.status_code}"}
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Verbindung zum Server fehlgeschlagen.")
            return {"error": "Verbindung zum Server fehlgeschlagen."}
        except Exception as e:
            logger.error(f"Fehler bei der Generierung: {e}")
            return {"error": f"Fehler bei der Generierung: {str(e)}"}


# Flask-Server für die Verwendung des LLM-Servers
app = Flask(__name__)
llm_server = None

@app.route("/generate", methods=["POST"])
def generate():
    """API-Endpunkt für Textgenerierung."""
    global llm_server
    if llm_server is None:
        return jsonify({"error": "LLM-Server nicht initialisiert."}), 500
        
    data = request.json
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "Kein Prompt angegeben."}), 400
        
    # Optionale Parameter
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    top_k = data.get("top_k", 40)
    stop = data.get("stop", [])
    
    # Generierung durchführen
    result = llm_server.generate_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop
    )
    
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    """Gesundheitscheck-Endpunkt."""
    global llm_server
    if llm_server is None:
        return jsonify({"status": "error", "message": "LLM-Server nicht initialisiert."}), 500
    
    if not llm_server.is_running:
        return jsonify({"status": "error", "message": "LLM-Server läuft nicht."}), 500
        
    return jsonify({"status": "ok", "message": "LLM-Server bereit."})

def main():
    """Hauptfunktion zum Starten des Servers."""
    parser = argparse.ArgumentParser(description="LLM Inference Server")
    parser.add_argument("--model", required=True, help="Pfad zur GGUF-Modelldatei")
    parser.add_argument("--port", type=int, default=5000, help="Port für den Flask-Server")
    parser.add_argument("--llm-port", type=int, default=8080, help="Port für den llama.cpp-Server")
    parser.add_argument("--context-size", type=int, default=2048, help="Kontextfenstergröße")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Anzahl der GPU-Layer (-1 für auto)")
    
    args = parser.parse_args()
    
    global llm_server
    llm_server = LLMInferenceServer(
        model_path=args.model,
        server_port=args.llm_port,
        context_size=args.context_size,
        n_gpu_layers=args.n_gpu_layers
    )
    
    # Server starten
    if not llm_server.start_server():
        logger.error("LLM-Server konnte nicht gestartet werden. Beende.")
        return
    
    # Shutdown-Handler hinzufügen
    def shutdown_handler():
        if llm_server:
            llm_server.stop_server()
    
    import atexit
    atexit.register(shutdown_handler)
    
    # Flask-Server starten
    app.run(host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()