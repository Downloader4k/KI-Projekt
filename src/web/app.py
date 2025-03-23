"""
app.py - Hauptwebserver für die KI-Anwendung
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Den Projektpfad zum Python-Pfad hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Orchestrator importieren (ersetzt direkte Verwendung der Persönlichkeitskomponente)
from src.orchestrator.orchestrator import Orchestrator

app = Flask(__name__)

# Orchestrator initialisieren
orchestrator = None

def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 'config', 'personality_config.json')
        orchestrator = Orchestrator(personality_config_path=config_path)
    return orchestrator

@app.route('/')
def index():
    """Hauptseite der Anwendung"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API-Endpunkt für Chat-Anfragen"""
    orchestrator = get_orchestrator()
        
    data = request.json
    user_input = data.get('message', '')
    
    # Anfrage durch Orchestrator verarbeiten
    result = orchestrator.process_query(user_input)
    
    return jsonify({
        'response': result['response'],
        'emotion': result['emotion'],
        'emotion_intensity': result['emotion_intensity'],
        'intent': result['intent']
    })

if __name__ == '__main__':
    app.run(debug=True)