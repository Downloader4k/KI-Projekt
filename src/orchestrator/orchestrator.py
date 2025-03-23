"""
orchestrator.py - Zentrale Komponente zur Steuerung der Anfragenverarbeitung

Diese Komponente koordiniert die verschiedenen Subsysteme und entscheidet, 
welches System für welche Anfrage zuständig ist.
"""

import sys
import os
import re
from typing import Dict, List, Any, Optional, Tuple

# Den Projektpfad zum Python-Pfad hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import der Persönlichkeitskomponente
from src.personality.personality import AIPersonality

# Import des ModelManagers
from src.local_model.model_manager import ModelManager

class IntentClassifier:
    """
    Klassifiziert die Benutzerabsicht, um die Anfrage an das richtige Subsystem weiterzuleiten.
    """
    
    def __init__(self):
        # Einfache regelbasierte Intents für den Anfang
        self.intent_patterns = {
            "gruss": [
                r"hallo", r"hi", r"hey", r"guten (morgen|tag|abend)", 
                r"servus", r"moin", r"grüß dich", r"wie geht'?s", r"wie geht es dir"
            ],
            "abschied": [
                r"tschüss", r"auf wiedersehen", r"bis später", r"bis bald",
                r"bye", r"ciao", r"man sieht sich", r"bis dann"
            ],
            "dank": [
                r"danke", r"vielen dank", r"herzlichen dank", r"besten dank",
                r"ich danke dir", r"thanks", r"merci"
            ],
            "hilfe": [
                r"hilfe", r"hilf mir", r"kannst du mir helfen", r"ich brauche hilfe",
                r"unterstützung", r"anleitung", r"wie funktionier(t|st)"
            ],
            "persönlich": [
                r"wie heißt du", r"wer bist du", r"stell dich vor", r"was kannst du",
                r"erzähl (mir )?(etwas )?(über )?dich", r"was bist du"
            ]
        }
        
        # Reguläre Ausdrücke für jedes Intent kompilieren
        self.compiled_patterns = {}
        for intent, patterns in self.intent_patterns.items():
            self.compiled_patterns[intent] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Klassifiziert den Text nach Benutzerabsicht.
        
        Args:
            text: Zu klassifizierender Text
            
        Returns:
            Tuple aus erkanntem Intent und Konfidenz (0.0-1.0)
        """
        # Standardwerte
        detected_intent = "unbekannt"
        max_confidence = 0.0
        
        # Text auf Pattern-Matches prüfen
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    # Einfaches Konfidenzmaß: Länge des Matches / Textlänge
                    matches = pattern.findall(text.lower())
                    match_length = sum(len(match) if isinstance(match, str) else len(match[0]) for match in matches)
                    confidence = min(1.0, match_length / len(text) * 2)  # *2 um Konfidenz zu verstärken
                    
                    if confidence > max_confidence:
                        detected_intent = intent
                        max_confidence = confidence
        
        return detected_intent, max_confidence


class MemoryManager:
    """
    Verwaltet das Kurzzeitgedächtnis für den Konversationskontext.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialisiert den Memory Manager.
        
        Args:
            max_history: Maximale Anzahl der zu speichernden Konversationspaare
        """
        self.conversation_history = []
        self.max_history = max_history
        
    def add_interaction(self, user_input: str, system_response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Fügt eine neue Interaktion zum Gedächtnis hinzu.
        
        Args:
            user_input: Benutzereingabe
            system_response: Systemantwort
            metadata: Zusätzliche Metadaten (z.B. Intent, Emotion)
        """
        if metadata is None:
            metadata = {}
            
        # Neues Konversationspaar hinzufügen
        self.conversation_history.append({
            "user_input": user_input,
            "system_response": system_response,
            "metadata": metadata,
            "timestamp": os.path.getmtime(__file__)  # Vereinfacht als aktuelle Zeit
        })
        
        # Älteste Einträge entfernen, wenn Maximum überschritten
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
            
    def get_recent_conversation(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        Gibt die letzten n Konversationspaare zurück.
        
        Args:
            n: Anzahl der zurückzugebenden Paare
            
        Returns:
            Liste der letzten n Konversationspaare
        """
        return self.conversation_history[-n:] if n <= len(self.conversation_history) else self.conversation_history
    
    def get_formatted_history(self, n: int = 3) -> str:
        """
        Gibt die letzten n Konversationspaare formatiert zurück.
        
        Args:
            n: Anzahl der zurückzugebenden Paare
            
        Returns:
            Formatierter Verlauf als String
        """
        recent = self.get_recent_conversation(n)
        formatted = []
        
        for item in recent:
            formatted.append(f"Benutzer: {item['user_input']}")
            formatted.append(f"System: {item['system_response']}")
            
        return "\n".join(formatted)
    
    def clear(self) -> None:
        """Löscht den gesamten Konversationsverlauf."""
        self.conversation_history = []


class Orchestrator:
    """
    Zentrale Komponente zur Koordination der verschiedenen Subsysteme.
    """
    
    def __init__(self, personality_config_path: Optional[str] = None, model_config_path: Optional[str] = None):
        """
        Initialisiert den Orchestrator.
        
        Args:
            personality_config_path: Pfad zur Persönlichkeitskonfiguration
            model_config_path: Pfad zur Modellkonfiguration
        """
        # Subsysteme initialisieren
        self.intent_classifier = IntentClassifier()
        self.memory_manager = MemoryManager()
        
        # Persönlichkeit laden
        if personality_config_path and os.path.exists(personality_config_path):
            self.personality = AIPersonality(config_path=personality_config_path)
        else:
            default_config = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                         'config', 'personality_config.json')
            if os.path.exists(default_config):
                self.personality = AIPersonality(config_path=default_config)
            else:
                self.personality = AIPersonality(name="Assistent")
        
        # Modell-Manager initialisieren
        if not model_config_path:
            model_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                           'config', 'model_config.json')
        
        self.model_manager = ModelManager(config_path=model_config_path)
    
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """
        Verarbeitet eine Benutzeranfrage und generiert eine Antwort.
        
        Args:
            user_input: Benutzereingabe
            
        Returns:
            Dict mit Antwort und Metadaten
        """
        # Intent klassifizieren
        intent, confidence = self.intent_classifier.classify(user_input)
        
        # Emotion erkennen
        emotion = self.personality.detect_emotion_from_text(user_input)
        
        # Gesprächsverlauf für Kontext abrufen
        conversation_history = self.memory_manager.get_formatted_history(3)
        
        # Systemanweisung mit Persönlichkeit und Emotion
        system_prompt = self.model_manager.config["system_prompt_template"].format(
            name=self.personality.name,
            emotion=self.personality.current_emotion
        )
        
        # Antwort generieren basierend auf Intent
        if intent == "gruss":
            base_response = f"Hallo! Ich bin {self.personality.name}. Wie kann ich dir heute helfen?"
        elif intent == "abschied":
            base_response = "Auf Wiedersehen! Es war schön, mit dir zu sprechen."
        elif intent == "dank":
            base_response = "Gerne! Ich freue mich, wenn ich helfen kann."
        elif intent == "hilfe":
            base_response = "Ich bin eine KI mit eigener Persönlichkeit und kann dir bei verschiedenen Aufgaben helfen. Was möchtest du wissen?"
        elif intent == "persönlich":
            base_response = f"Ich bin {self.personality.name}, eine KI mit eigener Persönlichkeit. Ich kann verschiedene Emotionen ausdrücken und lerne ständig dazu."
        else:
            # Für komplexere Anfragen das lokale Modell verwenden
            base_response = self.model_manager.generate_response(
                prompt=user_input,
                system_prompt=system_prompt,
                conversation_history=conversation_history
            )
        
        # Antwort basierend auf Persönlichkeit und Emotion anpassen
        adjusted_response = self.personality.adjust_response_for_emotion(base_response)
        
        # Interaktion zum Gedächtnis hinzufügen
        self.memory_manager.add_interaction(
            user_input=user_input,
            system_response=adjusted_response,
            metadata={
                "intent": intent,
                "confidence": confidence,
                "emotion": self.personality.current_emotion,
                "emotion_intensity": self.personality.emotion_intensity
            }
        )
        
        # Antwort und Metadaten zurückgeben
        return {
            "response": adjusted_response,
            "intent": intent,
            "confidence": confidence,
            "emotion": self.personality.current_emotion,
            "emotion_intensity": self.personality.emotion_intensity
        }


# Beispielverwendung
if __name__ == "__main__":
    orchestrator = Orchestrator()
    
    # Beispielanfragen testen
    test_inputs = [
        "Hallo, wie geht's?",
        "Ich muss über etwas nachdenken...",
        "Wow, das ist fantastisch!",
        "Wie heißt du eigentlich?",
        "Danke für deine Hilfe",
        "Tschüss, bis später!"
    ]
    
    for user_input in test_inputs:
        print(f"\nBenutzereingabe: {user_input}")
        result = orchestrator.process_query(user_input)
        print(f"Intent: {result['intent']} (Konfidenz: {result['confidence']:.2f})")
        print(f"Emotion: {result['emotion']} (Intensität: {result['emotion_intensity']:.2f})")
        print(f"Antwort: {result['response']}")
    
    # Konversationsverlauf anzeigen
    print("\nKonversationsverlauf:")
    print(orchestrator.memory_manager.get_formatted_history())