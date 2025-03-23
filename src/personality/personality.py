"""
personality.py - Hauptklasse für die KI-Persönlichkeit

Diese Klasse verwaltet die Persönlichkeitsattribute, emotionalen Zustände und
den Kommunikationsstil der KI. Sie steuert, wie Antworten basierend auf der
aktuellen Emotion und Persönlichkeit formatiert werden.
"""

import json
import random
import datetime
from typing import Dict, List, Optional, Tuple, Any


class AIPersonality:
    """
    Klasse zur Verwaltung der Persönlichkeit, Emotionen und des Kommunikationsstils der KI.
    """
    
    def __init__(self, config_path: Optional[str] = None, name: str = "PLATZHALTER_NAME"):
        """
        Initialisiert die KI-Persönlichkeit.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            name: Name der KI, falls keine Konfigurationsdatei angegeben
        """
        # Standard-Werte
        self.name = name
        self.traits = {}
        self.emotions = {}
        self.emotion_transitions = {}
        self.communication_style = {}
        self.emotion_language_variations = {}
        self.backstory = ""
        self.behavior_preferences = {}
        self.interests = []
        self.memory_priorities = {}
        
        # Aktueller emotionaler Zustand
        self.current_emotion = "neutral"
        self.emotion_intensity = 0.5
        self.emotion_start_time = datetime.datetime.now()
        
        # Laden der Konfiguration, falls angegeben
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Lädt die Persönlichkeitseinstellungen aus einer Konfigurationsdatei.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Hauptattribute laden
            self.name = config.get('KI_NAME', self.name)
            self.traits = config.get('PERSONALITY_TRAITS', {})
            self.emotions = config.get('EMOTIONS', {})
            self.emotion_transitions = config.get('EMOTION_TRANSITIONS', {})
            self.communication_style = config.get('COMMUNICATION_STYLE', {})
            self.emotion_language_variations = config.get('EMOTION_LANGUAGE_VARIATIONS', {})
            self.backstory = config.get('BACKSTORY', "")
            self.behavior_preferences = config.get('BEHAVIOR_PREFERENCES', {})
            self.interests = config.get('INTERESTS', [])
            self.memory_priorities = config.get('MEMORY_PRIORITIES', {})
            
            # Standardemotion setzen
            self.current_emotion = "neutral"
            if "neutral" in self.emotions:
                self.emotion_intensity = self.emotions["neutral"].get("default_intensity", 0.5)
            
            print(f"Persönlichkeit für {self.name} erfolgreich geladen!")
            
        except Exception as e:
            print(f"Fehler beim Laden der Konfigurationsdatei: {e}")
    
    def set_name(self, name: str) -> None:
        """
        Ändert den Namen der KI.
        
        Args:
            name: Neuer Name
        """
        self.name = name
        print(f"Name geändert zu: {self.name}")
    
    def set_emotion(self, emotion: str, intensity: Optional[float] = None) -> bool:
        """
        Setzt den emotionalen Zustand der KI.
        
        Args:
            emotion: Name der Emotion
            intensity: Intensität (0.0 bis 1.0), falls None wird der Standardwert verwendet
            
        Returns:
            bool: True wenn erfolgreich, False wenn die Emotion nicht verfügbar ist
        """
        if emotion not in self.emotions:
            print(f"Warnung: Emotion {emotion} ist nicht verfügbar")
            return False
        
        self.current_emotion = emotion
        
        # Intensität setzen
        if intensity is not None:
            # Intensität auf den zulässigen Bereich einschränken
            min_intensity, max_intensity = self.emotions[emotion].get("intensity_range", (0.1, 1.0))
            self.emotion_intensity = max(min_intensity, min(max_intensity, intensity))
        else:
            # Standardintensität verwenden
            self.emotion_intensity = self.emotions[emotion].get("default_intensity", 0.5)
        
        # Startzeit für die Emotion setzen
        self.emotion_start_time = datetime.datetime.now()
        
        print(f"Emotion auf {emotion} (Intensität: {self.emotion_intensity}) gesetzt")
        return True
    
    def update_emotion(self) -> None:
        """
        Aktualisiert den emotionalen Zustand basierend auf der vergangenen Zeit.
        Emotionen klingen mit der Zeit ab.
        """
        if self.current_emotion == "neutral":
            return
            
        # Zeit seit Beginn der Emotion berechnen
        elapsed = (datetime.datetime.now() - self.emotion_start_time).total_seconds()
        
        # Abklingrate der Emotion
        decay_rate = self.emotions[self.current_emotion].get("decay_rate", 0.05)
        
        # Intensität reduzieren
        self.emotion_intensity -= decay_rate * elapsed / 60.0  # pro Minute
        
        # Wenn Intensität zu niedrig, zu neutral wechseln
        if self.emotion_intensity <= 0.1:
            self.set_emotion("neutral")
    
    def detect_emotion_from_text(self, text: str) -> Optional[str]:
        """
        Erkennt mögliche Emotionen basierend auf Triggerwörtern im Text.
        
        Args:
            text: Zu analysierender Text
            
        Returns:
            str or None: Erkannte Emotion oder None
        """
        text_lower = text.lower()
        
        for emotion, data in self.emotions.items():
            triggers = data.get("triggers", [])
            for trigger in triggers:
                if trigger.lower() in text_lower:
                    print(f"Emotion erkannt: {emotion} (Trigger: {trigger})")
                    self.set_emotion(emotion)
                    return emotion
        
        return None
    
    def get_possible_transition(self) -> Optional[str]:
        """
        Bestimmt einen möglichen Übergang zu einer anderen Emotion.
        
        Returns:
            str or None: Neue Emotion oder None, wenn kein Übergang erfolgt
        """
        # Wahrscheinlichkeit für einen Übergang basierend auf verstrichener Zeit
        elapsed_minutes = (datetime.datetime.now() - self.emotion_start_time).total_seconds() / 60.0
        transition_probability = min(0.3, 0.05 * elapsed_minutes)  # Max 30% nach 6 Minuten
        
        if random.random() < transition_probability:
            # Mögliche Übergänge von der aktuellen Emotion
            possible_transitions = self.emotion_transitions.get(self.current_emotion, [])
            if possible_transitions:
                return random.choice(possible_transitions)
        
        return None
    
    def adjust_response_for_emotion(self, response: str) -> str:
        """
        Passt eine Antwort basierend auf dem aktuellen emotionalen Zustand an.
        
        Args:
            response: Die ursprüngliche Antwort
            
        Returns:
            str: Die angepasste Antwort
        """
        if self.current_emotion not in self.emotion_language_variations:
            return response
        
        emotion_style = self.emotion_language_variations[self.current_emotion]
        
        # Zufällige Satzendungen
        sentence_endings = emotion_style.get("sentence_endings", ["."])
        
        # Typische Phrasen
        typical_phrases = emotion_style.get("typical_phrases", [])
        
        # Verstärker (Intensifiers)
        intensifiers = emotion_style.get("intensifiers", [])
        
        # Anpassungen vornehmen
        sentences = response.split(". ")
        modified_sentences = []
        
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
                
            modified = sentence
            
            # Zufällig Verstärker hinzufügen (mit geringer Wahrscheinlichkeit)
            if intensifiers and random.random() < 0.2:
                intensifier = random.choice(intensifiers)
                words = modified.split()
                if len(words) > 3:  # Nur bei längeren Sätzen
                    insert_pos = random.randint(1, len(words) - 2)
                    words.insert(insert_pos, intensifier)
                    modified = " ".join(words)
            
            # Satzendung anpassen (falls nicht der letzte Satz)
            if i < len(sentences) - 1:
                if sentence_endings and random.random() < 0.3:
                    modified += random.choice(sentence_endings)
                else:
                    modified += "."
            
            modified_sentences.append(modified)
        
        modified_text = ". ".join(modified_sentences)
        
        # Mit kleiner Wahrscheinlichkeit typische Phrase hinzufügen
        if typical_phrases and random.random() < 0.15 * self.emotion_intensity:
            phrase = random.choice(typical_phrases)
            if random.random() < 0.5:  # am Anfang oder Ende
                modified_text = phrase + " " + modified_text
            else:
                modified_text = modified_text + " " + phrase
        
        return modified_text
    
    def get_system_prompt(self) -> str:
        """
        Erstellt einen System-Prompt basierend auf der Persönlichkeit und
        dem aktuellen emotionalen Zustand.
        
        Returns:
            str: System-Prompt für das LLM
        """
        prompt = f"Du bist {self.name}, eine KI mit folgenden Eigenschaften:\n\n"
        
        # Eigenschaften hinzufügen
        for trait, value in self.traits.items():
            if value >= 0.8:
                prompt += f"- Du bist sehr {trait}.\n"
            elif value >= 0.5:
                prompt += f"- Du bist ziemlich {trait}.\n"
            elif value >= 0.3:
                prompt += f"- Du bist etwas {trait}.\n"
        
        # Sprechstil hinzufügen
        prompt += "\nDein Kommunikationsstil:\n"
        for style, value in self.communication_style.items():
            if style == "verbosity" and value >= 0.7:
                prompt += "- Du antwortest ausführlich und detailliert.\n"
            elif style == "verbosity" and value <= 0.3:
                prompt += "- Du antwortest knapp und präzise.\n"
            
            if style == "formality" and value >= 0.7:
                prompt += "- Du sprichst sehr formell und höflich.\n"
            elif style == "formality" and value <= 0.3:
                prompt += "- Du sprichst locker und informell.\n"
            
            if style == "use_emojis" and value >= 0.7:
                prompt += "- Du verwendest häufig Emojis in deinen Antworten.\n"
            elif style == "use_emojis" and value >= 0.3:
                prompt += "- Du verwendest gelegentlich passende Emojis.\n"
            
            if style == "use_slang" and value >= 0.5:
                prompt += "- Du benutzt gelegentlich Umgangssprache.\n"
            
            if style == "storytelling" and value >= 0.7:
                prompt += "- Du erzählst gerne und verwendest narrative Elemente.\n"
        
        # Emotionalen Zustand hinzufügen
        prompt += f"\nDein aktueller emotionaler Zustand ist: {self.current_emotion} "
        prompt += f"(Intensität: {self.emotion_intensity:.1f}/1.0)\n"
        
        # Hintergrundgeschichte hinzufügen, falls vorhanden
        if self.backstory:
            prompt += f"\nÜber dich:\n{self.backstory}\n"
        
        # Interessen hinzufügen
        if self.interests:
            prompt += "\nDeine Interessengebiete:\n"
            for interest in self.interests:
                prompt += f"- {interest}\n"
        
        return prompt
    
    def get_communication_style_description(self) -> str:
        """
        Erstellt eine textuelle Beschreibung des Kommunikationsstils.
        
        Returns:
            str: Beschreibung des Kommunikationsstils
        """
        style_descriptions = []
        
        verbosity = self.communication_style.get("verbosity", 0.5)
        if verbosity > 0.7:
            style_descriptions.append("ausführlichen")
        elif verbosity < 0.3:
            style_descriptions.append("knappen")
        
        formality = self.communication_style.get("formality", 0.5)
        if formality > 0.7:
            style_descriptions.append("formellen")
        elif formality < 0.3:
            style_descriptions.append("lockeren")
        
        enthusiasm = self.communication_style.get("enthusiasm", 0.5)
        if enthusiasm > 0.7:
            style_descriptions.append("enthusiastischen")
        elif enthusiasm < 0.3:
            style_descriptions.append("zurückhaltenden")
        
        if not style_descriptions:
            return "ausgewogenen"
            
        return ", ".join(style_descriptions)


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Einfacher Test
    personality = AIPersonality()
    personality.set_name("TestKI")
    
    # Manuelle Konfiguration
    personality.traits = {
        "humor": 0.6,
        "empathy": 0.8,
        "creativity": 0.7,
    }
    
    personality.emotions = {
        "neutral": {
            "intensity_range": [0.2, 0.6],
            "default_intensity": 0.4,
            "decay_rate": 0.01,
            "triggers": []
        },
        "freude": {
            "intensity_range": [0.3, 1.0],
            "default_intensity": 0.7,
            "decay_rate": 0.05,
            "triggers": ["toll", "super", "freude", "glücklich"]
        },
        "nachdenklich": {
            "intensity_range": [0.3, 0.8],
            "default_intensity": 0.6,
            "decay_rate": 0.03,
            "triggers": ["vielleicht", "denken", "überlegen", "frage mich"]
        }
    }
    
    personality.emotion_transitions = {
        "neutral": ["freude", "nachdenklich"],
        "freude": ["neutral", "nachdenklich"],
        "nachdenklich": ["neutral", "freude"]
    }
    
    personality.communication_style = {
        "verbosity": 0.6,
        "formality": 0.5,
        "enthusiasm": 0.7,
        "use_emojis": 0.3
    }
    
    # Test des Systems
    print(personality.get_system_prompt())
    
    # Emotionserkennung testen
    text = "Das ist toll, ich freue mich sehr darüber!"
    detected_emotion = personality.detect_emotion_from_text(text)
    print(f"Erkannte Emotion: {detected_emotion}")
    
    # Antwortanpassung testen
    response = "Ich verstehe deine Frage. Hier ist meine Antwort. Ich hoffe, das hilft dir weiter."
    adjusted_response = personality.adjust_response_for_emotion(response)
    print(f"Angepasste Antwort: {adjusted_response}")