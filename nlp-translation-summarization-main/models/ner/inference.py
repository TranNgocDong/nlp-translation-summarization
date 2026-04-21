from underthesea import ner
from .config import ENTITY_TYPES, ENABLE_CACHING


class NERModel:
    def __init__(self):
        """Initialize NER model using Underthesea"""
        print("Initializing NER model...")
        self.cache = {} if ENABLE_CACHING else None
        print("✅ NER model initialized")
    
    def extract_entities(self, text):
        """
        Extract named entities from Vietnamese text
        
        Args:
            text (str): Vietnamese text to extract entities from
        
        Returns:
            list: List of entities with format:
                [
                    {"text": "Nhân vật A", "type": "PERSON"},
                    {"text": "Hà Nội", "type": "LOCATION"},
                    {"text": "bảo vật", "type": "OBJECT"}
                ]
        
        Example:
            >>> model = NERModel()
            >>> entities = model.extract_entities("Nhân vật A đến Hà Nội")
            >>> print(entities)
            [
                {"text": "Nhân vật A", "type": "PERSON"},
                {"text": "Hà Nội", "type": "LOCATION"}
            ]
        """
        if not text or not isinstance(text, str):
            return []
        
        # Check cache
        if ENABLE_CACHING and text in self.cache:
            return self.cache[text]
        
        try:
            # Extract using underthesea
            entities_raw = ner(text)
            
            # Format output
            entities = []
            for entity_text, entity_type in entities_raw:
                entities.append({
                    "text": entity_text,
                    "type": entity_type
                })
            
            # Cache result
            if ENABLE_CACHING:
                self.cache[text] = entities
            
            return entities
        
        except Exception as e:
            print(f"Error in extract_entities: {e}")
            return []
    
    def extract_with_positions(self, text):
        """
        Extract entities with their positions in text
        
        Args:
            text (str): Vietnamese text
        
        Returns:
            list: List of entities with positions:
                [
                    {
                        "text": "Nhân vật A",
                        "type": "PERSON",
                        "start": 0,
                        "end": 9
                    },
                    ...
                ]
        
        Example:
            >>> model = NERModel()
            >>> entities = model.extract_with_positions("Nhân vật A là chiến binh")
            >>> print(entities)
            [{"text": "Nhân vật A", "type": "PERSON", "start": 0, "end": 9}]
        """
        if not text or not isinstance(text, str):
            return []
        
        try:
            # Extract entities
            entities_raw = ner(text)
            
            # Add positions
            entities = []
            for entity_text, entity_type in entities_raw:
                start_pos = text.find(entity_text)
                end_pos = start_pos + len(entity_text) if start_pos != -1 else -1
                
                entities.append({
                    "text": entity_text,
                    "type": entity_type,
                    "start": start_pos,
                    "end": end_pos
                })
            
            return entities
        
        except Exception as e:
            print(f"Error in extract_with_positions: {e}")
            return []
    
    def get_entity_types(self):
        """Get list of supported entity types"""
        return ENTITY_TYPES


# Test code
if __name__ == "__main__":
    # Initialize model
    model = NERModel()
    
    # Test extraction
    text = "Nhân vật A đến Hà Nội gặp công chúa B"
    print(f"Text: {text}")
    
    # Basic extraction
    entities = model.extract_entities(text)
    print(f"Entities: {entities}")
    
    # With positions
    entities_with_pos = model.extract_with_positions(text)
    print(f"Entities with positions: {entities_with_pos}")