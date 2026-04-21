import re
from .config import RELATION_PATTERNS, CONFIDENCE_THRESHOLD, RELATION_TYPES


class RelationExtractor:
    def __init__(self):
        """Initialize Relation Extractor"""
        print("Initializing Relation Extractor...")
        self.patterns = RELATION_PATTERNS
        print("✅ Relation Extractor initialized")
    
    def extract_relations(self, text, entities):
        """
        Extract relations between entities
        
        Args:
            text (str): Vietnamese text
            entities (list): List of entities from NER
                [
                    {"text": "Nhân vật A", "type": "PERSON"},
                    {"text": "Nhân vật B", "type": "PERSON"}
                ]
        
        Returns:
            list: List of relations:
                [
                    {
                        "source": "Nhân vật A",
                        "target": "Nhân vật B",
                        "relation_type": "DEFEATS",
                        "confidence": 0.95
                    }
                ]
        
        Example:
            >>> extractor = RelationExtractor()
            >>> text = "Nhân vật A đánh bại nhân vật B"
            >>> entities = [
            ...     {"text": "Nhân vật A", "type": "PERSON"},
            ...     {"text": "Nhân vật B", "type": "PERSON"}
            ... ]
            >>> relations = extractor.extract_relations(text, entities)
            >>> print(relations)
            [{"source": "Nhân vật A", "target": "Nhân vật B", "relation_type": "DEFEATS", "confidence": 0.95}]
        """
        if not text or not entities:
            return []
        
        relations = []
        
        try:
            # Try all entity pairs
            for i, ent1 in enumerate(entities):
                for ent2 in entities[i+1:]:
                    # Check relation from ent1 to ent2
                    rel = self.predict_relation(
                        ent1["text"],
                        ent2["text"],
                        text
                    )
                    
                    if rel and rel["relation_type"] != "NO_RELATION":
                        relations.append({
                            "source": ent1["text"],
                            "target": ent2["text"],
                            "relation_type": rel["relation_type"],
                            "confidence": rel["confidence"]
                        })
                    
                    # Check relation from ent2 to ent1
                    rel = self.predict_relation(
                        ent2["text"],
                        ent1["text"],
                        text
                    )
                    
                    if rel and rel["relation_type"] != "NO_RELATION":
                        relations.append({
                            "source": ent2["text"],
                            "target": ent1["text"],
                            "relation_type": rel["relation_type"],
                            "confidence": rel["confidence"]
                        })
            
            return relations
        
        except Exception as e:
            print(f"Error in extract_relations: {e}")
            return []
    
    def predict_relation(self, entity1, entity2, text):
        """
        Predict relation between 2 entities
        
        Args:
            entity1 (str): First entity
            entity2 (str): Second entity
            text (str): Original text
        
        Returns:
            dict: {"relation_type": "...", "confidence": ...}
        
        Example:
            >>> extractor = RelationExtractor()
            >>> rel = extractor.predict_relation("Nhân vật A", "Nhân vật B", "Nhân vật A đánh bại nhân vật B")
            >>> print(rel)
            {"relation_type": "DEFEATS", "confidence": 0.95}
        """
        try:
            # Find positions of entities in text
            pos1 = text.find(entity1)
            pos2 = text.find(entity2)
            
            if pos1 == -1 or pos2 == -1:
                return {"relation_type": "NO_RELATION", "confidence": 0.0}
            
            # Extract text between entities
            if pos1 < pos2:
                between_text = text[pos1 + len(entity1):pos2]
            else:
                between_text = text[pos2 + len(entity2):pos1]
            
            # Check patterns
            for relation_type, keywords in self.patterns.items():
                for keyword in keywords:
                    if keyword.lower() in between_text.lower():
                        confidence = 0.95 if pos1 < pos2 else 0.85
                        return {
                            "relation_type": relation_type,
                            "confidence": confidence
                        }
            
            return {"relation_type": "NO_RELATION", "confidence": 0.0}
        
        except Exception as e:
            print(f"Error in predict_relation: {e}")
            return {"relation_type": "NO_RELATION", "confidence": 0.0}
    
    def get_relation_types(self):
        """Get list of supported relation types"""
        return list(RELATION_TYPES.keys())


# Test code
if __name__ == "__main__":
    # Initialize
    extractor = RelationExtractor()
    
    # Test data
    text = "Nhân vật A đánh bại nhân vật B và lấy được bảo vật"
    entities = [
        {"text": "Nhân vật A", "type": "PERSON"},
        {"text": "Nhân vật B", "type": "PERSON"},
        {"text": "bảo vật", "type": "OBJECT"}
    ]
    
    # Extract relations
    relations = extractor.extract_relations(text, entities)
    print(f"Relations: {relations}")