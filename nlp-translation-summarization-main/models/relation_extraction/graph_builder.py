class GraphBuilder:
    def __init__(self):
        """Initialize Graph Builder"""
        print("Initializing Graph Builder...")
        print("✅ Graph Builder initialized")
    
    def build_graph(self, entities, relations):
        """
        Build knowledge graph from entities and relations
        
        Args:
            entities (list): List of entities:
                [
                    {"text": "Nhân vật A", "type": "PERSON"},
                    {"text": "Hà Nội", "type": "LOCATION"}
                ]
            relations (list): List of relations:
                [
                    {
                        "source": "Nhân vật A",
                        "target": "Nhân vật B",
                        "relation_type": "DEFEATS",
                        "confidence": 0.95
                    }
                ]
        
        Returns:
            dict: Knowledge graph structure:
                {
                    "nodes": [...],
                    "edges": [...],
                    "node_count": N,
                    "edge_count": M
                }
        
        Example:
            >>> builder = GraphBuilder()
            >>> entities = [
            ...     {"text": "Nhân vật A", "type": "PERSON"},
            ...     {"text": "Nhân vật B", "type": "PERSON"}
            ... ]
            >>> relations = [
            ...     {"source": "Nhân vật A", "target": "Nhân vật B", "relation_type": "DEFEATS", "confidence": 0.95}
            ... ]
            >>> graph = builder.build_graph(entities, relations)
            >>> print(graph)
        """
        try:
            # Create nodes from entities
            nodes = []
            entity_to_id = {}  # Map entity text to node id
            
            for idx, entity in enumerate(entities):
                node = {
                    "id": str(idx),
                    "label": entity["text"],
                    "type": entity["type"]
                }
                nodes.append(node)
                entity_to_id[entity["text"]] = str(idx)
            
            # Create edges from relations
            edges = []
            
            for relation in relations:
                source_id = entity_to_id.get(relation["source"])
                target_id = entity_to_id.get(relation["target"])
                
                if source_id and target_id:
                    edge = {
                        "source": source_id,
                        "target": target_id,
                        "label": relation["relation_type"],
                        "confidence": relation["confidence"]
                    }
                    edges.append(edge)
            
            # Build graph
            graph = {
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
            
            return graph
        
        except Exception as e:
            print(f"Error in build_graph: {e}")
            return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}


# Test code
if __name__ == "__main__":
    # Initialize
    builder = GraphBuilder()
    
    # Test data
    entities = [
        {"text": "Nhân vật A", "type": "PERSON"},
        {"text": "Nhân vật B", "type": "PERSON"},
        {"text": "bảo vật", "type": "OBJECT"}
    ]
    
    relations = [
        {"source": "Nhân vật A", "target": "Nhân vật B", "relation_type": "DEFEATS", "confidence": 0.95},
        {"source": "Nhân vật A", "target": "bảo vật", "relation_type": "OBTAINS", "confidence": 0.88}
    ]
    
    # Build graph
    graph = builder.build_graph(entities, relations)
    print(f"Graph: {graph}")