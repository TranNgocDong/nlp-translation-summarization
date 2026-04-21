import json
import csv


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
    
    def export_json(self, graph, filepath):
        """
        Export graph to JSON file
        
        Args:
            graph (dict): Knowledge graph
            filepath (str): Output file path
        
        Returns:
            bool: Success status
        
        Example:
            >>> builder = GraphBuilder()
            >>> graph = builder.build_graph(entities, relations)
            >>> builder.export_json(graph, "graph.json")
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            print(f"✅ Graph exported to {filepath}")
            return True
        
        except Exception as e:
            print(f"Error in export_json: {e}")
            return False
    
    def export_csv(self, graph, nodes_path, edges_path):
        """
        Export graph to CSV files
        
        Args:
            graph (dict): Knowledge graph
            nodes_path (str): Output path for nodes CSV
            edges_path (str): Output path for edges CSV
        
        Returns:
            bool: Success status
        
        Example:
            >>> builder = GraphBuilder()
            >>> graph = builder.build_graph(entities, relations)
            >>> builder.export_csv(graph, "nodes.csv", "edges.csv")
        """
        try:
            # Export nodes
            with open(nodes_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'label', 'type'])
                writer.writeheader()
                writer.writerows(graph['nodes'])
            print(f"✅ Nodes exported to {nodes_path}")
            
            # Export edges
            with open(edges_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['source', 'target', 'label', 'confidence'])
                writer.writeheader()
                writer.writerows(graph['edges'])
            print(f"✅ Edges exported to {edges_path}")
            
            return True
        
        except Exception as e:
            print(f"Error in export_csv: {e}")
            return False


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
    
    # Export
    builder.export_json(graph, "graph.json")
    builder.export_csv(graph, "nodes.csv", "edges.csv")