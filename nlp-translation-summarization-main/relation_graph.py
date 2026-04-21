"""
Relation Graph Builder - Extract entities and relations from text
Combines NER, Relation Extraction, and Graph Building into a unified pipeline
"""

from typing import Optional
from models.ner.inference import NERModel
from models.relation_extraction.inference import RelationExtractor
from models.relation_extraction.graph_builder import GraphBuilder


class RelationGraphBuilder:
    """
    Complete pipeline for building relation graphs from Vietnamese text
    
    Process:
    1. Extract entities using NER model
    2. Extract relations between entities using Relation Extraction model
    3. Build knowledge graph with nodes (entities) and edges (relations)
    """
    
    def __init__(self):
        """Initialize relation graph builder"""
        print("Initializing Relation Graph Builder...")
        try:
            self.ner_model = NERModel()
            self.relation_extractor = RelationExtractor()
            self.graph_builder = GraphBuilder()
            print("✅ Relation Graph Builder initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning during initialization: {e}")
            self.ner_model = None
            self.relation_extractor = None
            self.graph_builder = None
    
    def build(self, text: str, verbose: bool = False) -> dict:
        """
        Build relation graph from Vietnamese text
        
        Pipeline:
        1. Extract entities using NER
        2. Extract relations between entities
        3. Build graph with nodes (entities) and edges (relations)
        
        Args:
            text (str): Input Vietnamese text to analyze
            verbose (bool): Print debug information. Default: False
        
        Returns:
            dict: Knowledge graph structure:
                {
                    "nodes": [
                        {
                            "id": "0",
                            "label": "Nhân vật A",
                            "type": "PERSON"
                        },
                        {
                            "id": "1",
                            "label": "Nhân vật B",
                            "type": "PERSON"
                        }
                    ],
                    "edges": [
                        {
                            "source": "0",
                            "target": "1",
                            "label": "DEFEATS",
                            "confidence": 0.95
                        }
                    ],
                    "node_count": 2,
                    "edge_count": 1
                }
        
        Raises:
            None (handles all errors internally, returns empty graph on error)
        
        Example:
            >>> builder = RelationGraphBuilder()
            >>> graph = builder.build("Nhân vật A đánh bại nhân vật B")
            >>> print(graph["node_count"])  # 2
            >>> print(graph["edge_count"])  # 1
        """
        
        # Input validation
        if not text or not isinstance(text, str):
            if verbose:
                print("[Relation Graph] Invalid input: text must be non-empty string")
            return self._empty_graph()
        
        text = text.strip()
        if not text:
            if verbose:
                print("[Relation Graph] Empty text after stripping")
            return self._empty_graph()
        
        try:
            # Step 1: Extract entities using NER
            if verbose:
                print(f"[Relation Graph] Step 1: Extracting entities from text...")
                print(f"[Relation Graph] Text length: {len(text)} characters")
            
            if self.ner_model is None:
                if verbose:
                    print("[Relation Graph] NER model not initialized, creating new instance...")
                self.ner_model = NERModel()
            
            entities = self.ner_model.extract_entities(text)
            
            if verbose:
                print(f"[Relation Graph] ✓ Found {len(entities)} entities")
                if entities:
                    for i, entity in enumerate(entities, 1):
                        print(f"  [{i}] {entity['text']} ({entity['type']})")
            
            if not entities:
                if verbose:
                    print("[Relation Graph] No entities found, returning empty graph")
                return self._empty_graph()
            
            # Step 2: Extract relations between entities
            if verbose:
                print(f"[Relation Graph] Step 2: Extracting relations between entities...")
            
            if self.relation_extractor is None:
                if verbose:
                    print("[Relation Graph] Relation Extractor not initialized, creating new instance...")
                self.relation_extractor = RelationExtractor()
            
            relations = self.relation_extractor.extract_relations(text, entities)
            
            if verbose:
                print(f"[Relation Graph] ✓ Found {len(relations)} relations")
                if relations:
                    for i, relation in enumerate(relations, 1):
                        print(f"  [{i}] {relation['source']} --({relation['relation_type']})-> {relation['target']}")
                        print(f"      confidence: {relation['confidence']:.2f}")
            
            # Step 3: Build graph from entities and relations
            if verbose:
                print(f"[Relation Graph] Step 3: Building knowledge graph...")
            
            if self.graph_builder is None:
                if verbose:
                    print("[Relation Graph] Graph Builder not initialized, creating new instance...")
                self.graph_builder = GraphBuilder()
            
            graph = self.graph_builder.build_graph(entities, relations)
            
            if verbose:
                print(f"[Relation Graph] ✓ Graph built successfully")
                print(f"[Relation Graph] Final graph: {graph['node_count']} nodes, {graph['edge_count']} edges")
            
            return graph
        
        except Exception as e:
            print(f"[Relation Graph] ❌ Error building relation graph: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_graph()
    
    @staticmethod
    def _empty_graph() -> dict:
        """Return empty graph structure"""
        return {
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0
        }


# Global instance for single-function usage
_builder_instance: Optional[RelationGraphBuilder] = None


def build_relation_graph(text: str, verbose: bool = False) -> dict:
    """
    Build relation graph from Vietnamese text (convenience function)
    
    Uses global builder instance for efficiency
    
    Args:
        text (str): Input Vietnamese text to analyze
        verbose (bool): Print debug information. Default: False
    
    Returns:
        dict: Knowledge graph with nodes (entities) and edges (relations)
            {
                "nodes": [...],
                "edges": [...],
                "node_count": N,
                "edge_count": M
            }
    
    Example:
        >>> graph = build_relation_graph("Nhân vật A đánh bại nhân vật B")
        >>> print(f"Graph: {graph['node_count']} nodes, {graph['edge_count']} edges")
    """
    
    global _builder_instance
    
    # Initialize global instance if needed
    if _builder_instance is None:
        _builder_instance = RelationGraphBuilder()
    
    # Build graph
    return _builder_instance.build(text, verbose=verbose)


def build_relation_graph_verbose(text: str) -> dict:
    """
    Build relation graph with verbose output
    
    Convenience function with verbose=True
    
    Args:
        text (str): Input Vietnamese text
    
    Returns:
        dict: Knowledge graph
    """
    return build_relation_graph(text, verbose=True)


if __name__ == "__main__":
    """
    Test relation graph building with sample texts
    """
    
    print("="*80)
    print("RELATION GRAPH BUILDER - TEST SUITE")
    print("="*80)
    
    # Initialize builder
    builder = RelationGraphBuilder()
    
    # Test cases
    test_cases = [
        {
            "name": "Simple DEFEATS relation",
            "text": "Nhân vật A đánh bại nhân vật B"
        },
        {
            "name": "Multiple relations",
            "text": "Nhân vật A đánh bại nhân vật B và lấy được bảo vật cổ"
        },
        {
            "name": "HELPS relation",
            "text": "Nhân vật A giúp đỡ nhân vật B"
        },
        {
            "name": "Complex story",
            "text": "Nhân vật A là một chiến binh mạnh mẽ. "
                   "Anh ta gặp công chúa B tại Hà Nội. "
                   "Cùng nhau họ đánh bại kẻ thù và bảo vệ làng."
        },
        {
            "name": "Text with no entities",
            "text": "Đây là một câu văn bình thường"
        }
    ]
    
    # Run tests
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'='*80}")
        print(f"Input: {test['text']}\n")
        
        graph = builder.build(test['text'], verbose=True)
        
        print(f"\nResult Summary:")
        print(f"  - Nodes: {graph['node_count']}")
        print(f"  - Edges: {graph['edge_count']}")
        
        if graph['node_count'] > 0:
            print(f"\nNodes:")
            for node in graph['nodes']:
                print(f"  [{node['id']}] {node['label']} ({node['type']})")
        
        if graph['edge_count'] > 0:
            print(f"\nEdges:")
            for edge in graph['edges']:
                print(f"  {edge['source']} -> {edge['target']}: {edge['label']} (confidence: {edge['confidence']:.2f})")
        
        print()
    
    print("="*80)
    print("✅ All tests completed!")
    print("="*80)