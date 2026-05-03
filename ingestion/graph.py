from langgraph.graph import StateGraph, END
from ingestion.state import IngestionState
from ingestion.nodes.loader import load_node
from ingestion.nodes.filter import filter_node
from ingestion.nodes.cleaner import clean_node
from ingestion.nodes.chunker import chunk_node
from ingestion.nodes.embedder import embed_node


def build_ingestion_graph() -> StateGraph:
    graph = StateGraph(IngestionState)
    graph.add_node("load", load_node)
    graph.add_node("filter", filter_node)
    graph.add_node("clean", clean_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("embed", embed_node)
    graph.set_entry_point("load")
    graph.add_edge("load", "filter")
    graph.add_edge("filter", "clean")
    graph.add_edge("clean", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", END)
    return graph.compile()
