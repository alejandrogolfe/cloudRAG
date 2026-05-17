from langgraph.graph import StateGraph, END
from retrieval.state import RetrievalState
from retrieval.nodes.retriever import retrieve_node
from retrieval.nodes.reranker import rerank_node
from retrieval.nodes.augmenter import augment_node
from retrieval.nodes.generator import generate_node


def build_retrieval_graph() -> StateGraph:
    graph = StateGraph(RetrievalState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("augment", augment_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "augment")
    graph.add_edge("augment", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
