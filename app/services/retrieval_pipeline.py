from app.services.logger import get_logger

logger = get_logger(__name__)

class RetrievalPipeline:

    def __init__(
        self,
        retriever,
        multiquery_transform=None,
        hyde_transform=None,
        fusion=None,
        reranker=None,
    ):
        self.retriever = retriever
        self.multiquery_transform = multiquery_transform
        self.hyde_transform = hyde_transform
        self.fusion = fusion
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int):
        logger.info(f"Starting retrieval: {query}")

        # -------------------------------------
        # 1. MULTIQUERY
        # -------------------------------------
        queries = [query]
        if self.multiquery_transform:
            queries = self.multiquery_transform.transform(query)

        logger.info(f"Queries generated: {queries}")

        # -------------------------------------
        # 2. RETRIEVAL LOOP
        # -------------------------------------
        all_results = []
        for q in queries[:1]:
            retrieval_query = q

            # HYDE (Hypothetical Document Embeddings)
            if self.hyde_transform:
                retrieval_query = self.hyde_transform.transform(q)

            nodes = self.retriever.retrieve(
                retrieval_query,
                top_k,
            )
            all_results.append(nodes)

        if not all_results:
            return []

        # -------------------------------------
        # 3. DEDUPLICATION (Finding Unique Results)
        # -------------------------------------
        # We use a dictionary to keep only the highest scoring instance of a node
        unique_nodes_dict = {}
        
        for node_list in all_results:
            for node_with_score in node_list:
                node_id = node_with_score.node.node_id
                
                # If the node is new OR this current score is higher than the one we stored
                if node_id not in unique_nodes_dict or node_with_score.score > unique_nodes_dict[node_id].score:
                    unique_nodes_dict[node_id] = node_with_score

        # Convert back to a list
        unique_results = list(unique_nodes_dict.values())
        logger.info(f"Deduplicated {sum(len(l) for l in all_results)} total nodes to {len(unique_results)} unique nodes.")

        # -------------------------------------
        # 4. FUSION
        # -------------------------------------
        # If a fusion strategy (like RRF) is provided, use it on raw lists.
        # Otherwise, use our deduplicated list sorted by score.
        if self.fusion:
            merged_nodes = self.fusion.fuse(all_results)
        else:
            merged_nodes = sorted(unique_results, key=lambda x: x.score, reverse=True)[:top_k]

        # -------------------------------------
        # 5. RERANK
        # -------------------------------------
        # if self.reranker and merged_nodes:
        #     merged_nodes = self.reranker.rerank(
        #         query,
        #         merged_nodes,
        #     )

        logger.info("Retrieval pipeline completed => ")
        return merged_nodes

