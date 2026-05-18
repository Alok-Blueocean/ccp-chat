from collections import defaultdict

from app.services.fusion.base import BaseFusion
from app.services.logger import get_logger

logger = get_logger(__name__)


class RRFFusion(BaseFusion):

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, retrieval_results):

        logger.info("Running RRF fusion")

        scores = defaultdict(float)

        node_map = {}

        for node_list in retrieval_results:

            for rank, node in enumerate(node_list):

                node_id = node.node.node_id

                node_map[node_id] = node

                scores[node_id] += 1 / (
                    self.k + rank + 1
                )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        final_nodes = [
            node_map[node_id]
            for node_id, _ in ranked
        ]

        logger.info(
            f"RRF produced {len(final_nodes)} nodes"
        )

        return final_nodes