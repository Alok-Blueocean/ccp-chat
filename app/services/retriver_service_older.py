# from indexing_pipeline import CONFIG
# from litellm import completion
# import os, json
# from llama_index.vector_stores.qdrant import (
#     QdrantVectorStore
# )
# from llama_index.core.indices.query.query_transform import (
#         HyDEQueryTransform
#     )
# from llama_index.core.schema import QueryBundle
# from llama_index.llms.openai import OpenAI
# from llama_index.core import VectorStoreIndex
# from llama_index.core.schema import (
#     TextNode,
#     NodeWithScore,
# )

# from indexing_pipeline.index_qdrant import QdrantClintManager

# os.environ["OPENAI_API_KEY"] =CONFIG['models']['openai']

# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.core import Settings

# # Set the global embedding model for LlamaIndex
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Settings.llm = OpenAI(model="gpt-4o-mini")

# hyde = HyDEQueryTransform(
#         include_original=True
#     )
# qdrant = QdrantClintManager()

# vector_store = QdrantVectorStore(
#     client=qdrant.client,
#     collection_name=qdrant.CHILD_COLLECTION,
#     enable_hybrid=True,
#     # Tell LlamaIndex which named vectors to use
#     vector_name="text_dense",    # Must match your _create_chunk_collection
#     sparse_vector_name="text_sparse" 
# )

# index = VectorStoreIndex.from_vector_store(
#     vector_store
# )

# retriever = index.as_retriever(
#     similarity_top_k=5,
#     vector_store_query_mode="hybrid"
# )

# from llama_index.core.postprocessor import (
#     SentenceTransformerRerank
# )

# reranker = SentenceTransformerRerank(
#     model="cross-encoder/ms-marco-MiniLM-L-6-v2",
#     top_n=5
# )

# # response = completion(
# #   model="openai/gpt-4o",
# #   messages=[{"role": "user", "content": "Hello, how are you?"}],
# #   temperature=0.7
# # )
# # print(response.choices[0].message.content)
# multiquery_prompt = """You are an expert search query generator.

# Your task is to generate three diverse search queries that retrieve relevant documents for the user's question.

# Guidelines:
# - Preserve the original meaning and intent.
# - Generate semantically different phrasings.
# - Include technical synonyms when useful.
# - Include keyword-style search queries.
# - Keep queries concise and retrieval-optimized.
# - Do not answer the question.
# - Do not add explanations.
# - Avoid duplicates.
# - Don't repeat same keyword in other queries
# - Semantically similar but with different keywords
# - Minimize overlapping words between queries

# Return ONLY the queries as a JSON array of strings.

# """
# def retrive(query: str, top_k: int):
#     def create_multiquery(query: str):
#         response = completion(
#             model="openai/gpt-4o-mini",
#             # This is the key line for reliability
#             response_format={"type": "json_object"}, 
#             messages=[
#                 {"role": "system", "content": multiquery_prompt + " Output format: {\"queries\": [\"string\", \"string\"]}"},
#                 {"role": "user", "content": query}
#             ]
#         )
        
#         raw_content = response.choices[0].message.content
#         print("raw content")
#         print(raw_content)
#         # Parse the JSON object and extract the list
#         try:
#             data = json.loads(raw_content)
#             # Handle both cases: a direct list or a dictionary containing the list
#             if isinstance(data, list):
#                 return data
#             return data.get("queries", [])
#         except json.JSONDecodeError:
#             print("Failed to parse JSON")
#             return [query] # Fallback to original query
#     nodes = qdrant.search(query, limit=5)
#     print(nodes)
#     multiple_queries = create_multiquery(query)
#     print(multiple_queries)

#     all_nodes = []

#     for q in multiple_queries[:3]:
#         query_bundle = QueryBundle(q)
        
#         # 2. Transform the query using HyDE
#         # This uses Settings.llm (set above) to generate the fake doc
#         hyde_query_bundle = hyde.run(query_bundle)
        
#         # 3. Extract the hypothetical answer
#         # embedding_strs[0] is the LLM-generated fake document
#         hypothetical_doc = hyde_query_bundle.embedding_strs[0]
        
#         print(f"Original Query: {q}")
#         print(f"HyDE Doc: {hypothetical_doc[:100]}...")

#         # 4. Use your CUSTOM working search method
#         # This bypasses the LlamaIndex 400 Error entirely
#         nodes = qdrant.search(hypothetical_doc, limit=top_k)

#         all_nodes.append(nodes)
#     # print(all_nodes)
#     # Simple deduplication logic
#     unique_nodes = {}

#     for node_list in all_nodes:

#         for point in node_list:

#             if point.id not in unique_nodes:

#                 text = point.payload.get("text", "")

#                 metadata = point.payload

#                 text_node = TextNode(
#                     text=text,
#                     metadata=metadata,
#                 )

#                 node_with_score = NodeWithScore(
#                     node=text_node,
#                     score=point.score
#                 )

#                 unique_nodes[point.id] = node_with_score
#     first_five_nodes = list(unique_nodes.values())[:5]

# # 3. Print the content

#     print(f"Total Unique Nodes Found: {len(unique_nodes)}")
#     print("-" * 30)

#     for i, node_with_score in enumerate(first_five_nodes):

#         node = node_with_score.node

#         content = node.text

#         print(f"Node {i+1}")

#         print(f"Score: {node_with_score.score}")

#         print(f"Content Snippet: {content[:150]}...")

#         print("-" * 30)
#     print("++++++++++++++++++\n\n")
#     query_bundle = QueryBundle(query)
#     rerank_candidates = list(unique_nodes.values())

#     reranked_nodes = reranker.postprocess_nodes(
#         rerank_candidates,
#         query_bundle=query_bundle
#         )
#     print(reranked_nodes)
    


  