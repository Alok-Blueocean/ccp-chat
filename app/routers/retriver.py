# from fastapi import APIRouter

# from app.models.schemas import (
#     QueryRequest,
#     RetrieverResponse,
# )

# from app.services.retriver_service_older import (
#     retrive,
# )

# router = APIRouter()


# @router.post(
#     "/search",
#     response_model=None,
# )
# def retriever_point(request: QueryRequest):

#     results = retrive(
#         query=request.query,
#         top_k=request.top_k,
#     )

#     print(results)

#     return None