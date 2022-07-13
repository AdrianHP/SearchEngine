from typing import List
from models.models import FeedbackModel, QueryResult
from fastapi import FastAPI
from api_model import get_documents, get_document_content, apply_feedback_to_model, get_query_expansions

app = FastAPI()


@app.get("/query")
async def get_query_result(query:str, offset:int) -> QueryResult:
    """
    Returns the ranked documents associated with the `query` skipping `offset`
    """
    return get_documents(query,offset=offset)


@app.get("/document")
async def get_query_result(document_dir:str) -> str:
    """
    Returns document's content associated with the given `document_dir`
    """
    return get_document_content(document_dir)

@app.get("/expand")
async def get_query_result(query:str) -> List[str]:
    """
    Returns query's expansions
    """
    return get_query_expansions(query)
 

@app.post("/feedback")
async def apply_feedback(feedback: FeedbackModel):
    """
    Apply the feedback to the model
    """
    apply_feedback_to_model(feedback)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
