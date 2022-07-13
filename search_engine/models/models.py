from pydantic import BaseModel
from typing import List

class Document(BaseModel):
    documentName:str
    documentDir:str
    documentTopic:str

class QueryResult(BaseModel):
    documents: List[Document]
    """
    Query response in milliseconds
    """
    responseTime: int = 0
    # precision: float = 0
    # recall: float = 0
    # f1: float = 0
    topic: str = ""
    query:str
    queryExpansions: List[str]

class ResponseModel(BaseModel):
    documents: List[Document]
    """
    Query response in milliseconds
    """
    responseTime: int = 0



class FeedbackModel(BaseModel):
    query:str
    relevants: List[str]
    not_relevants: List[str]
