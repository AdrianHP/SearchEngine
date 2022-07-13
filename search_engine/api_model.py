from typing import List, Set, Tuple
from models.models import Document, FeedbackModel, QueryResult, ResponseModel
from search_logic.ir_models.classification import ClassificationSVMModel
from search_logic.ir_models.utils import read_document, read_relevance
from search_logic.ir_models.vectorial import VectorialModel
# from nltk.corpus import wordnet
from pathlib import Path

import time as t
import ir_datasets as ir

###### CHANGE CONFIGURATIONS HERE
# corpus_name = "cranfield"
corpus_name = "med"

seed_feedback = True
# seed_feedback = False

# model_name = "Ranking SVM"
model_name = "Vectorial"

######

path = Path(__file__) / ".." / "test" / f"{corpus_name}_corpus"
CORPUS = path.resolve()

if model_name == "Vectorial":
    model = VectorialModel(CORPUS, dataset_name=corpus_name, seed_feedback=seed_feedback)
elif model_name == "Ranking SVM":
    model = ClassificationSVMModel(CORPUS, dataset_name=corpus_name,seed_feedback=seed_feedback)
else:
    raise Exception(f"Invalid model name {model_name}")

model.build()

print("Model built successfully")

def get_documents(query: str, offset:int, batch_size: int=15) -> QueryResult:
    s = t.time()
    values = model.resolve_query(query)[offset:offset+batch_size]
    e = t.time()

    return ResponseModel(
         documents = [Document(documentName=Path(doc["dir"]).name, documentDir=doc["dir"], documentTopic=doc["topic"]) for _,doc in values],
        responseTime=int((e - s) * 1000)
    )

def get_document_content(document_dir: str) -> str:
    doc = [doc["text"] for doc in model.build_result["documents"] if doc["dir"] == document_dir]
    if not doc:
        raise Exception(f"{document_dir} wasn't found")
    return doc[0]

def apply_feedback_to_model(feedback: FeedbackModel):
    relevant = [doc for doc in model.build_result["documents"] if doc["dir"] in feedback.relevants]
    not_relevant = [doc for doc in model.build_result["documents"] if doc["dir"] in feedback.not_relevants]
    model.add_relevant_and_not_relevant_documents(feedback.query, relevant, not_relevant)

def get_query_expansions(query: str) -> List[str]:
    return model.get_expansion_query(query)

def get_queries(dataset_name: str) -> List[str]:
    """
    Get querys from the given dataset
    """
    if dataset_name in ["cranfield"]:
        dataset = ir.load(dataset_name)
        # Queries
        return [q.text for q in dataset.queries_iter()]
    elif dataset_name in ["med"]:
        # Queries
        query_path = Path(__file__, "..", "test", f"{dataset_name}_raw", f"{dataset_name.upper()}.QRY").resolve()
        return [ text for q_id, text in read_document(query_path) ]

def get_relevants(query: str, dataset_name: str) -> Tuple[Set[str], Set[str]]:
    """
    Returns a tuple of (relevant docs ids, not relevant docs ids) for the given dataset
    """
    if dataset_name in ["cranfield"]:
        dataset = ir.load(dataset_name)
        # # Documents
        # doc_dic = { doc.doc_id: doc.text for doc in dataset.docs_iter() }

        # Queries
        queries_dict = { q.query_id for q in dataset.queries_iter() if query == q.text}

        relevant = set()
        not_relevant = set()

        # Relevance
        for qrel in dataset.qrels_iter():
            q_id, d_id, rel = qrel.query_id, qrel.doc_id, qrel.relevance
            if q_id in queries_dict:
                if rel > 0:
                    relevant.add(d_id)
                else:
                    not_relevant.add(d_id)    
        
        return relevant, not_relevant

    elif dataset_name in ["med"]:

        # Queries
        query_path = Path(__file__, "..", "test", f"{dataset_name}_raw", f"{dataset_name.upper()}.QRY").resolve()
        queries_dict = { q_id for q_id, text in read_document(query_path) if query == text}

        # Relevance
        relevance_path = Path(__file__, "..", "test", f"{dataset_name}_raw", f"{dataset_name.upper()}.REL").resolve()
        relevant = set()
        not_relevant = set()

        for q_id, d_id, rel in read_relevance(relevance_path):
            if q_id in queries_dict:
                if rel > 0:
                    relevant.add(d_id)
                else:
                    not_relevant.add(d_id)
        
        return relevant, not_relevant
