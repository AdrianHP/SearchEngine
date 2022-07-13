from .utils import cosine_sim
from .feedback import add_feedback_manager, add_feedback_to_query
from .query_expansion import add_query_expansion_manager
from ..pipes.pipeline import Pipeline
from .base import *
from sklearn.feature_extraction.text import TfidfVectorizer

def smooth_query_vec(context: dict):
    """
    Smooth calculated query vector in `query` by some constant
    if any in `smooth_query_alpha`, defaults to 0.
    
    alpha*idf_i + (1-alpha)ntf_{iq} idf_i
    """
    query = context["query"]
    alpha = context.get("smooth_query_alpha", 0)
    idf = context.get("idf")
    matrix = context["term_matrix"]
    for i,term in enumerate(matrix.all_terms):
        query["vector"][i] = (alpha*(idf[term] if idf else 1) + (1 - alpha) *
                              query["vector"][i])### query["vector"] is already a TFIDF Vector result of TfidfVectorizer

    return context

def rank_documents(context: dict):
    """
    Ranks the `documents` with the `query` returning in the result in
    `ranked_documents` key. If `rank_threshold` is given then only values 
    higher will be returned
    """
    
    
    rank_threshold = context.get("rank_threshold", 0)
    
    query = context["query"]
    documents = context["documents"]
    ranking = []
    for doc in documents:
        s = cosine_sim(query["vector"], doc["vector"])
        if s > rank_threshold:
            ranking.append((s, doc))
    ranking.sort(key=lambda x: -x[0])
    
    context["ranked_documents"] = ranking
    
    return context

def add_vectorizer_vectorial(context: dict) -> dict:
    """
    Build and add a TF-IDF vectorizer to the context
    """
    return add_vectorizer(context, vectorizer_class=TfidfVectorizer, vectorizer_kwargs={"use_idf":True})

def add_idf(context: dict):
    """
    Build an idf dictionary based in the `vectorizer` and the `term_matrix`. 
    The vectorizer must have an idf_ porperty that holds the idf for the i
    term matching matrix.all_terms index
    """
    vectorizer = context["vectorizer"]
    matrix = context["term_matrix"]
    context["idf"] = {term: vectorizer.idf_[i] for i,term in enumerate(matrix.all_terms)}
    return context

class VectorialModel(InformationRetrievalModel):
    
    def __init__(self, corpus_address: str, smooth_query_alpha=0.0, language="english", rank_threshold=0.0,
                 alpha_rocchio=1, beta_rocchio=0.75, ro_rocchio=0.1, add_document_pipe=None, dataset_name="cranfield",
                 seed_feedback=False, **kwargs) -> None:

        query_to_vec_pipeline = Pipeline(
            apply_text_processing_query, 
            build_query_matrix, 
            add_vector_to_query,
        )
        build_pipeline = Pipeline(
            read_documents_from_hard_drive if not add_document_pipe else add_document_pipe, 
            add_training_documents,
            add_tokens,
            add_stopwords,
            add_lemmatizer, # Stemmer XOR Lemmatizer 
            # add_stemmer, # Stemmer XOR Lemmatizer
            add_wordnet,
            add_vectorizer_vectorial, 
            apply_text_processing, 
            build_matrix, 
            add_idf,
            add_vector_to_doc,
            add_feedback_manager,
            add_query_expansion_manager,
        )
        
        query_pipeline = Pipeline(
            add_feedback_to_query, 
            smooth_query_vec, 
            rank_documents,
        )
        query_context = {
            "smooth_query_alpha": smooth_query_alpha,
            "language": language,
            "rank_threshold": rank_threshold,
            "alpha_rocchio": alpha_rocchio,
            "beta_rocchio": beta_rocchio,
            "ro_rocchio": ro_rocchio,
            "vectorial": self,
        }
        build_context = {
            "language": language,
            "dataset_name": dataset_name,
            "vectorial": self,
            "seed_feedback": seed_feedback
        }
        super().__init__(corpus_address, query_pipeline, query_to_vec_pipeline, build_pipeline, query_context, build_context)
