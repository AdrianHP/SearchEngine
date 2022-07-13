from cmath import nan
import ir_datasets as ir
import sys
from pathlib import Path
import time
import pandas as pd

if __name__ == "__main__":
    sys.path.append(str((Path(__file__) / ".." / "..").resolve()))

from search_logic.ir_models.base import InformationRetrievalModel
from search_logic.ir_models.classification import ClassificationSVMModel
from search_logic.ir_models.vectorial import VectorialModel
from search_logic.ir_models.utils import read_document, read_relevance

BASE_PATH = (Path(__file__) / "..").resolve()

def get_qrels_dataframe(qrels_iterator, raw_tuple=False):
    """
    Converts the Cranfield Relations Dataset into a DataFrame
    """
    if raw_tuple:
        df = pd.DataFrame((qrel for qrel in qrels_iterator) ,columns=["query_id", "doc_id", "relevance"])
    else:
        df = pd.DataFrame(((qrel.query_id, qrel.doc_id, qrel.relevance) for qrel in qrels_iterator) ,columns=["query_id", "doc_id", "relevance"])
    return df

def get_pickled_stats(model, corpus, tag) -> pd.DataFrame:
    """
    Read pickled stats.df 
    """
    base_path = BASE_PATH / f"{corpus}_corpus"
    stats_path = (base_path / ".." / f"{tag}{model}_{corpus}_corpus.df").resolve()
    
    stats = pd.read_pickle(str(stats_path))
    return stats

def eval_model(model_name: str, corpus_name: str, model: InformationRetrievalModel, tag="", use_pickled_stats=False):
    """
    Simple test to Cranfield to see basic metrics.
    """
    base_path = BASE_PATH / f"{corpus_name}_corpus"
    relevance_threshold = 0
    stats_path = (base_path / ".." / f"{tag}{model_name}_{corpus_name}_corpus.df").resolve()
    
    if use_pickled_stats and stats_path.exists():
        stats = get_pickled_stats(model_name, corpus_name, tag)
        print_stats_info(stats)
        return

    start = time.time()
    model.build()
    print("Build Time:", time.time() - start, "seconds")

    if corpus_name in ["cranfield"]:
        dataset = ir.load(corpus_name)
        queries = { q.query_id: q.text for q in dataset.queries_iter() }
        qrels_df = get_qrels_dataframe(dataset.qrels_iter())
    elif corpus_name in ["med"]:
        queries = {q_id: text for q_id, text in read_document(BASE_PATH / f"{corpus_name}_raw" / f"{corpus_name.upper()}.QRY")}
        qrel_iterator = read_relevance(BASE_PATH / f"{corpus_name}_raw" / f"{corpus_name.upper()}.REL")
        qrels_df = get_qrels_dataframe(qrel_iterator, True)

    stats = {
        "query_id": [],
        "recall": [],
        "precision": [],
        "f1": [],
        "relevant_retrieved": [],
        "total_relevants": [],
        "rank_threshold": []
    }

    for query_id, qrel in qrels_df.groupby(["query_id"]):
        query = queries.get(query_id)
        if not query: # The querys in some relations are missing
            continue
        # Relevance
        # -1: Not relevant
        # 1: Minimun interest
        # 2: Useful references
        # 3: High degree of relevance
        # 4: Complete answer to the question

        relevant_docs = qrel[qrel["relevance"] >= relevance_threshold]
        # non_relevant_docs = qrel[qrel["relevance"] < relevance_threshold]

        query_rank = model.resolve_query(queries[query_id])
        query_rec_docs = [Path(doc['dir']).stem for _, doc in query_rank]
        for total_rank_to_check in [30, 50, 100]:
            rec_docs = query_rec_docs[:total_rank_to_check]

            rec_rel_docs = relevant_docs[relevant_docs["doc_id"].isin(rec_docs)]
            prec = len(rec_rel_docs)/len(rec_docs) if rec_docs else nan
            rec = len(rec_rel_docs)/len(relevant_docs) if not relevant_docs.empty else nan
            f1 = 2 * prec * rec/(prec + rec if prec + rec != 0 else 1)
            stats["query_id"].append(query_id)
            stats["recall"].append(rec)
            stats["precision"].append(prec)
            stats["relevant_retrieved"].append(len(rec_rel_docs))
            stats["total_relevants"].append(len(relevant_docs))
            stats["f1"].append(f1)
            stats["rank_threshold"].append(total_rank_to_check)

    stats = pd.DataFrame(stats)
    print("Test Time:", time.time() - start, "seconds")
    stats.to_pickle(str(stats_path))
    print_stats_info(stats)

def print_stats_info(stats: pd.DataFrame):
    """
    Given a DataFrame with metrics (recall, precision, f1) print the
    results grouped by `rank_threshold`
    """
    clean_nan_stats = stats.dropna(axis=0)

    print("Nans", len(stats) - len(clean_nan_stats))

    for rank_threshold, clean_stats in clean_nan_stats.groupby("rank_threshold"):
        print("Rank Threshold", rank_threshold)
        print(clean_stats.describe())
        print("Recall mean", clean_stats["recall"].mean())
        print("Recall max", clean_stats["recall"].max())
        print("Precision mean", clean_stats["precision"].mean())
        print("F1 mean", clean_stats["f1"].mean())
        print()


use_saved_df = True
for corpus_name in ["cranfield", "med"]:
    for feedback in [False, True]:
        tag = "with_feedback_seeded_" if feedback else "without_feedback_seeded_"
        print("VECTORIAL", corpus_name, "Feedback", feedback)
        model = VectorialModel(BASE_PATH / f"{corpus_name}_corpus", dataset_name=corpus_name, seed_feedback=feedback) 
        eval_model("vectorial", corpus_name, model, tag, use_saved_df)
        print()

        print("SVM", corpus_name, "Feedback", feedback)
        model = ClassificationSVMModel(BASE_PATH / f"{corpus_name}_corpus", dataset_name=corpus_name, seed_feedback=feedback)
        eval_model("svm", corpus_name, model, tag, use_saved_df)
        print()
