import sys
import ir_datasets as ir
from pathlib import Path
import pandas as pd
import random

if __name__ == "__main__":
    sys.path.append(str((Path(__file__) / ".." / "..").resolve()))

from search_logic.ir_models.utils import read_document, read_relevance

def create_local_ir_dataset_corpus(corpus_name: str):
    """
    Create a local corpus. The names are the id of the files
    """

    base_path = Path(__file__) / ".."
    base_path: Path = base_path.resolve()

    dataset = ir.load(corpus_name)
    for doc in dataset.docs_iter():
        doc_id, text = doc.doc_id, doc.text
        if text:
            path = path / f"{corpus_name.replace('/', '_')}_corpus" / f"{doc_id}.txt"
            path.touch() # Creates the file if doesn't exist
            path.write_text(text)
        else:
            print(f"Document {doc_id} is empty")

# create_local_ir_dataset_corpus("cranfield")

def create_local_med_corpus():
    base_path = Path(__file__) / ".."
    base_path: Path = base_path.resolve()

    corpus_doc_path = base_path / "med_raw" / "MED.ALL"

    doc_ids = set()

    for doc_id, text in read_document(corpus_doc_path):
        text_path = base_path / "med_corpus" / f"{doc_id}.txt"
        text_path.write_text(text)
        doc_ids.add(doc_id)

    columns = ["query_id", "doc_id"] # ALL RELS ARE 1!!!!!!!
    relations = pd.DataFrame([(q_id, doc_id) for q_id, doc_id, rel in read_relevance(base_path/"med_raw"/"ORIG_MED.REL")], columns=columns)
    new_relations = {}

    for q_id, docs in relations.groupby("query_id"):
        complement = list(doc_ids.difference(docs["doc_id"]))
        random.shuffle(complement)
        new_relations[q_id] = complement[:len(docs)] # Adding potential non relevant relations. Balancing dataset
    
    relevance_doc = Path(base_path/"med_raw"/"ORIG_MED.REL")
    relevance_content = relevance_doc.read_text()
    relevance_doc = Path(base_path/"med_raw"/"MED.REL")
    relevance_content += "\n".join(f"{q_id} 0 {doc_id} 0" for q_id in sorted(new_relations) for doc_id in new_relations[q_id])
    relevance_doc.write_text(relevance_content)

create_local_med_corpus()