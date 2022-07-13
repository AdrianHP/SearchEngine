

import string
from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from .utils import save_object,get_object
from nltk import word_tokenize




def add_query_expansion_manager(context: dict) -> dict:
    """
    Add the feedback maanger used by the IR model
    """
    manager = QueryExpansionManager()
    context["query_expansion_manager"] = manager
    manager.build(context)

    return context

def add_query_expansions(context: dict) -> dict:
    """
    Adds the `query_expantions` of `query` to the context
    """

    query_expansion_manager: QueryExpansionManager = context.get('query_expansion_manager')
    query = context["query"]
    
    if query_expansion_manager is None:
        context["query_expansions"] = []
        return context
    
    expansions = query_expansion_manager.expand_query(context,query)

    context["query_expansions"] = expansions

    return context



class QueryExpansionManager:
    """
    Base class that manages the query expansion
    """

    def generate_words_dict(self,context:dict):
        
        docs = context["documents"]
        language = context.get("language", "english")
        
        cache_key = [doc['dir'] for doc in docs]

        words_dict = get_object(cache_key,"words_dict")
        word_list = get_object(cache_key,"words_list")
        
        if words_dict is None:
            words = set()
       
            for doc in docs:
                text_split = [ i.lower() for i in  word_tokenize(doc["text"], language) if  (len(i)>1 or (list(i)[0]) not in string.punctuation)]
                words = words.union(set(text_split))

            words_dict = {}
            for i,word in enumerate(words):
                words_dict[word] = i

            word_list = list(words)
            save_object(cache_key,words_dict, "words_dict")
            save_object(cache_key,word_list, "words_list")

        context["words_dict"] = words_dict
        context["words_list"] = word_list
            
        return context
    
    def generate_sparse_matrix(self,context:dict):
        
        docs = context["documents"]
        language = context.get("language", "english")

        cache_key = [doc['dir'] for doc in docs]
        sparse_matrix = get_object(cache_key,"sparse_matrix")

        if sparse_matrix is None:
            words_dict = context["words_dict"]
            sparse_matrix = csr_matrix((len(words_dict), len(words_dict)), dtype = np.int8)
            count = 0
            for doc in docs:
                text_split = [ i.lower() for i in  word_tokenize(doc["text"], language) if  (len(i)>1 or (list(i)[0]) not in string.punctuation)]
             
                for i in range(len(text_split)-1):
                    sparse_matrix[words_dict[text_split[i]], words_dict[text_split[i+1]]] += 1
                count +=1 
               
            save_object(cache_key,sparse_matrix, "sparse_matrix")
        
        context["sparse_matrix"] = sparse_matrix
        return context

    def get_expand_query(self,context:dict,word:str):
        dict = context["words_dict"]
        words_list = context["words_list"]
        sparse_m = context["sparse_matrix"]
        if word in dict:
            word_index = dict[word]
            word_row = sparse_m[word_index]
            non_zero_values = word_row.nonzero()[1]
            rank_words = []
            for val in non_zero_values:
                rank_words.append([word_row[0,val],val])

            rank_words.sort(reverse=True)    

            result = [words_list[i[1]] for i in rank_words]

            return result
        else: 
            return []
          
    def build(self, context: dict):
        """
        Initialize the manager
        """
        self.generate_words_dict(context)
        self.generate_sparse_matrix(context)
        return

    def expand_query(self,context:dict, query: dict) -> List[str]:
        """
        Returns a rank for the query expansion for the given query
        """
        if query["text"] == "":
            return[]
        
        words = word_tokenize(query["text"], context.get("language", "english"))[-1]
        word = words.lower()
        rank = self.get_expand_query(context,word)
        list_words = context["words_list"]
       
        if len(rank) == 0:
            rank = [i for i in list_words if i.startswith(query["text"])]
            i = 3
            return [""+ x for x in rank[:5]]
        
        
        return [query['text'] + " " + x for x in rank[:5]] # TODO
