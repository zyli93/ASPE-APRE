'''
    The helper function for dep-parse mining.

    File name: dependency.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 12/04/2020
    Date Last Modified: TODO
    Python Version: 3.6
'''

import spacy
from spacy.symbols import amod, nsubj, acomp, attr
from spacy import displacy
BE_VERB = set(["was", "is", "were", "are", "\'re", "\'s"])
def extract_aspair_deptree_spacy(doc):
    """Extract Aspect-Sentiment pairs from [spaCy] style dependency graph.


    Args:
        doc - [spacy doc] the doc object containing the dependency structure
    
    Return:
        pairs - [(str, str)] <aspect, sentiment> pairs extracted from the dep
    
    """
    def merge_compound_token(tmp_tok):
        """return single compound string"""
        if tmp_tok.i > 0 and \
            tmp_tok.nbor(-1).dep_ == "compound" and \
            tmp_tok.nbor(-1).head == tmp_tok:
            return doc[tmp_tok.i - 1: tmp_tok.i + 1].text

        return tmp_tok.text

    def find_conj_token(tmp_tok):
        """Return conjunct token
        Assumption:
            1 - only one conj for each token
            2 - amod isn't a child of its
        """
        l_conj = list(tmp_tok.conjuncts)
        if not l_conj:
            return None
        
        conj_tok = l_conj[0]
        cchd = list(conj_tok.children)
        if amod in set([x.dep for x in cchd]):
            return None
        return conj_tok.text
        
    aspairs = []

    for tok in doc:
        chd = list(tok.children)  # gen -> list of token
        chd_dep = [ctok.dep for ctok in chd]  # list of deprel

        # add amod
        if tok.dep == amod:
            # head token (htok) -> aspect
            print("tok", tok)
            print("tok.dep", tok.dep_)
            htok = tok.head

            # first aspect token
            tok_asp = merge_compound_token(htok)
            print("htok and merge compound htok", htok, tok_asp)
            aspairs.append((tok_asp, tok.text))

            # mine tok_asp's conj in str
            conj_asp = find_conj_token(htok)

            # update aspect-sentiment pairs
            if conj_asp:
                aspairs.append([conj_asp, tok.text])
            
            conj_senti = find_conj_token(tok)
            if conj_senti:
                aspairs.append([htok.text, conj_senti])


        # process acomp
        elif tok.dep == acomp:
            htok = tok.head

            # find nsubj: they, I, something, [notfound]
            found_nsubj = False
            for sib in list(htok.children):
                if sib.dep == nsubj:
                    found_nsubj = True
                    subject = sib
                    break
            
            if found_nsubj:
                # I am, he is, xxx
                if subject.text.lower() in ["i", "we", "you", "he", "she"]:
                    break
                # "They are ..."
                elif subject.text.lower() in ["it", "they"]:
                    if htok.text.lower() in BE_VERB:
                        aspairs.append(("ItemTok", tok.text))
                        # they are young and beautiful
                        cj_tok = find_conj_token(tok)
                        if cj_tok:
                            aspairs.append(("ItemTok", cj_tok))
                    else:
                        aspairs.append((htok.text, tok.text))
                        cj_tok = find_conj_token(tok)
                        if cj_tok:
                            aspairs.append((htok.text, cj_tok))
                # "apple tree is ..."
                else:
                    cmp_subject = merge_compound_token(subject)
                    aspairs.append((cmp_subject, tok.text))
                    
                    # tree and flower are ...
                    conj_subj = find_conj_token(subject)
                    if conj_subj:
                        aspairs.append((conj_subj, tok.text))

            # nsubj not found, e.g.: looks great!
            else:
                aspairs.append((htok.text, tok.text))

    # check return type
    assert all([isinstance(x[0], str) for x in aspairs]), \
        "not all aspect words are str" 
    assert all([isinstance(x[1], str) for x in aspairs]), \
        "not all sentiment words are str" 
    return aspairs


# Test cases:

if __name__ == "__main__":
    # run the following for testing
    tcase = [
        "Good car!",
        "Good and nice car!",
        "Babama chair is fancy!",
        "The wood chair is fancy!",
        "GloVe is a pre-trained embedding vector popularly used for a wide range of NLP tasks."
    ]
    exp = tcase[-1]
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(exp)
    print(extract_aspair_deptree_spacy(doc))