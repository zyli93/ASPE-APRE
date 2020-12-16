
    # sp = spacy.load("en_core_web_sm")  # commented due to nltk
        # for sent in sent_tokenize(line):
        #     sp_sent = sp(sent)  # pos tagging with spacy, replaced by nltk
        #     pos_tagging = nltk.pos_tag(sent)
        #     for word in sp_sent:
        #         if word.text.lower() in vocab:
        #             vocab_pos[word.text]["pos"].update(word.pos_)
        #             vocab_pos[word.text]["tag"].update(word.tag_)

    # vocab_pos_coarse = {word: vocab_pos[word]["pos"].most_common(1)[0][0]
    #                     for word in vocab_pos.keys()}
    # vocab_pos_fine = {word: vocab_pos[word]["tag"].most_common(1)[0][0]
    #                   for word in vocab_pos.keys()}


    # for polarity in ["POS", "NEG"]:
    #     aspect_opinions[polarity] = {}
    #     for seed in seeds[polarity]:
    #         match_tokens = []
    #         for (l_token, r_token), pmi_value in pmi_matrix.items():
    #             # criteria: (1) r_token in vocab_postags
    #             #           (2) r_token has particular POS tag in "filter_pos"
    #             if l_token == seed and r_token in vocab_postags and \
    #                     vocab_postags[r_token] in filters_pos_keep:
    #                 match_tokens.append(
    #                     (r_token, vocab_postags[r_token], pmi_value))
    #         match_tokens.sort(reverse=True, key=lambda x: x[2])
    #         aspect_opinions[polarity][seed] = match_tokens[:quota]


    # def remove_invalid_words(args, word_list):
    # """Remove invalid words in the word list dataframe.
    # We used nltk.corpus.words API as it returns a valid English word dictionary.

    # Args:
    #     word_list - the dataframe of words
    # """
    # print("[Annotate] removing invalid words ...", end=" ")
    # assert isinstance(
    #     word_list, pd.DataFrame), "word_chart should be a dataframe"
    # valid_word_list = word_list[word_list["word"].isin(words.words())]
    # valid_word_list.to_csv(
    #     args.path + "/valid_cand_senti_pol.csv", index=False)
    # print("Done!")
    # print("[Annotate] valid vocab saved at {}/valid_cand_senti_pol.csv".format(args.path))
    # return valid_word_list



    # ====== deprecated code ==========
    # class AnnotatedReviewInID:
    # def __init__(self, review_text, aspairs, num_asp, max_pad_len=100):
    #     self.total_num_asp = num_asp
    #     self.pad_len = max_pad_len

    #     # =====================================================================
    #     # get a tokenizer
    #     #   tokenizer will process two things:
    #     #       1. Review text. Produces: input_ids, attn_mask, token_type_ids.
    #     #       2. The sentiment term inside the aspairs
    #     # =====================================================================
    #     tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    #     # =====================================================================
    #     # handle review in different formats.
    #     #   for str, use nltk.sent_tokenize to split it into list of sentences
    #     #   for list of sentences, leave it be
    #     # =====================================================================
    #     if isinstance(review_text, str):  # string of review
    #         self.review_sents = sent_tokenize(review_text)
    #     elif isinstance(review_text, list):  # list of sentences
    #         self.review_sents = review_text
    #     else:  # TypeError of compatible review_text
    #         raise TypeError(
    #             "review_text can only be [str] or [list] of sentences. " +
    #             "But a {} received!".format(str(type(review_text))))
        
    #     # return PyTorch-styled ("pt") padded tensors
    #     self.tokenized_revsents = tokenizer(
    #         self.review_sents, return_tensors="pt", padding=True)
    #     # get the shape of tokenized_review_sents
    #     tkn_revsents_input_ids = self.tokenized_revsents['input_ids']
    #     input_ids_shape = tuple(tkn_revsents_input_ids)
        
    #     # =====================================================================
    #     # handle aspairs (int, str). Here's the logic
    #     #   1. senti_id_to_asp is a defaultdict(list) where keys are 
    #     #      sentiment term tokenized ids and values are lists of corresponding
    #     #      aspects modified by that sentiment. 
    #     #   2. For each id, we use masking matrices to record their positions
    #     #      (coordinates) and add the masking matrices to the corresponding
    #     #      self.asp_senti_coord matrix. self.asp_senti_coord is a dictionary 
    #     #      of aspect_id to csr_matrices where the sparse matrices are 
    #     #      the masks.
    #     # =====================================================================

    #     senti_id_to_asp = defaultdict(list)

    #     # save aspect-sentiment term coordinate mask
    #     def new_csr_matrix():
    #         return csr_matrix(input_ids_shape, dtype=np.int32)
    #     self.asp_senti_coord = defaultdict(new_csr_matrix)
        
    #     # Step 1: build senti_id --> asp_id mapping
    #     for asp_id, senti_term in aspairs:
    #         senti_id = tokenizer.tokenize(senti_term, 
    #             add_special_tokens=False, return_attention_mask=False)
    #         senti_id = senti_id['input_ids'][0]
    #         senti_id_to_asp[senti_id].append(asp_id)
        
    #     # Step 2: aspect senti location
    #     for senti_id in senti_id_to_asp:
    #         senti_coord = csr_matrix(
    #             (tkn_revsents_input_ids == senti_id)
    #                 .numpy().astype(np.int32))
    #         inv_aspects = senti_id_to_asp[senti_id]
    #         for inv_asp in inv_aspects:
    #             self.asp_senti_coord[inv_asp] += senti_coord
        
    #     # after the processing, reset defaultdict to dict
    #     self.asp_senti_coord = dict(self.asp_senti_coord)

    #     # =====================================================================
    #     # attribute variables to keep:
    #     #     * tokenizer                - to del
    #     #     * tkn_revsents_input_ids   - to del
    #     #     * senti_id_to_asp          - to del
    #     #     * self.review_sents        - to keep
    #     #     * self.tokenized_revsents  - to keep
    #     #     * self.asp_senti_coord     - to keep
    #     # =====================================================================
    
    # def get_tokenized_revsents(self):
    #     return self.tokenized_revsents
    
    # def get_asp_coord(self):
    #     return self.asp_senti_coord
    
    # def get_review_text(self):
    #     return self.review_sents
    
    # def get_num_asp(self):
    #     return self.total_num_asp

    # =======deprecated func =======
    # def agg_tokenized_data(df):
    # """convert data into AnnotatedReviewInID objects
    # Args:
    #     df - the input dataframe
    # Return:
    #     converted user_anno_reviews, item_anno_reviews
    # """
    # def process(x):
    #     """func to apply"""
    #     annotated_review = AnnotatedReviewInID(
    #         review_text=x[COL_REV_TEXT],
    #         aspairs=x[COL_ASPAIRS],
    #         num_asp=args.num_aspects,
    #         max_pad_len=args.max_pad_length)
    #     return annotated_review

    # pandarallel.initialize(
    #     nb_workers=args.num_workers, progress_bar=True, verbose=1)
    # df[COL_ANNO_REV] = df.parallel_apply(process)

    # user_anno_reviews = df.groupby('user_id')[COL_ANNO_REV].agg(list)
    # item_anno_reviews = df.groupby('item_id')[COL_ANNO_REV].agg(list)

    # # convert user anno reviews to dictionary for easier store
    # user_anno_reviews = dict(user_anno_reviews)
    # item_anno_reviews = dict(item_anno_reviews)

    # return user_anno_reviews, item_anno_reviews