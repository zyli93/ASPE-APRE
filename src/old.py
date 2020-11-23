
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