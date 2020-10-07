
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