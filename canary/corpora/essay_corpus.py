import nltk


def find_paragraph_features(feats, component, _essay):
    # find the para the component is in
    paras = [k for k in _essay.text.split("\n") if k != ""]
    for para in paras:
        # found it
        if component.arg1.mention in para or component.arg2.mention in para:
            # find other relations in paragraph
            relations = []
            for r in _essay.relations:
                if r.arg1.mention in para or r.arg2.mention in para:
                    relations.append(r)
            feats["n_para_components"] = len(relations)

            # find preceding and following components
            i = relations.index(component)
            feats["n_following_components"] = len(relations[i + 1:])
            feats["n_preceding_components"] = len(relations[:i])

            # calculate ratio of components
            feats["n_attack_components"] = len([r for r in relations if r.type == "attacks"])
            feats["n_support_components"] = len([r for r in relations if r.type == "supports"])

            break


def find_cover_sentence_features(feats, _essay, rel):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer._params.abbrev_types.update(['i.e', "etc", "e.g"])

    sentences = tokenizer.tokenize(_essay.text)

    feats["arg1_covering_sentence"] = None
    feats["arg2_covering_sentence"] = None
    feats["arg1_preceding_tokens"] = 0
    feats["arg1_following_tokens"] = 0
    feats["arg2_preceding_tokens"] = 0
    feats["arg2_following_tokens"] = 0

    for sentence in sentences:
        if sentence[-1] == '.':
            sentence = sentence[:(len(sentence) - 1)]
        if rel.arg1.mention in sentence:
            feats["arg1_covering_sentence"] = sentence
            split = sentence.split(rel.arg1.mention)
            feats["arg1_preceding_tokens"] = len(nltk.word_tokenize(split[0]))
            feats["arg1_following_tokens"] = len(nltk.word_tokenize(split[1]))

        if rel.arg2.mention in sentence:
            feats["arg2_covering_sentence"] = sentence
            split = sentence.split(rel.arg2.mention)
            feats["arg2_preceding_tokens"] = len(nltk.word_tokenize(split[0]))
            feats["arg2_following_tokens"] = len(nltk.word_tokenize(split[1]))

    if feats['arg2_covering_sentence'] is None or feats['arg1_covering_sentence'] is None:
        raise ValueError("Failed in finding cover sentences for one or more relations")


def find_cover_sentence(_essay, rel):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer._params.abbrev_types.update(['i.e', "etc", "e.g"])

    sentences = tokenizer.tokenize(_essay.text)

    for sentence in sentences:
        if sentence[-1] == '.':
            sentence = sentence[:(len(sentence) - 1)]
        if rel.mention in sentence:
            return sentence

    raise ValueError("...")
