from term_lists import *

def add_space(word):
    return ' ' + word + ' '


def count_terms(text, terms=all_terms):
    res = dict.fromkeys(terms, 0)
    for elem in terms:
        res[elem] = text.count(add_space(elem))
    return res


def mask_by_dict(review, terms):
    """
    mask_byDict: Mask terms in a text
    args
        review (str): Text
        terms (dict): tems. Mask kes by value.
    return tuple [(str) new masked review text, (dict) term occurances]
    """
    for word, initial in terms.items():
        review = review.replace(add_space(word), add_space(initial))
    return review


def make_male(review):
    return mask_by_dict(review, terms_f2m)


def make_female(review):
    return mask_by_dict(review, terms_m2f)


def make_neutral(text, terms=all_terms):
    for elem in terms:
        text = text.replace(add_space(elem), ' ')
    return text


def make_all(l, fun):
    reviews = []
    freqs = []
    for elem in l:
        rev, freq = fun(elem)
        reviews.append(rev)
        freqs.append(freq)
    return reviews, freqs


def make_all_df(df):
    """
    +++ Description +++
    args
        df with columns 'text' (str) and 'label' (int)
    """
    texts = df.text.tolist()
    df["count_table"] = [count_terms(e) for e in texts]
    df["count_total"] = [sum(e.values()) for e in df["count_table"].tolist()]
    df["count_table_weat"] = [count_terms(e, all_weat) for e in texts]
    df["count_weat"] = [sum(e.values()) for e in df["count_table_weat"].tolist()]
    df["count_prons"] = [sum([e[pronoun] for pronoun in all_prons]) for e in df["count_table"].tolist()]
    df["len"] = [len(e.split()) for e in texts]
    print(' make_all_df: finish counts and length')
    # ---
    df["text_all_M"] = [mask_by_dict(e, terms_f2m) for e in texts]
    print(' make_all_df: finish text_all_M')
    df["text_all_F"] = [mask_by_dict(e, terms_m2f) for e in texts]
    print(' make_all_df: finish text_all_F')
    df["text_all_N"] = [make_neutral(e, all_terms) for e in texts]
    print(' make_all_df: finish text_all_N')
    # ---
    df["text_weat_M"] = [mask_by_dict(e, weat_f2m) for e in texts]
    print(' make_all_df: finish text_weat_M')
    df["text_weat_F"] = [mask_by_dict(e, weat_m2f) for e in texts]
    print(' make_all_df: finish text_weat_F')
    df["text_weat_N"] = [make_neutral(e, all_weat) for e in texts]
    print(' make_all_df: finish text_weat_N')
    # ---
    df["text_pro_M"] = [mask_by_dict(e, prons_f2m) for e in texts]
    print(' make_all_df: finish text_pro_M')
    df["text_pro_F"] = [mask_by_dict(e, prons_m2f) for e in texts]
    print(' make_all_df: finish text_pro_F')
    df["text_pro_N"] = [make_neutral(e, all_prons) for e in texts]
    print(' make_all_df: finish text_pro_N')


def check_df(foo):
    c1 = all(foo[foo['count_total'] >= foo['count_weat']])
    c2 = all(foo[foo['count_total'] >= foo['count_prons']])
    # there is "best man" and "best men" in the lage dict. this is why it is ">=" instead of "==" ok that does not
    # work either, due to "paper boy" and the only cover terms. judt skip c3 c3 = all([x >= y for x,y in list(zip( [
    # len(s.split()) for s in foo['text_all_M'].tolist()], [len(s.split()) for s in foo['text_all_F'].tolist()] ))])
    c4 = all([x == y for x, y in list(zip([len(s.split()) for s in foo['text_weat_M'].tolist()],
                                          [len(s.split()) for s in foo['text_weat_F'].tolist()]))])
    c5 = all([x == y for x, y in list(zip([len(s.split()) for s in foo['text_pro_M'].tolist()],
                                          [len(s.split()) for s in foo['text_pro_F'].tolist()]))])
    c6 = all([x >= y for x, y in list(zip(foo['len'].tolist(), [len(s.split()) for s in foo['text_all_N'].tolist()]))])
    c7 = all([x >= y for x, y in list(zip(foo['len'].tolist(), [len(s.split()) for s in foo['text_weat_N'].tolist()]))])
    c8 = all([x >= y for x, y in list(zip(foo['len'].tolist(), [len(s.split()) for s in foo['text_pro_N'].tolist()]))])
    if c1 and c2 and c4 and c5 and c6 and c7 and c8:
        print('all tests ok')
        print('tested dataframe - everything is fine')
    else:
        print('Error: something is wrong in your DataFrame')
        print(c1, c2, c4, c5, c6, c7, c8)
