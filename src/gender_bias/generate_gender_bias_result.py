import pickle as pickle
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def posneg(list):
    pos = [x for x in list if x > 0]
    neg = [x for x in list if x < 0]

    return np.mean(pos), np.mean(neg), len(pos), len(neg)


def rename(name):
    specs = [
        ['pro', 'remove Pronouns'],
        ['weat', 'remove WEAT'],
        ['all', 'remove All'],
        ['mix_pro', 'mix Pronouns'],
        ['mix_weat', 'mix WEAT'],
        ['mix_all', 'mix All'],
        ['original_Rall', 'All'],
        ['original_Rweat', 'WEAT'],
        ['original_Rpro', 'Pronouns'],
    ]
    for spec in specs:
        if spec[0] in name:
            return spec[0]


def calc_bias_dict(df_dict):
    bias_dict = {}

    for spec in df_dict.keys():
        bias_l = df_dict[spec].bias.tolist()
        # total bias
        overall_bias_total = np.mean(bias_l)
        overall_bias_total_noZero = np.mean([i for i in bias_l if i != 0])
        # absolute bias
        overall_bias_abs = np.mean([abs(x) for x in bias_l])
        overall_bias_abs_noZero = np.mean([abs(x) for x in bias_l if x != 0])
        # pos neg bias
        pos, neg, pos_n, neg_n = posneg(bias_l)

        bias_dict[spec] = [
            overall_bias_total,  # 0
            overall_bias_abs,  # 1
            pos, neg,  # 2 3
            pos_n, neg_n,  # 4 5
            overall_bias_total_noZero,  # 6
            overall_bias_abs_noZero]  # 7
    return bias_dict


def get_bias(task='IMDB', model_id_='tinybert'):
    files = glob("results/*")

    df_dict = {}
    for file in files:
        if '_{}_'.format(model_id_) in file and task in file:
            with open(file, "rb") as fh:
                data = pickle.load(fh)
            df_dict[rename(file)] = data
    return df_dict


specs = ['original_Rall', 'original_Rweat', 'original_Rpro', "N_pro", "N_weat", "N_all", "mix_pro", "mix_weat",
         "mix_all"]  # , ]
model_ids = ["albertbase", "albertlarge", "bertbase", "bertlarge", "distbase", "robertabase", "robertalarge"]
specs = [
    'pro',
    'weat',
    'all'
]
model_ids = ["tinybert"]


def get_bias_bydict(dic, spec):
    df = dic[spec]
    df_no_zero = df.loc[(df != 0).all(axis=1)]

    bias = df.bias.mean()
    zero_bias = df_no_zero.bias.mean()

    neg_count = 0
    pos_count = 0
    zero_count = 0
    for elem in df.bias.tolist():
        if elem > 0:
            pos_count += 1
        elif elem < 0:
            neg_count += 1
        elif elem == 0:
            zero_count += 1

    return bias, zero_bias, neg_count, pos_count, zero_count


for model in model_ids:
    dic = get_bias('IMDB')
    for spec in specs:
        b, zb, neg, pos, zero = get_bias_bydict(dic, spec)
        print(model, "& ", spec, "& ", b, "& ", zb, "& ", neg, "& ", pos, "& ", zero, "\\""\\ ")

IMDB_training_details = [
    ["IMDB", "albertbase", "N_all", 0.1, 5],
    ["IMDB", "albertbase", "N_pro", 0.05, 4],
    ["IMDB", "albertbase", "N_weat", 0.05, 8],
    ["IMDB", "albertbase", "mix_all", 0.1, 19],
    ["IMDB", "albertbase", "mix_pro", 0.1, 13],
    ["IMDB", "albertbase", "mix_weat", 0.2, 6],
    ["IMDB", "albertbase", "original", 0.1, 8],
    ["IMDB", "albertlarge", "N_all", 0.05, 17],
    ["IMDB", "albertlarge", "N_pro", 0.05, 11],
    ["IMDB", "albertlarge", "N_weat", 0.05, 12],
    ["IMDB", "albertlarge", "mix_all", 0.2, 18],
    ["IMDB", "albertlarge", "mix_pro", 0.1, 7],
    ["IMDB", "albertlarge", "mix_weat", 0.1, 19],
    ["IMDB", "albertlarge", "original", 0.2, 12],
    ["IMDB", "bertbase", "N_all", 0.1, 12],
    ["IMDB", "bertbase", "N_pro", 0.1, 11],
    ["IMDB", "bertbase", "N_weat", 0.1, 12],
    ["IMDB", "bertbase", "mix_all", 0.2, 12],
    ["IMDB", "bertbase", "mix_pro", 0.2, 5],
    ["IMDB", "bertbase", "mix_weat", 0.2, 10],
    ["IMDB", "bertbase", "original", 0.1, 10],
    ["IMDB", "bertlarge", "N_all", 0.05, 19],
    ["IMDB", "bertlarge", "N_pro", 0.05, 7],
    ["IMDB", "bertlarge", "N_weat", 0.1, 6],
    ["IMDB", "bertlarge", "mix_all", 0.2, 14],
    ["IMDB", "bertlarge", "mix_pro", 0.2, 19],
    ["IMDB", "bertlarge", "mix_weat", 0.2, 13],
    ["IMDB", "bertlarge", "original", 0.05, 13],
    ["IMDB", "distbase", "N_all", 0.05, 16],
    ["IMDB", "distbase", "N_pro", 0.05, 18],
    ["IMDB", "distbase", "N_weat", 0.05, 19],
    ["IMDB", "distbase", "mix_all", 0.2, 14],
    ["IMDB", "distbase", "mix_pro", 0.2, 18],
    ["IMDB", "distbase", "mix_weat", 0.2, 19],
    ["IMDB", "distbase", "original", 0.05, 8],
    ["IMDB", "robertabase", "N_all", 0.05, 10],
    ["IMDB", "robertabase", "N_pro", 0.05, 7],
    ["IMDB", "robertabase", "N_weat", 0.05, 10],
    ["IMDB", "robertabase", "mix_all", 0.05, 11],
    ["IMDB", "robertabase", "mix_pro", 0.05, 4],
    ["IMDB", "robertabase", "mix_weat", 0.05, 7],
    ["IMDB", "robertabase", "original", 0.05, 8],
    ["IMDB", "robertalarge", "N_all", 0.05, 15],
    ["IMDB", "robertalarge", "N_pro", 0.05, 10],
    ["IMDB", "robertalarge", "N_weat", 0.05, 14],
    ["IMDB", "robertalarge", "mix_all", 0.05, 5],
    ["IMDB", "robertalarge", "mix_pro", 0.05, 5],
    ["IMDB", "robertalarge", "mix_weat", 0.05, 5],
    ["IMDB", "robertalarge", "original", 0.05, 11]
]


def r(n, d=4):
    return round(n, d)


for model in model_ids:
    dic = get_bias('IMDB', model_id_=model)
    for spec in specs:
        b, zb, neg, pos, zero = get_bias_bydict(dic, spec)
        print(model, "& ", spec, "& $", round(b, 4), "$ & $", round(zb, 4), "$ & ", neg, "& ", pos, "& ", zero,
              "\\""\\ ")

for model in model_ids:
    dic = get_bias('IMDB', model_id_=model)
    bias_dict = calc_bias_dict(dic)
    for spec in specs:
        overall_bias_total, overall_bias_abs, pos, neg, pos_n, neg_n, overall_bias_total_noZero, overall_bias_abs_noZero = \
            bias_dict[spec]
        b, zb, neg, pos, zero = get_bias_bydict(dic, spec)
        print(model, "& ", spec, "& ",
              r(overall_bias_abs_noZero), "& ", r(zb), "& ",
              r(overall_bias_abs), "& ", r(b), "& ",
              neg, "& ", pos, "& ", zero, "\\""\\ ")

special_spec = ['original_Rpro', "N_pro", "mix_pro",
                'original_Rweat', "N_weat", "mix_weat",
                'original_Rall', "N_all", "mix_all"]

res_dic = {}
res_dic['ss'] = specs
for m in model_ids:
    res_dic[m + '_abs'] = []
    res_dic[m + '_tot'] = []

for model in model_ids:
    dic = get_bias('IMDB', model_id_=model)
    bias_dict = calc_bias_dict(dic)

    for spec in specs:
        overall_bias_total, overall_bias_abs, pos, neg, pos_n, neg_n, overall_bias_total_noZero, overall_bias_abs_noZero = \
            bias_dict[spec]
        res_dic[model + '_abs'].append(r(overall_bias_abs_noZero))
        res_dic[model + '_tot'].append(r(overall_bias_total_noZero))

res = pd.DataFrame(res_dic)
res = res.transpose()

special_spec = ['original_Rpro', "N_pro", "mix_pro",
                'original_Rweat', "N_weat", "mix_weat",
                'original_Rall', "N_all", "mix_all"]

res_dic = {'mods': model_ids}
for spec in specs:
    res_dic[spec + '_abs'] = []
    res_dic[spec + '_tot'] = []

for model in model_ids:
    dic = get_bias('IMDB', model_id_=model)
    bias_dict = calc_bias_dict(dic)

    for spec in specs:
        overall_bias_total, overall_bias_abs, pos, neg, pos_n, neg_n, overall_bias_total_noZero, overall_bias_abs_noZero = \
            bias_dict[spec]
        res_dic[spec + '_abs'].append(r(overall_bias_abs_noZero))
        res_dic[spec + '_tot'].append(r(overall_bias_total_noZero))

res = pd.DataFrame(res_dic)
res = res.transpose()
print(res)

print(res[0].to_latex())


def stats(x, y, a="two-sided"):
    rs = wilcoxon(x, y, alternative=a)
    s = 'X'
    if rs[1] < 0.001:
        s = '***'
    elif rs[1] < 0.01:
        s = '**'
    elif rs[1] < 0.05:
        s = '*'
    return rs, s


for model in model_ids:
    dic = get_bias('IMDB', model_id_=model)
    for spec in specs:
        m = dic[spec]['pos_prob_m'].tolist()
        f = dic[spec]['pos_prob_f'].tolist()
        rs, s = stats(m, f)
        print(model, spec, '####', rs, s)
        print('####m-f<0')
        print(model, spec, '####', rs, s)
        print('#####')
