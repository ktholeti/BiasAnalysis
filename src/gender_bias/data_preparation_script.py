import os

import pandas as pd

import data_masking as masking
from train_functions import check_path


def download_data():
    # Download IMDb Data & read IMDB data
    os.system('git clone https://github.com/sciphie/IMDB-Movie-Reviews-50k.git')

    df_train_complete = pd.read_excel('IMDB-Movie-Reviews-50k/train_data_complete.xlsx', dtype=str)
    df_test_complete = pd.read_excel('IMDB-Movie-Reviews-50k/test_data_complete.xlsx', dtype=str)
    return df_train_complete, df_test_complete


def clean_data(df_train_complete, df_test_complete):
    # clean IMDb data
    df_train_complete.ID = df_train_complete['Set'] + '_' + df_train_complete['Sentiment'] + '_' + df_train_complete[
        'ID']
    df_test_complete.ID = df_test_complete['Set'] + '_' + df_test_complete['Sentiment'] + '_' + df_test_complete['ID']

    df_train = df_train_complete.drop(['Set', 'Rating', 'Review'], axis=1).rename(
        {
            'Review clean': 'text',
            'Sentiment': 'label'
        }, axis=1)

    df_test = df_test_complete.drop(['Set', 'Rating', 'Review'], axis=1).rename(
        {
            'Review clean': 'text',
            'Sentiment': 'label'
        }, axis=1)
    # Align naming of the general dataframes with the "spec" implementation
    df_train.to_pickle('IMDB_training/IMDB_original_train')
    df_test.to_pickle('IMDB_training/IMDB_original_test')

    return df_train, df_test


def get_gender_neutral_data_set(df_train, df_test):
    # IMDb - Step 1: Gender neutral data sets for training
    df_train_ = df_train.copy()
    df_test_ = df_test.copy()

    masking.make_all_df(df_train_)
    masking.make_all_df(df_test_)

    masking.check_df(df_test_)
    masking.check_df(df_train_)

    # Save whole table (large)
    df_train_.to_pickle("IMDB_l_train")
    df_test_.to_pickle("IMDB_l_test")
    return df_train_, df_test_


def save_dfs(df_train_, df_test_):
    # IMDb - Step 2: Save training and test dataframes for different training conditions.
    # neutral
    for spec in ['_all', '_pro', '_weat']:
        df_train_[['ID', 'text' + spec + '_N', 'label']].to_pickle('IMDB_training/IMDB_N' + spec + '_train')
        df_test_[['ID', 'text' + spec + '_N', 'label']].to_pickle('IMDB_training/IMDB_N' + spec + '_test')

    # mixed M+F
    for spec in ['_all', '_pro', '_weat']:
        m_tr = df_train_[['ID', 'text' + spec + '_M', 'label']].rename(columns={'text' + spec + '_M': 'text'})
        f_tr = df_train_[['ID', 'text' + spec + '_F', 'label']].rename(columns={'text' + spec + '_F': 'text'})
        tr = m_tr.append(f_tr)
        tr.to_pickle('IMDB_training/IMDB_mix' + spec + '_train')

        m_te = df_test_[['ID', 'text' + spec + '_M', 'label']].rename(columns={'text' + spec + '_M': 'text'})
        f_te = df_test_[['ID', 'text' + spec + '_F', 'label']].rename(columns={'text' + spec + '_F': 'text'})
        te = m_te.append(f_te)
        te.to_pickle('IMDB_training/IMDB_mix' + spec + '_test')

        # no term sample
        df_train_no_pron = df_train_[df_train_['count_total'] == 0][['ID', 'text', 'label']]
        assert (df_train_no_pron.shape == (4178, 3))
        df_test_no_pron = df_test_[df_test_['count_total'] == 0][['ID', 'text', 'label']]
        assert (df_test_no_pron.shape == (4222, 3))

        df_train_no_weat = df_train_[df_train_['count_weat'] == 0][['ID', 'text', 'label']]
        assert (df_train_no_weat.shape == (6488, 3))
        df_test_no_weat = df_test_[df_test_['count_weat'] == 0][['ID', 'text', 'label']]
        assert (df_test_no_weat.shape == (6668, 3))

        df_train_no_all = df_train_[df_train_['count_prons'] == 0][['ID', 'text', 'label']]
        assert (df_train_no_all.shape == (8322, 3))
        df_test_no_all = df_test_[df_test_['count_prons'] == 0][['ID', 'text', 'label']]
        assert (df_test_no_all.shape == (8655, 3))

        df_train_no_pron.to_pickle('IMDB_training/IMDB_no_pron_train')
        df_test_no_pron.to_pickle('IMDB_training/IMDB_no_pron_test')
        df_train_no_weat.to_pickle('IMDB_training/IMDB_no_weat_train')
        df_test_no_weat.to_pickle('IMDB_training/IMDB_no_weat_test')
        df_train_no_all.to_pickle('IMDB_training/IMDB_no_all_train')
        df_test_no_all.to_pickle('IMDB_training/IMDB_no_all_test')


def save_datasets(df_train_, df_test_):
    # Create Data Sets with no only samples that do contain a minimal number of term of the dict
    min_term_count = 1
    df_train__ = df_train_.rename(columns={'count_total': 'count_all', 'count_prons': 'count_pro'})
    df_test__ = df_test_.rename(columns={'count_total': 'count_all', 'count_prons': 'count_pro'})

    for spec in ['_all', '_pro', '_weat']:
        df_train_MIN = df_train__[df_train__['count' + spec] >= min_term_count]
        df_test_MIN = df_test__[df_test__['count' + spec] >= min_term_count]

        # all
        df_train_MIN[['ID', 'text', 'label']].to_pickle('IMDB_training/IMDB_MIN' + spec + '_test')
        df_test_MIN[['ID', 'text', 'label']].to_pickle('IMDB_training/IMDB_MIN' + spec + '_train')

        # neutral M+F
        df_train_MIN[['ID', 'text' + spec + '_N', 'label']].to_pickle('IMDB_training/IMDB_MIN_N' + spec + '_train')
        df_test_MIN[['ID', 'text' + spec + '_N', 'label']].to_pickle('IMDB_training/IMDB_MIN_N' + spec + '_test')

        # mixed
        m_tr = df_train_MIN[['ID', 'text' + spec + '_M', 'label']].rename(columns={'text' + spec + '_M': 'text'})
        f_tr = df_train_MIN[['ID', 'text' + spec + '_F', 'label']].rename(columns={'text' + spec + '_F': 'text'})
        tr = m_tr.append(f_tr)
        tr.to_pickle('IMDB_training/IMDB_MIN_mix' + spec + '_train')

        m_te = df_test_MIN[['ID', 'text' + spec + '_M', 'label']].rename(columns={'text' + spec + '_M': 'text'})
        f_te = df_test_MIN[['ID', 'text' + spec + '_F', 'label']].rename(columns={'text' + spec + '_F': 'text'})
        te = m_te.append(f_te)
        te.to_pickle('IMDB_training/IMDB_MIN_mix' + spec + '_test')

        assert (tr.shape[1] == 3)
        assert (te.shape[1] == 3)


def main():
    check_path('IMDB_training')
    df_train_complete, df_test_complete = download_data()
    df_train, df_test = clean_data(df_train_complete, df_test_complete)
    df_train_, df_test_ = get_gender_neutral_data_set(df_train, df_test)
    save_dfs(df_train_, df_test_)
    save_datasets(df_train_, df_test_)
    print("DONE")


if __name__ == '__main__':
    main()
