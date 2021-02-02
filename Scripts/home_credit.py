# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables.
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))



# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Alıntı başlangıç
def feature_early_shutdown(row):
    if row.CREDIT_ACTIVE == "Closed" and row.DAYS_ENDDATE_FACT < row.DAYS_CREDIT_ENDDATE:
        return 1
    elif row.CREDIT_ACTIVE == "Closed" and row.DAYS_CREDIT_ENDDATE <= row.DAYS_ENDDATE_FACT:
        return 0
    else:
        return np.nan


def get_age_label(days_birth):
    """ Return the age group label (int). """
    age_years = -days_birth / 365
    if age_years < 27: return 1
    elif age_years < 40: return 2
    elif age_years < 50: return 3
    elif age_years < 65: return 4
    elif age_years < 99: return 5
    else: return 0


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows=None, nan_as_category=False):
    # Read data and merge
    df = pd.read_csv('HomeCredit/Dataset/application_train.csv', nrows=num_rows)
    test_df = pd.read_csv('HomeCredit/Dataset/application_test.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)


    # Başvuru sırasında müşterinin gün cinsinden yaşı eksili olarak verilmiş
    # -1 ile çarpıp 365'e böldüğümüzde kaç yaşında olduğunu buluyoruz
    df["NEW_APP_AGE"] = round(df["DAYS_BIRTH"] * -1 / 365)

    # Yaşa göre sınıflandırma
    df['NEW_APP_AGE_RANGE'] = df['NEW_APP_AGE'].apply(lambda x: get_age_label(x))

    # Başvurudan kaç gün önce kişinin mevcut işe başladığı eksili olrak ifade edilmiş
    # -1 ile çarpıp 365'e böldüğümüzde yıl cinsinden hesaplamış oluyoruz
    df["NEW_APP_YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"] * -1 / 365

    # Kredi Miktarı / kredi yıllık ödemesi = kredi ödeme dönem sayısı
    df["NEW_APP_CREDIT_TERM"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

    # Verilen kredi miktarı kredinin verilidiği malların miktarından büyük ise 1 değilse 0
    # etkisi az
    df["NEW_APP_OVER_EXPECT_CREDIT"] = (df["AMT_CREDIT"] > df["AMT_GOODS_PRICE"]).map({False: 0, True: 1})

    #Kredi Yıllık Ödemesi / Müşterinin Geliri
    df["NEW_APP_ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    #Kişinin kaç gündür mevcut işte çalıştığı / kişinin güm cinsinden yaşı
    df["NEW_APP_EMPLOYED_AGE_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    # 20 belgenin sağlanıp sağlanmadığını gösteren 20 değişken var
    # Bu değişkenler Flag_Doc olarak ifade edilmiş
    # Contains kullnarak bir listeye atıyoruz isimlerini
    # Toplam sağlanan belge sayısını gösteren yeni bir değişken oluşturuyoruz
    flag_docs = df.columns[df.columns.str.contains("FLAG_DOC", regex=True)]
    df["NEW_APP_TOTAL_DOCS"] = df[flag_docs].sum(axis=1)

    #3 tane dış kaynaklardan sağlanmış skor var
    #bunların çarpımından yeni bir değişken oluşturduk
    df['NEW_APP_EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    #önem grafiğine bakıp ağırlaklandırıp değerleri toplayıp yenş bir değişken oluşturduk.
    # etkisi az
    df['NEW_APP_EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_2 * 3 + df.EXT_SOURCE_3 * 4

    #Müşterinin geliri / çalıştığı gün sayısı
    df['NEW_APP_INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']* -1

    # Müşterinin geliri / gün cinsinden yaşı
    df['NEW_APP_INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']* -1


    #drop_list = ['FLAG_OWN_REALTY', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'REGION_RATING_CLIENT',
    #            'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_WORK_CITY',
    #             'LIVE_CITY_NOT_WORK_CITY', 'COMMONAREA_AVG', 'LANDAREA_AVG', 'ELEVATORS_MODE',
    #             'ENTRANCES_MODE', 'NONLIVINGAPARTMENTS_MODE', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI',
    #             'ENTRANCES_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',
    #             'NONLIVINGAPARTMENTS_MEDI', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
    #             'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
    #             'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
    #             'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON']

    #df.drop(drop_list, axis=1, inplace=True)

    del test_df
    gc.collect()
    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('HomeCredit/Dataset/bureau.csv', nrows=num_rows)
    bb = pd.read_csv('HomeCredit/Dataset/bureau_balance.csv', nrows=num_rows)


    bureau['CREDIT_DURATION'] = -bureau['DAYS_CREDIT'] + bureau['DAYS_CREDIT_ENDDATE']
    bureau['ENDDATE_DIF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    # Credit to debt ratio and difference
    bureau['DEBT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
    bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    bureau['CREDIT_TO_ANNUITY_RATIO'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']


    bureau['CREDIT_ACTIVE'] = bureau['CREDIT_ACTIVE'].replace(['Bad debt', 'Sold'], 'Closed')
    bureau['CREDIT_CURRENCY'] = bureau['CREDIT_CURRENCY'].replace(['currency 2', 'currency 3', 'currency 4'],'currency others')
    bureau['CNT_CREDIT_PROLONG'] = bureau['CNT_CREDIT_PROLONG'].replace([2, 3, 4, 5, 6, 7, 8, 9],1)

    # Açık kredilerin erken kapanması yoktur. Bu yüzden süreleri, kredinin başlangıç günü + biteceği günden hesaplanır.
    bureau.loc[bureau["CREDIT_ACTIVE"] == "Active", "NEW_KREDI_SURESI"] = - (bureau.loc[bureau["CREDIT_ACTIVE"] == "Active", "DAYS_CREDIT"]) + bureau.loc[bureau["CREDIT_ACTIVE"] == "Active", "DAYS_CREDIT_ENDDATE"]

    # Kapalı krediler erken kapanmış olabilir. Bu yüzden süreleri, kredinin başlangıç günü + kapatıldığı gündür.
    bureau.loc[bureau["CREDIT_ACTIVE"] == "Closed", "NEW_KREDI_SURESI"] = -(bureau.loc[bureau["CREDIT_ACTIVE"] == "Closed", "DAYS_CREDIT"]) + bureau.loc[ bureau["CREDIT_ACTIVE"] == "Closed", "DAYS_ENDDATE_FACT"]

    # Kredinin kapatılma tarihleri arasındaki fark. Erken kapanırsa +, geç kapatılırsa - değer alır. Zamanındakiler için 0 olur.
    bureau['NEW_ENDDATE_FARK'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']

    # Anlık kredinin yıllık krediye oranı
    bureau['NEW_KREDI/YILLIK_ORAN'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']

    # Açık kredilerin borçları krediden yüksekse, ödenmesi gereken kredi tutarı bu borç olarak değiştirilmelidir.
    # Kapalı krediler için 0 değeri atanmalıdır.
    bureau["ODENMESI_GEREKEN_TUTAR"] = np.nan

    bureau.loc[bureau["AMT_CREDIT_SUM_DEBT"] > bureau["AMT_CREDIT_SUM"], "NEW_ODENMESI_GEREKEN_TUTAR"] = \
        bureau.loc[bureau["AMT_CREDIT_SUM_DEBT"] > bureau["AMT_CREDIT_SUM"], "AMT_CREDIT_SUM_DEBT"]

    bureau.loc[bureau["CREDIT_ACTIVE"] == "Closed", "NEW_ODENMESI_GEREKEN_TUTAR"] = 0.0

    # Kredinin erken ödenip ödenmemesi. erken ödeme = 1, ödememe 0. Aktif krediler için ise nan

    bureau["NEW_ERKEN_KAPAMA"] = bureau.apply(lambda x: feature_early_shutdown(x), axis=1)

    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)


    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum','mean','max'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'NEW_ODENMESI_GEREKEN_TUTAR': ['max', 'mean', 'sum'],
        'NEW_ERKEN_KAPAMA': ['max', 'mean','sum'],
        'NEW_KREDI/YILLIK_ORAN':['min', 'max', 'mean'],
        'NEW_ENDDATE_FARK':['min', 'max', 'mean'],
        'NEW_KREDI_SURESI':['min', 'max', 'mean']
    }

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv('HomeCredit/Dataset/previous_application.csv', nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    prev['NEW_PREV_APPLICATION_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['NEW_PREV_APPLICATION_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['NEW_PREV_CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
    prev['NEW_PREV_DOWN_PAYMENT_TO_CREDIT'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']
    total_payment = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    prev['NEW_PREV_SIMPLE_INTERESTS'] = (total_payment / prev['AMT_CREDIT'] - 1) / prev['CNT_PAYMENT']

    prev["NEW_PREV_new_1"] = (prev.AMT_DOWN_PAYMENT * prev.RATE_DOWN_PAYMENT)
    prev["NEW_PREV_new_2"] = (prev.AMT_DOWN_PAYMENT * prev.AMT_CREDIT)
    prev["NEW_PREV_new_3"] = (prev.AMT_APPLICATION * prev.AMT_GOODS_PRICE)
    prev["NEW_PREV_new_4"] = (prev.AMT_DOWN_PAYMENT * prev.AMT_APPLICATION)
    prev["NEW_PREV_new_5"] = (prev.AMT_DOWN_PAYMENT * prev.AMT_ANNUITY)

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'NEW_PREV_APPLICATION_CREDIT_DIFF':['min', 'max', 'mean', 'var'],
        'NEW_PREV_APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean', 'var'],
        'NEW_PREV_CREDIT_TO_ANNUITY_RATIO':['min', 'max', 'mean', 'var'],
        'NEW_PREV_DOWN_PAYMENT_TO_CREDIT':['min', 'max', 'mean', 'var'],
        'NEW_PREV_SIMPLE_INTERESTS': ['min', 'max', 'mean', 'var'],
        'NEW_PREV_new_1':['min', 'max', 'mean', 'var'],
        'NEW_PREV_new_2': ['min', 'max', 'mean', 'var'],
        'NEW_PREV_new_3': ['min', 'max', 'mean', 'var'],
        'NEW_PREV_new_4': ['min', 'max', 'mean', 'var'],
        'NEW_PREV_new_5': ['min', 'max', 'mean', 'var']

    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('HomeCredit/Dataset/POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)

    pos['NEW_POS_LATE_PAYMENT'] = pos['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'NEW_POS_LATE_PAYMENT': ['mean']

    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv('HomeCredit/Dataset/installments_payments.csv', nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['NEW_INS_PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['NEW_INS_PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['NEW_INS_DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['NEW_INS_DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['NEW_INS_DPD'] = ins['NEW_INS_DPD'].apply(lambda x: x if x > 0 else 0)
    ins['NEW_INS_DBD'] = ins['NEW_INS_DBD'].apply(lambda x: x if x > 0 else 0)


    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'NEW_INS_DPD': ['max', 'mean', 'sum'],
        'NEW_INS_DBD': ['max', 'mean', 'sum'],
        'NEW_INS_PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'NEW_INS_PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('HomeCredit/Dataset/credit_card_balance.csv', nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)

    cc['NEW_CC_LIMIT_USE'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc['NEW_CC_LIMIT_USE2'] = cc['AMT_CREDIT_LIMIT_ACTUAL'] / cc['AMT_BALANCE']

    cc['NEW_CC_PAYMENT_DIV_MIN'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY']

    cc['NEW_CC_LATE_PAYMENT'] = cc['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)

    cc['NEW_CC_DRAWING_LIMIT_RATIO'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL']

    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=200, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)
    display_importances(feature_importance_df)
    return feature_importance_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[:100].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def main(debug=True):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds=10, stratified=False, debug=debug)


if __name__ == "__main__":
    submission_file_name = "submission_main.csv"
    with timer("Full model run"):
        main()