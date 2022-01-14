from hw1_main import *

if __name__ == '__main__':
    df_train = pd.read_csv("csv_files/hw#1/text_training.csv", encoding="UTF-8")
    df_train = clean_data(df_train)
    x, y = get_xy(df_train)
    XGB = XGBClassifier(silent=False, n_jobs=13, random_state=15, n_estimators=100)
    rf = RandomForestClassifier(n_estimators=2000, min_samples_split=5, min_samples_leaf=1, max_features='auto',
                                max_depth=None, bootstrap=False)

    df = pd.read_csv("csv_files/hw#2/train_data/reviews_training.csv", encoding="UTF-8")
    res_xgb = create_recommendation_file(x, y, ("XGB",XGB), df)
    res_rf = create_recommendation_file(x, y, ("rf", rf), df)
    df = pd.read_csv("csv_files/hw#2/train_data/ffp_train.csv", encoding="UTF-8")

    df = pd.merge(df, res_xgb, on="ID", how="left")
    df = pd.merge(df, res_rf, on="ID", how="left")
    df = df.fillna(-1)
    df.to_csv("csv_files/hw#2/train_data/ffp_train_with_rev.csv", index=False)
