#%%
def tpr_weight_function(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 'tprw',0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3,True
#%%
def kfold_lightgbm(train_df,test_df, num_folds, stratified = True):
    # Divide in training/validation and test data
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=13)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=13)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Tag'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['Tag'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['Tag'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            #application='binary',
            nthread=4,
            is_unbalance=True,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=60,
            colsample_bytree=0.3010,
            subsample=0.6726,
            max_depth=8,
            reg_alpha=0.9125,
            reg_lambda=9.5702,
            min_split_gain=0.1049,
            min_child_weight=0.2663,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
        eval_metric=tpr_weight_function, verbose= 100, early_stopping_rounds= 200)
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["shap_values"] = abs(shap.TreeExplainer(clf).shap_values(valid_x)[:,:test_df[feats].shape[1]]).mean(axis=0).T
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d tpr : %.6f' % (n_fold + 1, tpr_weight_function(valid_y, oof_preds[valid_idx])[1]))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full tpr score %.6f' % tpr_weight_function(train_df['Tag'], oof_preds)[1])
    # Write submission file and plot feature importance
    test_df['Tag'] = sub_preds
    test_df[['UID', 'Tag']].to_csv('10_5.csv', encoding='utf8', index= False)
    feature_importance_df.to_csv('feature_importance_10_5.csv')
    display_importances(feature_importance_df)
    display_shapley_values(feature_importance_df)
    return feature_importance_df
# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances_10_5.jpg')
# Display/plot shapley values
def display_shapley_values(feat_importance):
    best_features = feat_importance[["feature", "shap_values"]].groupby("feature")["shap_values"].agg(['mean', 'std']) \
                                                               .sort_values(by="mean", ascending=False).head(40).reset_index()
    best_features.columns = ["feature", "mean shapley values", "err"]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="mean shapley values", y="feature", xerr=best_features['err'], data=best_features)
    plt.title('LightGBM shapley values (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_shapley values_10_5.jpg')
#%%
feat_importance = kfold_lightgbm(train,test, num_folds= 5, stratified= True)
