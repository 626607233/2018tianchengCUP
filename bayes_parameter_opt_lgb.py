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
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3


def eval_error(pred, train_set):
    labels = train_set.get_label()
    score = tpr_weight_function(labels, pred)
    return 'TPR', score, True

#%%

def bayes_parameter_opt_lgb(X, y, init_round, opt_round, n_folds, random_seed, n_estimators, learning_rate, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {'application':'binary','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':200, 'eval_metric':'tprw','is_unbalance':True}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =100,feval=eval_error)
        return max(cv_result['TPR-mean'])
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (30, 80),
                                            'feature_fraction': (0.1, 0.4),
                                            'bagging_fraction': (0.6, 0.9),
                                            'max_depth': (4, 8),
                                            'lambda_l1': (0, 10),
                                            'lambda_l2': (0, 10),
                                            'min_split_gain': (0.01, 0.6),
                                            'min_child_weight': (0.1, 30)},random_state=random_seed)#
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return best parameters
    return lgbBO.res['max']['max_params']

#%%
opt_params = bayes_parameter_opt_lgb(train[feats],train['Tag'], init_round=5, opt_round=10, n_folds=5, random_seed=13, n_estimators=10000, learning_rate=0.02)
print(opt_params)
