# random forest using sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dt import preprocess_dt as pp
import matplotlib.pyplot as plt


def _format(train_data):
    for col in train_data.columns:
        train_data[col] = train_data[col].map(lambda x: 1 if x == 'yes' else 0)
    return train_data


def part_c():
    # read the data
    # one hot encoder

    train_data = _format(pp.one_hot_encoding(pd.read_csv(
        'bank_dataset/bank_train.csv', sep=';')))

    test_data = _format(pp.one_hot_encoding(pd.read_csv(
        'bank_dataset/bank_test.csv', sep=';')))
    val_data = _format(pp.one_hot_encoding(pd.read_csv(
        'bank_dataset/bank_val.csv', sep=';')))

    X_train, Y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, Y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
    X_val, Y_val = val_data.iloc[:, :-1], val_data.iloc[:, -1]
    # print(X_train)
    param_grid = {'n_estimators': [50, 150, 250, 350, 450], 'max_features': [
        0.1, 0.3, 0.5, 0.7, 0.9], 'min_samples_split': [2,  4, 6, 8, 10]}
    oob_scores = []
    maxsofar = -1
    best_features = []
    for n_estimate in param_grid['n_estimators']:
        for max_feature in param_grid['max_features']:
            for min_samples_split in param_grid['min_samples_split']:
                rf = RandomForestClassifier(criterion='entropy', bootstrap=True,
                                            oob_score=True, n_estimators=n_estimate, max_features=max_feature, min_samples_split=min_samples_split)
                rf.fit(X_train, Y_train)
                print("n_estimators: ", n_estimate, "max_features: ",
                      max_feature, "min_samples_split: ", min_samples_split)
                print(rf.oob_score_)
                oob_scores.append(rf.oob_score_)
                # write oob_score to csv
                with open('dt_results/oob_score.csv', 'a') as f:
                    f.write(str(n_estimate) + ',' + str(max_feature) + ',' +
                            str(min_samples_split) + ',' + str(rf.oob_score_) + '\n')
                if rf.oob_score_ > maxsofar:
                    maxsofar = rf.oob_score_
                    best_features = [n_estimate,
                                     max_feature, min_samples_split]
    print("Best features: ", best_features)
    # best_features = [350, 0.3, 10]
    rf_best = RandomForestClassifier(bootstrap=True, oob_score=True,
                                     n_estimators=best_features[0], max_features=best_features[1], min_samples_split=best_features[2])
    rf_best.fit(X_train, Y_train)
    print("oob score:", rf_best.oob_score_)
    print("Train accuracy:", rf_best.score(X_train, Y_train))
    print("Test accuracy:", rf_best.score(X_test, Y_test))
    print("Val accuracy:", rf_best.score(X_val, Y_val))


def part_d():
    # read the data
    # one hot encoder

    train_data = _format(pp.one_hot_encoding(pd.read_csv(
        'bank_dataset/bank_train.csv', sep=';')))

    test_data = _format(pp.one_hot_encoding(pd.read_csv(
        'bank_dataset/bank_test.csv', sep=';')))
    val_data = _format(pp.one_hot_encoding(pd.read_csv(
        'bank_dataset/bank_val.csv', sep=';')))

    X_train, Y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, Y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
    X_val, Y_val = val_data.iloc[:, :-1], val_data.iloc[:, -1]
    # print(X_train)
    param_grid = {'n_estimators': [50, 150, 250, 350, 450], 'max_features': [
        0.1, 0.3, 0.5, 0.7, 0.9], 'min_samples_split': [2,  4, 6, 8, 10]}
    optimal_parameters = {'n_estimators': 350,
                          'max_features': 0.3, 'min_samples_split': 10}
    n_est_train = []
    n_est_test = []
    n_est_val = []
    max_f_train = []
    max_f_test = []
    max_f_val = []
    min_s_train = []
    min_s_test = []
    min_s_val = []
    for n_estimate in param_grid['n_estimators']:
        print("n_estimators: ", n_estimate)
        rf = RandomForestClassifier(criterion='entropy', bootstrap=True,
                                    oob_score=True, n_estimators=n_estimate, max_features=optimal_parameters['max_features'], min_samples_split=optimal_parameters['min_samples_split'])
        rf.fit(X_train, Y_train)
        print(rf.oob_score_)
        n_est_train.append(rf.oob_score_)
        n_est_test.append(rf.score(X_test, Y_test))
        n_est_val.append(rf.score(X_val, Y_val))
    # plot the results
    plt.figure()
    plt.plot(param_grid['n_estimators'], n_est_train, label='train-oob-score')
    plt.plot(param_grid['n_estimators'], n_est_test, label='test')
    plt.plot(param_grid['n_estimators'], n_est_val, label='val')
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.title('n_estimators vs accuracy')
    plt.legend()
    plt.savefig('dt_results/dt-qd-n_estimators_accuracy.png')

    for max_feature in param_grid['max_features']:
        print("max_features: ", max_feature)
        rf = RandomForestClassifier(criterion='entropy', bootstrap=True,
                                    oob_score=True, n_estimators=optimal_parameters['n_estimators'], max_features=max_feature, min_samples_split=optimal_parameters['min_samples_split'])
        rf.fit(X_train, Y_train)
        max_f_train.append(rf.oob_score_)
        max_f_test.append(rf.score(X_test, Y_test))
        max_f_val.append(rf.score(X_val, Y_val))
    # plot the results
    plt.figure()
    plt.plot(param_grid['max_features'], max_f_train, label='train-oob-score')
    plt.plot(param_grid['max_features'], max_f_test, label='test')
    plt.plot(param_grid['max_features'], max_f_val, label='val')
    plt.xlabel('max_features')
    plt.ylabel('accuracy')
    plt.title('max_features vs accuracy')
    plt.legend()
    plt.savefig('dt_results/dt-qd-max_features_accuracy.png')

    for min_samples_split in param_grid['min_samples_split']:
        print("min_samples_split: ", min_samples_split)
        rf = RandomForestClassifier(criterion='entropy', bootstrap=True,
                                    oob_score=True, n_estimators=optimal_parameters['n_estimators'], max_features=optimal_parameters['max_features'], min_samples_split=min_samples_split)
        rf.fit(X_train, Y_train)
        min_s_train.append(rf.oob_score_)
        min_s_test.append(rf.score(X_test, Y_test))
        min_s_val.append(rf.score(X_val, Y_val))
    # plot the results
    plt.figure()
    plt.plot(param_grid['min_samples_split'],
             min_s_train, label='train-oob-score')
    plt.plot(param_grid['min_samples_split'], min_s_test, label='test')
    plt.plot(param_grid['min_samples_split'], min_s_val, label='val')
    plt.xlabel('min_samples_split')
    plt.ylabel('accuracy')
    plt.title('min_samples_split vs accuracy')
    plt.legend()
    plt.savefig('dt_results/dt-qd-min_samples_split_accuracy.png')
