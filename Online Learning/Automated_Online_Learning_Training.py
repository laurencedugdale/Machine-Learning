############################# Section 1

import numpy
import pandas
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = pandas.read_csv(input('Enter the filename of the dataset to use for training:'))

if vars().__contains__('mod_dec'):
    pass
else:
    output = input('Enter the name of the column containing the target attribute:')
    d_size = len(data[output])
    adj_acc_base = data[output].nunique()

if vars().__contains__('mod_dec'):
    pass
else:
    column_number = 0
    categorical_cols = []
    numerical_cols = []
    for column_type in data.dtypes:
        if column_type == object and data.columns[column_number] != output:
            categorical_cols.append(column_number)
        elif data.columns[column_number] != output:
            numerical_cols.append(column_number)
        column_number += 1         
    
data_categorical = data[data.columns[categorical_cols]]
data_numerical = data[data.columns[numerical_cols]]

if len(categorical_cols) > 0:
    if vars().__contains__('mod_dec'):
        pass
    else:
        number_unique = data_categorical.nunique()
        col_num = 0
        low_nuniq_cols = []
        for col_nuniq in number_unique:
            if col_nuniq <= 20:
                low_nuniq_cols.append(col_num)
            col_num +=1

    data_categorical = data_categorical[data_categorical.columns[low_nuniq_cols]]
    data_categorical = data_categorical.fillna('None')    
    
    if vars().__contains__('mod_dec'):
        encoder = OneHotEncoder(categories = enc_cat, sparse = False, handle_unknown = 'ignore')
        encoder = encoder.fit_transform(data_categorical)
    else:
        encoder = OneHotEncoder(sparse = False)
        encoder = encoder.fit(data_categorical)
        enc_cat = encoder.categories_
        encoder = encoder.transform(data_categorical)
        
    data_categorical = pandas.DataFrame(encoder)

if len(numerical_cols) > 0:
    scaler = StandardScaler().fit(data_numerical)
    data_numerical = pandas.DataFrame(scaler.transform(data_numerical))
    data_numerical = data_numerical.fillna(0) 

data = data_categorical.join(other = data_numerical.join(data[output]), rsuffix = 'n')

############################# Section 2

if vars().__contains__('mod_dec'):
    if mod_dec == 'dynamic':
        rest = data
        rest_input = rest.drop(output, axis=1)
        rest_output = rest[output]
        
        exec(open('dynamic weighted majority.py').read())
    else:
        test_ensemble = data
        test_ensemble_input = test_ensemble.drop(output, axis=1)
        test_ensemble_output = test_ensemble[output]
        
        exec(open('weighted majority.py').read())
		
############################# Section 3		
		
else:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler
    
    n_features = len(data.columns)
    n_rows = len(data[data.columns[0]])
    number_categories = len(data_categorical.columns)

    if len(categorical_cols) > 0:
        ratio_num_cat = len(numerical_cols) / len(categorical_cols)
    else:
        ratio_num_cat = 'NA'

    if len(numerical_cols) > 0:
        mean_of_skew = ((data_numerical.mean() - data_numerical.median()) / data_numerical.std()).mean()
        std_of_skew = ((data_numerical.mean() - data_numerical.median()) / data_numerical.std()).std()
    else:
        mean_of_skew = 'NA'
        std_of_skew = 'NA'
    
    metadata = [n_features, n_rows, number_categories, ratio_num_cat, mean_of_skew, std_of_skew] 
    all_metadata = pandas.read_csv('all metadata')
    all_metadata_input = all_metadata.drop(['Best model'], axis = 1)
    all_metadata_output = all_metadata['Best model']
    
    norm = MinMaxScaler()
    norm.fit(all_metadata_input)
    transf_inp = norm.transform(all_metadata_input)
    transf_met = norm.transform(numpy.reshape(metadata, [1, -1]))
    
    mod_select = KNeighborsClassifier(n_neighbors = 1)
    mod_select.fit(transf_inp, all_metadata_output)
    mod_chosen = mod_select.predict(transf_met)
	
############################# Section 4.1	
    
    if mod_chosen == 'DWM':
        mod_dec = 'dynamic'
        
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        
        if d_size >= 1000:
            tr_size = 500
        else:
            tr_size = d_size / 2
        
        train, rest = train_test_split(data, train_size = tr_size, shuffle = False)
        train_input = train.drop(output, axis=1)
        train_output = train[output]
        rest_input = rest.drop(output, axis=1)
        rest_output = rest[output]

        all_models = {'model_1' : [DecisionTreeClassifier(), 'U'],
                      'model_2' : [DecisionTreeClassifier(), 'U'],
                      'model_3' : [DecisionTreeClassifier(), 'U'],
                      'model_4' : [DecisionTreeClassifier(), 'U'],
                      'model_5' : [DecisionTreeClassifier(), 'U'],
                      'model_6' : [DecisionTreeClassifier(), 'U'],
                      'model_7' : [DecisionTreeClassifier(), 'U'],
                      'model_8' : [DecisionTreeClassifier(), 'U'],
                      'model_9' : [DecisionTreeClassifier(), 'U'],
                      'model_10' : [DecisionTreeClassifier(), 'U']}

        all_models['model_1'][0].fit(train_input, train_output)
        all_models['model_1'][1] = 'T'
		
############################# Section 4.2		

        def dynamic(new_input, new_output, prior_correct_predictions, prior_incorrect_predictions, 
                    prior_number_pred, p_weight_m1, p_weight_m2, p_weight_m3, p_weight_m4, p_weight_m5, 
                    p_weight_m6, p_weight_m7, p_weight_m8, p_weight_m9, p_weight_m10):

            global number_pred, correct_predictions, incorrect_predictions, weight_m1, weight_m2, weight_m3
            global weight_m4, weight_m5, weight_m6, weight_m7, weight_m8, weight_m9, weight_m10

            if all_models['model_1'][1] == 'T': 
                pred_model_1 = all_models['model_1'][0].predict(new_input)
            else:
                pred_model_1 = None           
            if all_models['model_2'][1] == 'T': 
                pred_model_2 = all_models['model_2'][0].predict(new_input)
            else:
                pred_model_2 = None
            if all_models['model_3'][1] == 'T': 
                pred_model_3 = all_models['model_3'][0].predict(new_input)
            else:
                pred_model_3 = None
            if all_models['model_4'][1] == 'T': 
                pred_model_4 = all_models['model_4'][0].predict(new_input)
            else:
                pred_model_4 = None
            if all_models['model_5'][1] == 'T': 
                pred_model_5 = all_models['model_5'][0].predict(new_input)
            else:
                pred_model_5 = None
            if all_models['model_6'][1] == 'T': 
                pred_model_6 = all_models['model_6'][0].predict(new_input)
            else:
                pred_model_6 = None
            if all_models['model_7'][1] == 'T': 
                pred_model_7 = all_models['model_7'][0].predict(new_input)
            else:
                pred_model_7 = None
            if all_models['model_8'][1] == 'T': 
                pred_model_8 = all_models['model_8'][0].predict(new_input)
            else:
                pred_model_8 = None
            if all_models['model_9'][1] == 'T': 
                pred_model_9 = all_models['model_9'][0].predict(new_input)
            else:
                pred_model_9 = None
            if all_models['model_10'][1] == 'T': 
                pred_model_10 = all_models['model_10'][0].predict(new_input)
            else:
                pred_model_10 = None

            uniq_preds = {}
            all_preds = [[pred_model_1, weight_m1], [pred_model_2, weight_m2], [pred_model_3, weight_m3],
                         [pred_model_4, weight_m4], [pred_model_5, weight_m5], [pred_model_6, weight_m6], 
                         [pred_model_7, weight_m7], [pred_model_8, weight_m8], [pred_model_9, weight_m9],
                         [pred_model_10, weight_m10]]

            for m in all_preds:
                if m[0] != None:
                    if m[0][0] not in uniq_preds:
                        uniq_preds.update({m[0][0]:m[1]})
                    else:
                        uniq_preds.update({m[0][0]:(uniq_preds[m[0][0]] + m[1])})

            ensemble_pred = max(uniq_preds, key = lambda d:uniq_preds[d])
			
############################# Section 4.3			

            if new_output != 'unknown':   
                number_pred = prior_number_pred + 1

                if ensemble_pred == new_output:
                    correct_predictions = prior_correct_predictions + 1
                else:
                    incorrect_predictions = prior_incorrect_predictions + 1
                    wrong_pred_input.append(new_input)
                    wrong_pred_output.append(new_output)

                if pred_model_1 != new_output:
                    if pred_model_1 != None:
                        if weight_m1 >= 0.001: 
                            weight_m1 = p_weight_m1 - 0.0005
                elif weight_m1 < 5:
                    weight_m1 = p_weight_m1 + 0.0005
                if pred_model_2 != new_output:
                    if pred_model_2 != None:
                        weight_m2 = p_weight_m2 - 0.0005
                elif weight_m2 < 5:
                    weight_m2 = p_weight_m2 + 0.0005
                if pred_model_3 != new_output:
                    if pred_model_3 != None:
                        weight_m3 = p_weight_m3 - 0.0005
                elif weight_m3 < 5:
                    weight_m3 = p_weight_m3 + 0.0005  
                if pred_model_4 != new_output:
                    if pred_model_4 != None:
                        weight_m4 = p_weight_m4 - 0.0005
                elif weight_m4 < 5:
                    weight_m4 = p_weight_m4 + 0.0005
                if pred_model_5 != new_output:
                    if pred_model_5 != None:
                        weight_m5 = p_weight_m5 - 0.0005
                elif weight_m5 < 5:
                    weight_m5 = p_weight_m5 + 0.0005
                if pred_model_6 != new_output:
                    if pred_model_6 != None:
                        weight_m6 = p_weight_m6 - 0.0005
                elif weight_m6 < 5:
                    weight_m6 = p_weight_m6 + 0.0005
                if pred_model_7 != new_output:
                    if pred_model_7 != None:
                        weight_m7 = p_weight_m7 - 0.0005
                elif weight_m7 < 5:
                    weight_m7 = p_weight_m7 + 0.0005
                if pred_model_8 != new_output:
                    if pred_model_8 != None:
                        weight_m8 = p_weight_m8 - 0.0005
                elif weight_m8 < 5:
                    weight_m8 = p_weight_m8 + 0.0005
                if pred_model_9 != new_output:
                    if pred_model_9 != None:
                        weight_m9 = p_weight_m9 - 0.0005
                elif weight_m9 < 5:
                    weight_m9 = p_weight_m9 + 0.0005
                if pred_model_10 != new_output:
                    if pred_model_10 != None:
                        weight_m10 = p_weight_m10 - 0.0005
                elif weight_m10 < 5:
                    weight_m10 = p_weight_m10 + 0.0005
            else:
                blind_predictions.append(ensemble_pred)

        correct_predictions = 0
        incorrect_predictions = 0
        weight_m1 = 1
        weight_m2 = 1
        weight_m3 = 1
        weight_m4 = 1
        weight_m5 = 1
        weight_m6 = 1
        weight_m7 = 1
        weight_m8 = 1
        weight_m9 = 1
        weight_m10 = 1
        
        exec(open('dynamic weighted majority.py').read())
		
############################# Section 5.1		
		
    else:
        mod_dec = 'weighted'
        
        import numpy
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.model_selection import GridSearchCV
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn import svm
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score

        train, test = train_test_split(data, test_size = 0.3)
        test_models, test_ensemble = train_test_split(test, test_size = 0.5)
        train_input = train.drop(output, axis=1)
        train_output = train[output]
        test_models_input = test_models.drop(output, axis=1)
        test_models_output = test_models[output]
        test_ensemble_input = test_ensemble.drop(output, axis=1)
        test_ensemble_output = test_ensemble[output]

        grid_tree = {'min_samples_split':[2, 3, 4, 5]}
        initial_tree = DecisionTreeClassifier()
        optimiser_tree = GridSearchCV(initial_tree, grid_tree, cv = 3)
        optimiser_tree.fit(train_input, train_output)
        model_tree = optimiser_tree.best_estimator_
        predictions_tree = model_tree.predict(test_models_input)
        accuracy_tree = accuracy_score(test_models_output, predictions_tree)

        grid_knn = {'n_neighbors':[3, 5, 7, 9], 'metric':['euclidean','manhattan']}
        initial_knn = KNeighborsClassifier()
        optimiser_knn = GridSearchCV(initial_knn, grid_knn, cv = 3)
        optimiser_knn.fit(train_input, train_output)
        model_knn = optimiser_knn.best_estimator_
        predictions_knn = model_knn.predict(test_models_input)
        accuracy_knn = accuracy_score(test_models_output, predictions_knn)

        grid_nb = {'var_smoothing':[0.75e-9, 1e-9, 1.25e-9]}
        initial_nb = GaussianNB()
        optimiser_nb = GridSearchCV(initial_nb, grid_nb, cv = 3)
        optimiser_nb.fit(train_input, train_output)
        model_nb = optimiser_nb.best_estimator_
        predictions_nb = model_nb.predict(test_models_input)
        accuracy_nb = accuracy_score(test_models_output, predictions_nb)

        model_svm = svm.SVC(gamma = 'auto')
        model_svm.fit(train_input, train_output)
        predictions_svm = model_svm.predict(test_models_input)
        accuracy_svm = accuracy_score(test_models_output, predictions_svm)

        grid_nn = {'hidden_layer_sizes':[(5), (10), (15)]}
        initial_nn = MLPClassifier(max_iter = 500, random_state = 1)
        optimiser_nn = GridSearchCV(initial_nn, grid_nn, cv = 3)
        optimiser_nn.fit(train_input, train_output)
        model_nn = optimiser_nn.best_estimator_
        predictions_nn = model_nn.predict(test_models_input)
        accuracy_nn = accuracy_score(test_models_output, predictions_nn)

        scores = [{'model':model_tree, 'accuracy':accuracy_tree}, {'model':model_knn, 'accuracy':accuracy_knn},
                  {'model':model_nb, 'accuracy':accuracy_nb}, {'model':model_svm, 'accuracy':accuracy_svm}, 
                  {'model':model_nn, 'accuracy':accuracy_nn}]

        scores_sorted = sorted(scores, key = lambda d:d['accuracy'], reverse = True)

        first_model = scores_sorted[0]['model']
        second_model = scores_sorted[1]['model']
        third_model = scores_sorted[2]['model']
		
############################# Section 5.2		

        def online(new_data_input, new_data_output, prior_weight_fm, prior_weight_sm, prior_weight_tm,
                   prior_correct_predictions, prior_incorrect_predictions):

            global weight_fm, weight_sm, weight_tm, correct_predictions, incorrect_predictions, whole_ensemble_prediction

            ensemble_prediction_fm = first_model.predict(new_data_input)
            ensemble_prediction_sm = second_model.predict(new_data_input)
            ensemble_prediction_tm = third_model.predict(new_data_input)

            if ensemble_prediction_fm == ensemble_prediction_sm == ensemble_prediction_tm:
                    whole_ensemble_prediction = ensemble_prediction_fm
            elif ensemble_prediction_fm == ensemble_prediction_sm:
                if weight_fm + weight_sm >= weight_tm:
                    whole_ensemble_prediction = ensemble_prediction_fm
                else:
                    whole_ensemble_prediction = ensemble_prediction_tm
            elif ensemble_prediction_fm == ensemble_prediction_tm:
                if weight_fm + weight_tm >= weight_sm:
                    whole_ensemble_prediction = ensemble_prediction_fm
                else:
                    whole_ensemble_prediction = ensemble_prediction_sm
            elif ensemble_prediction_sm == ensemble_prediction_tm:
                if weight_sm + weight_tm >= weight_fm:
                    whole_ensemble_prediction = ensemble_prediction_sm
                else:
                    whole_ensemble_prediction = ensemble_prediction_fm
            else:
                if weight_fm > weight_sm and weight_fm > weight_tm:
                    whole_ensemble_prediction = ensemble_prediction_fm
                elif weight_sm > weight_fm and weight_sm > weight_tm:
                    whole_ensemble_prediction = ensemble_prediction_sm
                elif weight_tm > weight_fm and weight_tm > weight_sm:
                    whole_ensemble_prediction = ensemble_prediction_tm
                elif weight_fm == weight_sm == weight_tm:
                    whole_ensemble_prediction = ensemble_prediction_fm
                elif weight_fm == weight_sm:
                    whole_ensemble_prediction = ensemble_prediction_fm
                elif weight_fm == weight_tm:
                    whole_ensemble_prediction = ensemble_prediction_fm
                else:
                    whole_ensemble_prediction = ensemble_prediction_sm
					
############################# Section 5.3					

            if new_data_output != 'unknown':            
                if ensemble_prediction_fm != new_data_output:
                    weight_fm = prior_weight_fm * 0.999
                if ensemble_prediction_sm != new_data_output:
                    weight_sm = prior_weight_sm * 0.999
                if ensemble_prediction_tm != new_data_output:
                    weight_tm = prior_weight_tm * 0.999
                if weight_fm >= weight_sm and weight_fm >= weight_tm:
                    weight_sm = weight_sm / weight_fm
                    weight_tm = weight_tm / weight_fm
                    weight_fm = weight_fm / weight_fm
                elif weight_sm >= weight_tm:
                    weight_fm = weight_fm / weight_sm
                    weight_tm = weight_tm / weight_sm
                    weight_sm = weight_sm / weight_sm
                else:
                    weight_fm = weight_fm / weight_tm
                    weight_sm = weight_sm / weight_tm
                    weight_tm = weight_tm / weight_tm
                if whole_ensemble_prediction == new_data_output:
                    correct_predictions = prior_correct_predictions + 1
                else:
                    incorrect_predictions = prior_incorrect_predictions + 1
            else:
                blind_predictions.append(whole_ensemble_prediction)

        weight_fm = 1
        weight_sm = 1
        weight_tm = 1
        correct_predictions = 0
        incorrect_predictions = 0
            
        exec(open('weighted majority.py').read())
