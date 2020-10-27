############################# Section 6.1

while len(rest_output) > 0:  
  
    number_pred = 0
    wrong_pred_input = []
    wrong_pred_output = []

    for stream_in, stream_out in zip(numpy.array(rest_input), numpy.array(rest_output)):
        
        dynamic(numpy.reshape(stream_in, [1, -1]), stream_out, correct_predictions, incorrect_predictions,
                number_pred, weight_m1, weight_m2, weight_m3, weight_m4, weight_m5, weight_m6, weight_m7, 
                weight_m8, weight_m9, weight_m10)
        
        if weight_m1 <= 0:
            all_models['model_1'][1] = 'U'
            all_models['model_1'][0] = DecisionTreeClassifier()
            weight_m1 = 1
        if weight_m2 <= 0:
            all_models['model_2'][1] = 'U'
            all_models['model_2'][0] = DecisionTreeClassifier()
            weight_m2 = 1
        if weight_m3 <= 0:
            all_models['model_3'][1] = 'U'
            all_models['model_3'][0] = DecisionTreeClassifier()
            weight_m3 = 1
        if weight_m4 <= 0:
            all_models['model_4'][1] = 'U'
            all_models['model_4'][0] = DecisionTreeClassifier()
            weight_m4 = 1
        if weight_m5 <= 0:
            all_models['model_5'][1] = 'U'
            all_models['model_5'][0] = DecisionTreeClassifier()
            weight_m5 = 1
        if weight_m6 <= 0:
            all_models['model_6'][1] = 'U'
            all_models['model_6'][0] = DecisionTreeClassifier()
            weight_m6 = 1
        if weight_m7 <= 0:
            all_models['model_7'][1] = 'U'
            all_models['model_7'][0] = DecisionTreeClassifier()
            weight_m7 = 1
        if weight_m8 <= 0:
            all_models['model_8'][1] = 'U'
            all_models['model_8'][0] = DecisionTreeClassifier()
            weight_m8 = 1
        if weight_m9 <= 0:
            all_models['model_9'][1] = 'U'
            all_models['model_9'][0] = DecisionTreeClassifier()
            weight_m9 = 1
        if weight_m10 <= 0:
            all_models['model_10'][1] = 'U'
            all_models['model_10'][0] = DecisionTreeClassifier()
            weight_m10 = 1
        
        if len(wrong_pred_output) >= tr_size:
            break
			
############################# Section 6.2			

    df_wrong_in = pandas.DataFrame(data = numpy.reshape(numpy.array(wrong_pred_input),
                                                        [-1, len(train_input.columns)]))
    col_names = train_input.columns
    df_wrong_in.columns = col_names
    df_wrong_out = pandas.DataFrame(wrong_pred_output)
    rest_input = rest_input.drop(rest_input.index[0:number_pred])
    rest_output = rest_output.drop(rest_output.index[0:number_pred])

    if all_models['model_1'][1] == 'U':
        all_models['model_1'][0].fit(df_wrong_in, df_wrong_out)
        all_models['model_1'][1] = 'T'
    elif all_models['model_2'][1] == 'U':
        all_models['model_2'][0].fit(df_wrong_in, df_wrong_out)
        all_models['model_2'][1] = 'T'
    elif all_models['model_3'][1] == 'U':
        all_models['model_3'][0].fit(df_wrong_in, df_wrong_out)
        all_models['model_3'][1] = 'T'
    elif all_models['model_4'][1] == 'U':
        all_models['model_4'][0].fit(df_wrong_in, df_wrong_out)
        all_models['model_4'][1] = 'T'
    elif all_models['model_5'][1] == 'U':
        all_models['model_5'][0].fit(df_wrong_in, df_wrong_out)
        all_models['model_5'][1] = 'T'
    elif all_models['model_6'][1] == 'U':
        all_models['model_6'][0].fit(df_wrong_in, df_wrong_out)
        all_models['model_6'][1] = 'T'
    elif all_models['model_7'][1] == 'U':
        all_models['model_7'][0].fit(df_wrong_in, df_wrong_out)
        all_models['model_7'][1] = 'T'
    elif all_models['model_8'][1] == 'U':
        all_models['model_8'][0].fit(df_wrong_in, df_wrong_out)
        all_models['model_8'][1] = 'T'
    elif all_models['model_9'][1] == 'U':
        all_models['model_9'][0].fit(df_wrong_in, df_wrong_out)
        all_models['model_9'][1] = 'T'
    elif all_models['model_10'][1] == 'U':
        all_models['model_10'][0].fit(df_wrong_in, df_wrong_out)
        all_models['model_10'][1] = 'T'

accuracy_ensemble = correct_predictions / (correct_predictions + incorrect_predictions)
adj_accuracy_ensemble = (correct_predictions / (correct_predictions + incorrect_predictions)) / (1 / adj_acc_base)

print('The chosen algorithm is Dynamic Weighted Majority')
print('The accuracy of the ensemble is', round(accuracy_ensemble, 2))
print('The adjusted accuracy of the ensemble is', round(adj_accuracy_ensemble, 2))

if all_models['model_1'][1] == 'T':
    print('The weight of model 1 is', round(weight_m1, 3))
if all_models['model_2'][1] == 'T':
    print('The weight of model 2 is', round(weight_m2, 3))
if all_models['model_3'][1] == 'T':
    print('The weight of model 3 is', round(weight_m3, 3))
if all_models['model_4'][1] == 'T':
    print('The weight of model 4 is', round(weight_m4, 3))
if all_models['model_5'][1] == 'T':
    print('The weight of model 5 is', round(weight_m5, 3))
if all_models['model_6'][1] == 'T':
    print('The weight of model 6 is', round(weight_m6, 3))
if all_models['model_7'][1] == 'T':
    print('The weight of model 7 is', round(weight_m7, 3))
if all_models['model_8'][1] == 'T':
    print('The weight of model 8 is', round(weight_m8, 3))
if all_models['model_9'][1] == 'T':
    print('The weight of model 9 is', round(weight_m9, 3))
if all_models['model_10'][1] == 'T':
    print('The weight of model 10 is', round(weight_m10, 3))
