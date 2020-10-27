############################# Section 8

prediction_data = pandas.read_csv(input('Enter the filename of the dataset to be predicted:'))
pred_cat = prediction_data[prediction_data.columns[categorical_cols]]
pred_num = prediction_data[prediction_data.columns[numerical_cols]]

if len(categorical_cols) > 0:
    pred_cat = pred_cat[pred_cat.columns[low_nuniq_cols]]
    pred_cat = pred_cat.fillna('None')
    pred_enc = OneHotEncoder(categories = enc_cat, sparse = False, handle_unknown = 'ignore')
    pred_enc = pred_enc.fit_transform(pred_cat)
    pred_cat = pandas.DataFrame(pred_enc)
    
if len(numerical_cols) > 0:
    pred_num = pandas.DataFrame(scaler.transform(pred_num))
    pred_num = pred_num.fillna(0)

prediction_data_mod = pred_cat.join(other = pred_num.join(prediction_data[output]), rsuffix = 'n')
pred_dat_input = prediction_data_mod.drop(output, axis = 1)
pred_dat_output = prediction_data_mod[output]
blind_predictions = []

############################# Section 9

if mod_dec == 'dynamic':
    for stream_in, stream_out in zip(numpy.array(pred_dat_input), numpy.array(pred_dat_output)):

        dynamic(numpy.reshape(stream_in, [1, -1]), stream_out, correct_predictions, incorrect_predictions,
                number_pred, weight_m1, weight_m2, weight_m3, weight_m4, weight_m5, weight_m6, weight_m7, 
                weight_m8, weight_m9, weight_m10)

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
else:
    for stream_input, stream_output in zip(numpy.array(pred_dat_input), numpy.array(pred_dat_output)):
        
        online(numpy.reshape(stream_input, [1, -1]), stream_output, weight_fm, weight_sm, weight_tm,
               correct_predictions, incorrect_predictions)

    print('The chosen algorithm is Weighted Majority')
    print('The first model is', first_model, 'and has a weight of', round(weight_fm, 2))
    print('The second model is', second_model, 'and has a weight of', round(weight_sm, 2))
    print('The third model is', third_model, 'and has a weight of', round(weight_tm, 2))
    print('The accuracy of the ensemble is', round(accuracy_ensemble, 2))   
    print('The adjusted accuracy of the ensemble is', round(adj_accuracy_ensemble, 2))
        
if pred_dat_output[0] == 'unknown':
    unknown_predictions = pandas.DataFrame(blind_predictions, columns = ['Prediction'])
    prediction_input = prediction_data.drop(output, axis = 1)
    full_pred = prediction_input.join(unknown_predictions)

print(full_pred)
