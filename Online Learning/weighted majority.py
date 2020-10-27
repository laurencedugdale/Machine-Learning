############################# Section 7

for stream_input, stream_output in zip(numpy.array(test_ensemble_input), numpy.array(test_ensemble_output)):
    online(numpy.reshape(stream_input, [1, -1]), stream_output, weight_fm, weight_sm, weight_tm,
           correct_predictions, incorrect_predictions)

accuracy_ensemble = correct_predictions / (correct_predictions + incorrect_predictions)
adj_accuracy_ensemble = (correct_predictions / (correct_predictions + incorrect_predictions)) / (1 / adj_acc_base) 

print('The chosen algorithm is Weighted Majority')
print('The first model is', first_model, 'and has a weight of', round(weight_fm, 2))
print('The second model is', second_model, 'and has a weight of', round(weight_sm, 2))
print('The third model is', third_model, 'and has a weight of', round(weight_tm, 2))
print('The accuracy of the ensemble is', round(accuracy_ensemble, 2))
print('The adjusted accuracy of the ensemble is', round(adj_accuracy_ensemble, 2))
