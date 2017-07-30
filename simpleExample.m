lgbmLoad

tmptrain=dlmread(fullfile(lgbmPath,'examples','regression','regression.train'));
labeltrain=tmptrain(:,1);
matrixtrain=tmptrain(:,2:end);

tmptest=dlmread(fullfile(lgbmPath,'examples','regression','regression.test'));
labeltest=tmptest(:,1);
matrixtest=tmptest(:,2:end);

dstrain=lgbmDataset(matrixtrain);
setField(dstrain,'label',labeltrain);

dstest=lgbmDataset(matrixtest,dstrain);
setField(dstest,'label',labeltest);

% % datasets can also be read directly from datafiles:
% dstrain=lgbmDataset(fullfile(lgbmPath,'examples','regression','regression.train'));
% dstest=lgbmDataset(fullfile(lgbmPath,'examples','regression','regression.test'),dstrain);

parameters=containers.Map;
parameters('task')='train';
parameters('boosting_type')='gbdt';
parameters('objective')='regression';
parameters('metric')='l2,auc';
parameters('num_leaves')=31;
parameters('learning_rate')=0.05;
parameters('feature_fraction')=0.9;
parameters('bagging_fraction')=0.8;
parameters('bagging_freq')=5;
parameters('num_threads')=1;
parameters('verbose')=0;

[booster, bestIteration, metrics, metricNames]=train(dstrain,parameters,20,dstest,2);

for i=1:size(metrics,1)
    subplot(1,size(metrics,1),i)
    plot(squeeze(metrics(i,:,:))')
    title(metricNames(i))
end

disp(['bestIteration: ' num2str(bestIteration)])

booster.saveModel('model.txt',bestIteration);

disp('model.txt saved')

disp(['rmse of prediction: ' num2str(sqrt(mean((booster.predictMatrix(matrixtest,bestIteration)-labeltest).^2)))])

featureNames=booster.featureNames();

disp('feature names')
disp(featureNames)

importance=booster.importance('split',bestIteration);

disp('feature importance') 
disp(importance)

lgbmUnload

% adapted from LightGBM/examples/python-guide/simple_example.py
% python output, setting 'num_threads'=1 and early_stopping_rounds=2

% Load data...
% Start training...
% [1]  valid_0's l2: 0.242963  valid_0's auc: 0.755797
% Training until validation scores don't improve for 2 rounds.
% [2]  valid_0's l2: 0.239442  valid_0's auc: 0.755071
% [3]  valid_0's l2: 0.235933  valid_0's auc: 0.777275
% [4]  valid_0's l2: 0.231946  valid_0's auc: 0.778678
% [5]  valid_0's l2: 0.228124  valid_0's auc: 0.78183
% [6]  valid_0's l2: 0.225515  valid_0's auc: 0.781387
% [7]  valid_0's l2: 0.222806  valid_0's auc: 0.789441
% [8]  valid_0's l2: 0.220129  valid_0's auc: 0.788982
% [9]  valid_0's l2: 0.217686  valid_0's auc: 0.789764
% [10] valid_0's l2: 0.215047  valid_0's auc: 0.791578
% [11] valid_0's l2: 0.212324  valid_0's auc: 0.796077
% [12] valid_0's l2: 0.210524  valid_0's auc: 0.795948
% [13] valid_0's l2: 0.208313  valid_0's auc: 0.797713
% [14] valid_0's l2: 0.206929  valid_0's auc: 0.796988
% [15] valid_0's l2: 0.205516  valid_0's auc: 0.797157
% Early stopping, best iteration is:
% [13] valid_0's l2: 0.208313  valid_0's auc: 0.797713
% Save model...
% Start predicting...
% The rmse of prediction is: 0.456413493233
% Dump model to JSON...
% Feature names: ['Column_0', 'Column_1', 'Column_2', 'Column_3', 'Column_4', 'Column_5', 'Column_6', 'Column_7', 'Column_8', 
% 'Column_9', 'Column_10', 'Column_11', 'Column_12', 'Column_13', 'Column_14', 'Column_15', 'Column_16', 'Column_17', 'Column_18',
% 'Column_19', 'Column_20', 'Column_21', 'Column_22', 'Column_23', 'Column_24', 'Column_25', 'Column_26', 'Column_27']
% Calculate feature importances...
% Feature importances: [9, 6, 3, 21, 1, 41, 6, 1, 3, 16, 8, 3, 0, 8, 6, 4, 1, 9, 2, 4, 0, 13, 49, 3, 39, 57, 35, 42]
