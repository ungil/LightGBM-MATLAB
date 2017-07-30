w=whos;

lgbmBoosters={w(strcmp({w.class}.','lgbmBooster')).name};
if ~isempty(lgbmBoosters)
    clear(lgbmBoosters{:})
end

lgbmDatasets={w(strcmp({w.class}.','lgbmDataset')).name};
if ~isempty(lgbmDatasets)
    clear(lgbmDatasets{:})
end

clear w
clear lgbmPath
clear lgbmBoosters
clear lgbmDatasets

pause(1)

unloadlibrary lib_lightgbm
