if ispc
    lgbmPath = fullfile(getenv('HOMEDRIVE'),getenv('HOMEPATH'),'LightGBM');
else
    lgbmPath = fullfile(getenv('HOME'),'LightGBM');
end

if not(libisloaded('lib_liblightgbm'))
    loadlibrary(fullfile(lgbmPath,'Release','lib_lightgbm.dll'),'c_api.h')
end

% libfunctions lib_lightgbm -full

% libfunctionsview lib_lightgbm
