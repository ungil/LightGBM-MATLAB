classdef lgbmDataset < handle
    properties
        pointer
    end
   
    methods
        function obj=lgbmDataset(fileOrMatrix,reference,params,type)
            if nargin<2
                ref=libpointer('voidPtr');
            else
                assert(isa(reference,'lgbmDataset'))
                checkPointer(reference)
                ref=reference.pointer;
            end
            if nargin<3
                params='';
            end
            if isa(params,'containers.Map')
                params=parametersString(params);
            end
            if nargin<4
                type='float';
            end
            if ischar(fileOrMatrix)
                file=fileOrMatrix;
                % [int32, cstring, cstring, voidPtr, voidPtrPtr] LGBM_DatasetCreateFromFile(cstring, cstring, voidPtr, voidPtrPtr)
                [ret,~,~,~,dset]=calllib('lib_lightgbm','LGBM_DatasetCreateFromFile',file,params,ref,libpointer('voidPtr'));
                checkError(ret)
                obj.pointer=dset;
            elseif isa(fileOrMatrix,'lib.pointer')
                obj.pointer=fileOrMatrix;
            elseif issparse(fileOrMatrix)
                error('cannot handle sparse matrices')
            elseif ismatrix(fileOrMatrix)
                if strcmp(type,'float')
                    type=0;
                    ptrType='singlePtr';
                elseif strcmp(type,'double')
                    type=1;
                    ptrType='doublePtr';
                else
                    error('type sould be float or double')
                end
                matrix=fileOrMatrix;
                nrow=size(matrix,1);
                ncol=size(matrix,2);
                isrowmajor=0;
                data=reshape(matrix,1,numel(matrix));
                data=libpointer(ptrType,data);
                % [int32, voidPtr, cstring, voidPtr, voidPtrPtr] LGBM_DatasetCreateFromMat(voidPtr, int32, int32, int32, int32, cstring, voidPtr, voidPtrPtr)
                [ret,~,~,~,dset]=calllib('lib_lightgbm','LGBM_DatasetCreateFromMat',data,type,nrow,ncol,isrowmajor,params,ref,libpointer('voidPtr'));
                checkError(ret)
                obj.pointer=dset;
            end
        end
        
        function newobj=emptyCopy(obj,nrows)
            checkPointer(obj)
            % [int32, voidPtr, voidPtrPtr] LGBM_DatasetCreateByReference(voidPtr, int64, voidPtrPtr)
            [ret,~,newdset]=calllib('lib_lightgbm','LGBM_DatasetCreateByReference',obj.pointer,nrows,libpointer('voidPtr'));
            checkError(ret)
            newobj=lgbmDataset(newdset);
        end
        
        function pushrows(obj,rows,startRow,type)
            checkPointer(obj)
            if nargin<4
                type='float';
            end
            if strcmp(type,'float')
                type=0;
            elseif strcmp(type,'double')
                type=1;
            else
                error('type sould be float or double')
            end
            nrow=size(rows,1);
            ncol=size(rows,2);
            data=reshape(rows',1,numel(rows));
            % [int32, voidPtr, voidPtr] LGBM_DatasetPushRows(voidPtr, voidPtr, int32, int32, int32, int32)
            ret=calllib('lib_lightgbm','LGBM_DatasetPushRows',obj.pointer,data,type,nrow,ncol,startRow);
            checkError(ret)
        end
                      
        function newobj=slice(obj,rows,params)
            checkPointer(obj)
            if nargin<3
                params='';
            end
            if isa(params,'containers.Map')
                params=parametersString(params);
            end
            % [int32, voidPtr, int32Ptr, cstring, voidPtrPtr] LGBM_DatasetGetSubset(voidPtr, int32Ptr, int32, cstring, voidPtrPtr)
            [ret,~,~,~,newdset]=calllib('lib_lightgbm','LGBM_DatasetGetSubset',obj.pointer,libpointer('int32Ptr',rows),numel(rows),params,libpointer('voidPtr'));
            checkError(ret)
            newobj=lgbmDataset(newdset);
        end
        
        function free(obj)            
            checkPointer(obj)
            % [int32, voidPtr] LGBM_DatasetFree(voidPtr)
            ret=calllib('lib_lightgbm','LGBM_DatasetFree',obj.pointer);
            checkError(ret)
            obj.pointer=[];
        end
                
        function delete(obj)            
            free(obj)
        end
        
        function size=size(obj,dim)
            checkPointer(obj)
            if nargin==1
                size=[obj.size(1) obj.size(2)];
            else
                if dim==1
                    % [int32, voidPtr, int32Ptr] LGBM_DatasetGetNumData(voidPtr, int32Ptr)
                    [ret,~,size]=calllib('lib_lightgbm','LGBM_DatasetGetNumData',obj.pointer,0);
                    checkError(ret)
                elseif dim==2
                    % [int32, voidPtr, int32Ptr] LGBM_DatasetGetNumFeature(voidPtr, int32Ptr)
                    [ret,~,size]=calllib('lib_lightgbm','LGBM_DatasetGetNumFeature',obj.pointer,0);
                    checkError(ret)
                end
            end
        end
        
        function values=field(obj,field)
            checkPointer(obj)
            if strcmp(field,'label') || strcmp(field,'weight') 
                type=0;
                ptrType='singlePtr';
             elseif strcmp(field,'init_score')
                type=1;
                ptrType='doublePtr';
            elseif strcmp(field,'group') || strcmp(field,'query')
                type=2;
                ptrType='int32Ptr';
            else
                error('field should be one of: label weight init_score group query')
            end
            % [int32, voidPtr, cstring, int32Ptr, voidPtrPtr, int32Ptr] LGBM_DatasetGetField(voidPtr, cstring, int32Ptr, voidPtrPtr, int32Ptr)
            [ret,~,~,len,out]=calllib('lib_lightgbm','LGBM_DatasetGetField',obj.pointer,field,0,libpointer('voidPtr'),type);
            checkError(ret)
            if len==0
                values=[];
            else
                setdatatype(out,ptrType,1,len);
                values=out.Value;
            end
        end

        function setField(obj,field,values)
            checkPointer(obj)
            if strcmp(field,'label') || strcmp(field,'weight') 
                type=0;
                ptrType='singlePtr';
             elseif strcmp(field,'init_score')
                type=1;
                ptrType='doublePtr';
            elseif strcmp(field,'group') || strcmp(field,'query')
                type=2;
                ptrType='int32Ptr';
            else
                error('field should be one of: label weight init_score group query')
            end
            data=libpointer(ptrType,values);
            % [int32, voidPtr, cstring, voidPtr] LGBM_DatasetSetField(voidPtr, cstring, voidPtr, int32, int32)
            ret=calllib('lib_lightgbm','LGBM_DatasetSetField',obj.pointer,field,data,length(values),type);
            checkError(ret)
        end

        function names=featureNames(obj)
            checkPointer(obj)
            maxlen=100;
            n=size(obj,2);
            names=cell(1,n);
            for i=1:length(names)
                names{i}=repmat(' ',1,maxlen);
            end
            % [int32, voidPtr, stringPtrPtr, int32Ptr] LGBM_DatasetGetFeatureNames(voidPtr, stringPtrPtr, int32Ptr)
            [ret,~,names,len]=calllib('lib_lightgbm','LGBM_DatasetGetFeatureNames',obj.pointer,libpointer('stringPtrPtr',names),0);
            checkError(ret)
            assert(len==n)
            names=names';
        end
        
        function setFeatureNames(obj,names)
            checkPointer(obj)
            % [int32, voidPtr, stringPtrPtr] LGBM_DatasetSetFeatureNames(voidPtr, stringPtrPtr, int32)
            ret=calllib('lib_lightgbm','LGBM_DatasetSetFeatureNames',obj.pointer,libpointer('stringPtrPtr',names),28);
            checkError(ret)
        end

        function saveBinary(obj,file)
            checkPointer(obj)
            % [int32, voidPtr, cstring] LGBM_DatasetSaveBinary(voidPtr, cstring)
            ret=calllib('lib_lightgbm','LGBM_DatasetSaveBinary',obj.pointer,file);
            checkError(ret)
        end
        
        function [booster, bestIteration, metrics, metricNames] = train(obj,parameters,numRounds,validationSets,earlyStopping)
            if nargin<5
                earlyStopping=0;
            end
            if nargin<4
                validationSets=[];
            end
            bestIteration=numRounds; % return numRounds if not stopped
            booster=lgbmBooster(obj,parameters);
            for i=1:length(validationSets)
                booster.addValidationData(validationSets(i));
            end
            metricNames=evalNames(booster);
            nmetrics=length(metricNames);
            metrics=nan(nmetrics,length(validationSets)+1,numRounds);
            bestSoFar=nan(1,length(validationSets)*nmetrics);
            countNoImprovement=zeros(1,length(validationSets)*nmetrics);
            for i=1:numRounds
                updateOneIter(booster);
                fprintf('[%4d]', getIteration(booster)-1);
                for j=0:length(validationSets)
                    if j==0
                        name='train';
                    else
                        name=['valid-' num2str(j)];
                    end
                    values=getEval(booster,j);
                    metrics(:,j+1,getIteration(booster)-1)=values;
                    for k=1:nmetrics
                        fprintf('  %s %s %.6f',name,metricNames{k},values(k));
                        if earlyStopping && j>0
                            idx=(j-1)*nmetrics+k;
                            if strcmp(metricNames{k},'auc') || strcmp(metricNames{k},'ndcg')
                                improvement=(values(k)>bestSoFar(idx));
                            else
                                improvement=(values(k)<bestSoFar(idx));
                            end
                            if isnan(bestSoFar(idx)) || improvement
                                bestSoFar(idx)=values(k);
                                countNoImprovement(idx)=0;
                            else
                                countNoImprovement(idx)=countNoImprovement(idx)+1;
                            end
                        end
                    end
                end
                fprintf('\n');
                if max(countNoImprovement)>=earlyStopping
                    bestIteration=getIteration(booster)-1-earlyStopping;
                    break
                end
            end
        end
    end
end

% TODO: [int32, voidPtr, voidPtr, int32Ptr, voidPtr] LGBM_DatasetPushRowsByCSR(voidPtr, voidPtr, int32, int32Ptr, voidPtr, int32, int64, int64, int64, int64)
% TODO: [int32, doublePtrPtr, int32PtrPtr, int32Ptr, cstring, voidPtrPtr] LGBM_DatasetCreateFromSampledColumn(doublePtrPtr, int32PtrPtr, int32, int32Ptr, int32, int32, cstring, voidPtrPtr)
% TODO: [int32, voidPtr, int32Ptr, voidPtr, cstring, voidPtr, voidPtrPtr] LGBM_DatasetCreateFromCSC(voidPtr, int32, int32Ptr, voidPtr, int32, int64, int64, int64, cstring, voidPtr, voidPtrPtr)
% TODO: [int32, voidPtr, int32Ptr, voidPtr, cstring, voidPtr, voidPtrPtr] LGBM_DatasetCreateFromCSR(voidPtr, int32, int32Ptr, voidPtr, int32, int64, int64, int64, cstring, voidPtr, voidPtrPtr)

function checkPointer(obj)
if ~isa(obj.pointer,'lib.pointer')
    error('the object is not associated to a pointer')
end
end

function checkError(ret)
if ret~=0
    error(calllib('lib_lightgbm','LGBM_GetLastError'))
end
end

function str = parametersString(map)
keys=map.keys;
values=map.values;
for i=1:map.length
    if i==1
        str='';
    else
        str=[str ' '];
    end
    str=[str keys{i} '=' num2str(values{i})];
end
end