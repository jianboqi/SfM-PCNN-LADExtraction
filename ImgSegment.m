function ImgSegment(img_dir)
    % img_dir: path to images
    
    %% Specify data to run net on:
    modelPath = 'TrainedModels';
    modelName = 'net_90_183';
    imglist = getImageList( img_dir, '','jpg');
    segNames = cellfun(@(x)getSegName(x),imglist,'uniformoutput',false);
    
    fprintf('Will run on %d images in folder: %s\n', numel(imglist), img_dir );
    
    %% Load network
    netName = [modelPath filesep modelName];
    a = load(netName, 'net') ;
    net = dagnn.DagNN.loadobj(a.net) ;
    fprintf('Loaded net: %s\n', netName );
    
    %% Run network on images
    minfrac = 0.1;    %Used in merging segments to build leaf estimates
    edgeRadius = 1.5; %Used in evaluation of edge boundary pixels
    doGPU = false;    %if have GPU and compiled MatConvNet with GPU set to true
    redoEdges = false;
    redoSegs = false;
    doAverage = false;
    overwrite = true;
    crop = [];
    averageScales = false;
    imglist = {imglist{1}};
    edgelist = runNetEdges( net, imglist, img_dir, doGPU, overwrite, crop, averageScales );
    
end

%% get segmented file name
function segfilename = getSegName(filepath)
    [filepath,name,ext] = fileparts(filepath);
    segfilename = fullfile(filepath,[name '_seg' ext]);
end