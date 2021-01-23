
%% load data
txtFilePath = 'C:/Users/Clarissa/Desktop/Projekt/ICVL/Training/';
temp = importdata([txtFilePath, 'labels.txt']);

imageFileNames = temp.textdata;
GroundTruthUVDs = temp.data;
GroundTruthUVDs = reshape(GroundTruthUVDs,[length(GroundTruthUVDs),3,16]); %331006samples x 3dimension x 16joints


%% Define HDF 5 file and container matrices for pointcloud and gtposes.
h5fileSize = 30000;  %store every 30000 samples into one h5 file.
pointClouds = zeros(3,4000,h5fileSize);  %container for point clouds, 4000 randomly sampled points for each image.
joints=zeros(48,h5fileSize);  %48 coordinates values to be predicted (3x16) red points
center3ds = zeros(3,h5fileSize);  %centroid coordinate for mean subtraction and also revoery of the original pose during test.

tic; %start the timer
for ind=129175:129175%331006 should it be, now just test with one sample
    
imageFilename = [txtFilePath,'Depth/' ,char(imageFileNames(ind))];
if isfile(imageFilename)       %(ind~=129174)
    % signal every 100th frame
    if mod(ind,1000)==0 % plot the processing time used for last 1000 images
        disp(ind); toc; tic;
    end
    

    %% get the groudtruth pose's uvd and convert to xyz;
    GroundTruthUVD = squeeze(GroundTruthUVDs(ind,:,:));  % get this images groundtruth pose's uvds, squeeze eliminates useless dimension (1,3,16)-->(3,16)
    GroundTruthUVD = reshape(GroundTruthUVD,[3,16]);
    [tempX, tempY, tempZ] = uvd2xyz(GroundTruthUVD(1,:),GroundTruthUVD(2,:),GroundTruthUVD(3,:)); %uvd to xyz conversion for groundtruth pose 
    GroundTruthXYZ = [tempX; tempY; tempZ];
    
    %% get depth image and read into double
    imageFilename = [txtFilePath,'Depth/' ,char(imageFileNames(ind))]; %get current image file name
    image = double(imread(imageFilename));  %convert from integer to double
    
    %% convert depth iamge to point clouds
    allUs = ones(240,1)*(1:320);
    allVs = (1:240)' * ones(1,320);
    allDs = image; 
    validMask = (allDs<30000); % mask for the valid points
    allUs = allUs(validMask);     allVs = allVs(validMask);     allDs = allDs(validMask);
    [allX, allY, allZ] = uvd2xyz(allUs(:), allVs(:), allDs(:)); % allUs(:) means vecotrize a matrix [320,240]-->[76800]
    pointCloud = [allX, allY, allZ]';
    
    pointCloud = pointCloud/1000;  %mm to m conversion
    GroundTruthXYZ = GroundTruthXYZ/1000; %mm to m conversion
    
    %% visualize before mean subtraction (only for debug, during runtime comment this section)
   %
   clf;

    scatter3(pointCloud(1,:), pointCloud(2,:), pointCloud(3,:), 5,'filled','b'); %visualize the point cloud in blue, size 5
    hold on;
    scatter3(GroundTruthXYZ(1,:), GroundTruthXYZ(2,:), GroundTruthXYZ(3,:), 80,'filled','r'); %visualize the ground truth pose in red size 80
    view([0,0,-1])
    
    %% get center of mass and subtract pointCloud and GroundTruthXYZ with it
    COM = mean(pointCloud,2);
    
    T=COM*ones(1,length(pointCloud));
    pointCloud = pointCloud - COM*ones(1,length(pointCloud));
    GroundTruthXYZ = GroundTruthXYZ - COM*ones(1,length(GroundTruthXYZ));

    %% visualize after mean subtraction (only for debug, during runtime comment this section)
    %
    %clf;
%     figure;
%     scatter3(pointCloud(1,:), pointCloud(2,:), pointCloud(3,:), 5,'filled','b'); %visualize the point cloud in blue, size 5
%     hold on;
%     scatter3(GroundTruthXYZ(1,:), GroundTruthXYZ(2,:), GroundTruthXYZ(3,:), 80,'filled','r'); %visualize the ground truth pose in red size 80
%     view([0,0,-1])
%     set(gca,'XLim',[-0.15 0.15],'YLim',[-0.15 0.15],'ZLim',[-0.15 0.15])
%     %
    %% randomly sample 4000 points
    if (length(pointCloud)<3)
        pointCloud = zeros(3,4000);
    end
    randIndices = randperm(length(pointCloud));
    if (length(pointCloud)<4000)
        randIndices = repmat(randIndices, 1, ceil(4000/length(pointCloud)));
    end
   
    pointCloud = pointCloud(:, randIndices(1:4000));  %takes 4000 radom colums of pointCloud
  
    %% Write HDF5 file    
    batchIndex = ind-floor((ind-1)/h5fileSize)*h5fileSize;  %get the index for the current hdf5 file [1:30000] from the global ind (1:331006)
    pointClouds(:,:,batchIndex) = pointCloud;    % pointClouds
    joints(:,batchIndex)= (reshape(GroundTruthXYZ,[48,1]));
    center3ds(:,batchIndex) = COM; 
    
    if (mod(ind,h5fileSize)==0 || ind==331006)
        toc
        savePath = 'C:/Users/Clarissa/Desktop/Projekt/ICVL/pointdata/';
        h5filename = [savePath,'ICVLTrainPointCloud_',num2str(ceil(ind/h5fileSize)),'.h5'];
        h5filename
        delete(h5filename);
        h5create(h5filename,'/pointCloud',[3,4000,batchIndex],'Datatype','single');
        h5write(h5filename,'/pointCloud', pointClouds(:,:,1:batchIndex));
        h5create(h5filename,'/joint',[48,batchIndex],'Datatype','single');
        h5write(h5filename,'/joint',joints(:,1:batchIndex));
        h5create(h5filename,'/center3d',[3,batchIndex],'Datatype','single');
        h5write(h5filename,'/center3d',center3ds(:,1:batchIndex));
      
        pointClouds=zeros(3,4000,h5fileSize);
        joints=zeros(48,h5fileSize);
        center3ds = zeros(3,h5fileSize);
        tic
    end
end
end