%The depth_projection and bounding_box functions and loading data code are from an implementation written
%by Chen Chen, Kui Liu, and Nasser Kehtarnavaz. The rest is my work.

%script can be called by typing DMM_HOG on the Matlab command line.
file_dir = 'MSR-Action3D/';
frame_remove = 5;  
ratio = 1/2;        


dim = 30000; %limiting each projection to 100 x 100 resloution.

data =['a02', 'a05', 'a01', 'a10', 'a13', 'a18', 'a11'];
TotalNum = 7*10*3;
TotalFeature = zeros(dim,TotalNum);

% Depth Motion Maps generation%
subject_ind = cell(1,NumAct);
OneActionSample = zeros(1,NumAct);

%Stores the DMM-HOG. One row per video sequence
DMM_HOG_front = zeros(226,4356);
DMM_HOG_side = zeros(226,4356);
DMM_HOG_top = zeros(226,4356);

classes = zeros(226);
counter = 1;

for i = 1:NumAct
    action = data((i-1)*3+1:i*3);
    action_dir = strcat(file_dir,action,'/');
    fpath = fullfile(action_dir, '*.mat');
    depth_dir = dir(fpath);
    
    % temporarily store the DMMs projections.
    temp_dmms_front = zeros(100,100,length(depth_dir));
    temp_dmms_side = zeros(100,100,length(depth_dir));
    temp_dmms_top = zeros(100,100,length(depth_dir));
    
    ind = zeros(1,length(depth_dir));
    for j = 1:length(depth_dir)
        depth_name = depth_dir(j).name;
        sub_num = str2double(depth_name(6:7));
        ind(j) = sub_num;
        load(strcat(action_dir,depth_name));
        depth = depth(:,:,frame_remove+1:end-frame_remove);
        % get the bounding box of the normalized projections.
        [front, side, top] = depth_projection(depth);
        
        %resize so we have consistent DMMs and HoG features.
        front = imresize(front, [100 100]);
        side = imresize (side, [100 100]);
        top = imresize (top, [100 100]);
        
        % extract the HoG features
        DMM_HOG_front(counter,:) =  extractHOGFeatures(front);
        DMM_HOG_side(counter,:) =  extractHOGFeatures(side);
        DMM_HOG_top(counter,:) =  extractHOGFeatures(top);
        classes(counter) = j;
        counter = counter + 1;
    end
   
    OneActionSample(i) = length(depth_dir);
    subject_ind{i} = ind;
end
TotalFeature = TotalFeature(:,1:sum(OneActionSample));


% Test and training data generated randomly each run of the script

total_trial = 50;

F_train_size = zeros(1,NumAct);
F_test_size  = zeros(1,NumAct);
HOGs_train_all= zeros(size(DMM_HOG_front,1),size(DMM_HOG_front,2));
HOGs_test_all= zeros(size(DMM_HOG_front,1),size(DMM_HOG_front,2));
train_class = zeros(1,size(DMM_HOG_front,1));
test_class = zeros(1,size(DMM_HOG_front,1));

    count = 0;
    for i = 1:NumAct 
        ID = subject_ind{i};
        F = TotalFeature(:,count+1:count+OneActionSample(i));
        HOG_train_front =DMM_HOG_front(count+1:count+OneActionSample(i), :);
        HOG_train_side = DMM_HOG_side(count+1:count+OneActionSample(i), :);
        HOG_train_top = DMM_HOG_top(count+1:count+OneActionSample(i), :);
        
        index = IND{trial,i};
        train_index = index(1:ceil(length(index)*ratio));
        for k = 1:length(train_index)
            ID(ID==train_index(k)) = 0;
        end
        
        HOGs = HOG_train_front(ID==0,:) + HOG_train_side(ID==0,:) + HOG_train_top(ID==0,:);
        HOGs_t = HOG_train_front(ID>0,:) + HOG_train_side(ID>0,:) + HOG_train_top(ID>0,:);
        HOGs_train_all(count+1:count+size(HOGs,1), :) = HOGs;
        HOGs_test_all(count+1:count+size(HOGs_t,1), :) = HOGs_t;
        classtrain= zeros(1, size(HOGs,1)); %for this particular class
        classtest = zeros(1, size(HOGs_t, 1));
        
        for k1 = 1: size(HOGs,1)
            classtrain(k1)= i;
        end
        
        for k2 = 1: size(HOGs_t,1)
            classtest(k2)= i;
        end
        
        train_class(1, count+1:count+size(HOGs,1)) = classtrain; %add those examples to the rest of the data
        test_class(1, count+1:count+size(HOGs_t,1)) = classtest;
        F_train_size(i) = sum(ID==0);
        F_test_size(i)  = size(F,2) - F_train_size(i);
        
        count = count + OneActionSample(i);
    end
    
    %remove zeros from the sets. There are zeros because we don't know
    %beforehand how many training and testing samples will be picked as its
    %random.
     HOGs_test_all(~any(HOGs_test_all,2), : ) = [];
     HOGs_train_all(~any(HOGs_train_all,2), : ) = [];
     train_class(~any(train_class,1)) = [];
     test_class(~any(test_class,1)) = [];

     
     disp('Classifying query sequence using DMM-HoG and KNN. To use SVMs you need to uncomment lines 144-147')
     accuracy = 0;
     predictions = zeros(size(HOGs_test_all,1),1);
     
     %for each test sequence, compare it will all the training sequences in
     %the library
     for i = 1:size(HOGs_test_all,1)
         votes = zeros(7,1);
         for j=1 : size(HOGs_train_all,1)
             [A,B,r] = canoncorr(HOGs_test_all(i,:)', HOGs_train_all(j,:)'); %compute the canonical correlation
             votes(train_class(j)) = votes(train_class(j)) + sum (r);
         end
         [c, ind] = max(votes);
         if (ind == test_class(1,i))
             accuracy= accuracy +1;
         end
         predictions(i,1) = ind;
     end
  disp('accuracy')
  accuracy/size(HOGs_test_all,1)
  confusionmat(test_class',predictions)
  
%uncomment the following for SVM classification
%   temptrain= train_class';
%   temptest=test_class';
%   mod = svmtrain(temptrain,HOGs_train_all);
%   [predicted, acc, prob_est] = svmpredict(temptest, HOGs_test_all, mod);




