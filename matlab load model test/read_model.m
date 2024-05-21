
%% Load model
clear;clc; close all
net_alt  = importKerasLayers("model_h5.h5");

%% Create image datastore
I_train = [];
L_train = [];
for scan = ["scan30", "scan31", "scan32"]
    imds_loc = strcat("E:\YKL\Thorlabs VSCAN Labeling\",scan);
    imds = imageDatastore(imds_loc);
    classes = ["a","b","c","d","e"];
    classIDs = [1,2,3,4,5];
    pxds_loc = strcat("E:\YKL\Thorlabs VSCAN Labeling\",scan,"\LabelingProject\GroundTruthProject\PixelLabelData")
    pxds = pixelLabelDatastore(pxds_loc,classes,classIDs);

    % form training image set
    
    for k=1:size(imds.Files,1)
        I = im2gray(imread(char(imds.Files(k))));
        I = histeq(imresize(I,[352 400],"nearest"));
        % I = imresize(I,[352 400],"nearest");
        I(I>255) = 255;
        I(I<0) = 0;
        I_denoised = imnlmfilt(I,"ComparisonWindowSize",7,"SearchWindowSize",21,"DegreeOfSmoothing",3);
        %I_denoised = I;
        I_train = cat(3,I_train,I_denoised);
    end
    clear k
    
    % form ground truth set
    for k=1:size(imds.Files,1)
        L = imread(char(pxds.Files(k)));
        L = imresize(L,[352 400],"nearest");
        L(L==5) = 4;
        L(L==0) = 5;
        L = onehotencode_v2(L);
        L_train = cat(4,L_train,L);
    end
    clear k
end
%%
% L_train = gpuArray(L_train);
% I_train = gpuArray(I_train);

I_train = reshape(I_train, [352 400 1 size(I_train,3)]);

shuffled_idx = randperm(size(I_train, 4));
I_train_2 = I_train(:,:,:,shuffled_idx);
L_train_2 = L_train(:,:,:,shuffled_idx);

partition_idx = round(0.8*size(I_train, 4));

I_train_train = I_train_2(:,:,:,1:partition_idx);
I_valid = I_train_2(:,:,:,partition_idx+1:end);
L_train_train = L_train_2(:,:,:,1:partition_idx);
L_valid = L_train_2(:,:,:,partition_idx+1:end);

%%
options = trainingOptions('adam', ...
    MaxEpochs=50, ...
    MiniBatchSize=3, ...
    Shuffle='every-epoch', ...
    InitialLearnRate=1e-3, ...
    LearnRateSchedule='none', ...
    GradientDecayFactor=0, ...
    SquaredGradientDecayFactor=0.999, ...
    Epsilon=1e-7, ... % end of Adam properties
    L2Regularization=0.01, ... % no weight decay
    SequenceLength='longest', ...
    SequencePaddingDirection='left', ... 
    SequencePaddingValue=0, ... % end of sequence properties
    ValidationData={I_valid,L_valid},...
    Verbose=true, ...
    Plots='training-progress');

net_alt_trained = trainNetwork(I_train_train,L_train_train,net_alt,options);

save net_alt_trained
%% Read image
tic
I_test = im2gray(imread("E:\YKL\Thorlabs VSCAN Labeling\scan27\VSCAN_0027_190.png"));
I_test = histeq(imresize(I_test,[352 400],"nearest"));
I_test(I_test>255) = 255;
I_test(I_test<0) = 0;
I_denoised = imnlmfilt(I_test,"ComparisonWindowSize",7,"SearchWindowSize",21,"DegreeOfSmoothing",7);
imshow(I_denoised)
toc

pred = round(predict(net_alt_trained,I_test));

seg = one_hot_to_seg(pred,5);

imwrite(seg,"test2.png")
figure()
imshow(seg)

%% Read train image
tic
I_test = im2gray(imread("E:\YKL\Thorlabs VSCAN Labeling\scan30\VSCAN_0030_150.png"));
I_test = histeq(imresize(I_test,[352 400],"nearest"));
I_test(I_test>255) = 255;
I_test(I_test<0) = 0;
I_denoised = imnlmfilt(I_test,"ComparisonWindowSize",7,"SearchWindowSize",21,"DegreeOfSmoothing",7);
figure()
imshow(I_denoised)
toc

pred = round(predict(net_alt_trained,I_test));

seg = one_hot_to_seg(pred,5);
figure()
imshow(seg)

%imwrite(seg,"test2.png")

%%

imwrite(pred(:,:,1)*255,"pred_ch1.png");imwrite(pred(:,:,2)*255,"pred_ch2.png");imwrite(pred(:,:,3)*255,"pred_ch3.png");imwrite(pred(:,:,4)*255,"pred_ch4.png");

%TODO: look at prediction result

%%
function ohe_img = onehotencode_v2(img)
    ohe_img = [];
    num_classes = 5;
    for i=1:num_classes
        img_i = img;
        img_i(img_i~=i) = 0;
        img_i(img_i==i) = 1;
        ohe_img = cat(3,ohe_img,img_i);
    end
end

%%
function result = one_hot_to_seg(one_hot_img, num_lbls)
    result = zeros([size(one_hot_img,1) size(one_hot_img,2) 3]);
    for label=1:num_lbls-1
        ch = one_hot_img(:,:,label);
        ch = repmat(ch,1,1,3);
        if label==4
            ch = ch*0;
        elseif label==1
            ch(:,:,1) = ch(:,:,1)*255;
            ch(:,:,2) = ch(:,:,2)*0;
            ch(:,:,3) = ch(:,:,3)*0;
        elseif label==2
            ch(:,:,1) = ch(:,:,1)*0;
            ch(:,:,2) = ch(:,:,2)*255;
            ch(:,:,3) = ch(:,:,3)*0;
        elseif label==3
            ch(:,:,1) = ch(:,:,1)*0;
            ch(:,:,2) = ch(:,:,2)*0;
            ch(:,:,3) = ch(:,:,3)*255;
        end
        result = result + ch;
    end
end
