load net_alt_trained.mat
%%
tic
I_test = im2gray(imread("E:\YKL\Thorlabs VSCAN Labeling\scan27\VSCAN_0027_190.png"));
%I_test = histeq(imresize(I_test,[352 400],"nearest"));
I_test = imresize(I_test,[352 400],"nearest");
I_test(I_test>255) = 255;
I_test(I_test<0) = 0;
I_denoised = imnlmfilt(I_test,"ComparisonWindowSize",7,"SearchWindowSize",21,"DegreeOfSmoothing",3);
imshow(I_denoised)
toc

%%
pred2 = round(predict(net_alt_trained,I_denoised));
seg = one_hot_to_seg(pred2,5);
imwrite(seg,"test4.png");
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
