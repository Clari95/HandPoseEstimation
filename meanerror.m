clear;
close all;
filenameresult = '/home/clari/Schreibtisch/Projekt/Postprocessing/results.txt';
%'/home/clari/Schreibtisch/Projekt/ICVL/Testing/';
temp =importdata(filenameresult);
%results=temp.data;
predicted = reshape(temp,[length(temp),3,16]);


txtFilePath = '/home/clari/Schreibtisch/Projekt/ICVL/Testing';
temp1 = importdata([txtFilePath, '/test_seq_1.txt']);
%imageFileNames = temp1.textdata;
GroundTruthUVD1 = temp1.data;
pointsseq1 = reshape(GroundTruthUVD1,[length(GroundTruthUVD1),3,16]);

txtFilePath2 = '/home/clari/Schreibtisch/Projekt/ICVL/Testing';
temp2 = importdata([txtFilePath2, '/test_seq_2.txt']);
%imageFileNames = temp2.textdata;
GroundTruthUVD2 = temp2.data;
pointsseq2 = reshape(GroundTruthUVD2,[length(GroundTruthUVD2),3,16]);


h5filename1 = '/home/clari/Schreibtisch/Projekt/ICVL/pointdata/Testing/ICVLTestPointCloud_seq1_1.h5';
h5filename2= '/home/clari/Schreibtisch/Projekt/ICVL/pointdata/Testing/ICVLTestPointCloud_seq2_1.h5';


h5read1 = h5read(h5filename1,'/joint'); %,[3,4000,batchIndex]); 
h5read2 = h5read(h5filename2, '/joint');

h5read1 = reshape(h5read1,[3,16,length(h5read1)]);
h5read2 = reshape(h5read2,[3,16,length(h5read2)]);

testseq = h5read1;
testseq(:,:,703:1596)= h5read2(:,:,:);


predicted = permute(predicted, [2, 3, 1]);
loss = (testseq - predicted);
 
%Eucledian Loss
% sqrt( (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)

lossquad=((testseq - predicted).^2);
add= squeeze(lossquad(1,:,:) + lossquad(2,:,:) + lossquad(3,:,:));
eucloss=(sqrt(add));
euclossmean= mean(eucloss,2);
overallloss = mean(euclossmean)

maxeucloss= max(eucloss(:));
mineucloss= min(eucloss(:));


J=10;
figure;
x=[0:1:15];
names={'Palm', 'Thumb root', 'Thumb mid', 'Thumb tip', 'Index root', 'Index mid', 'Index tip','Middle root', 'Middle mid', 'Middle tip', 'Ring root', 'Ring mid', 'Ring tip', 'Pinky root', 'Pinky mid', 'Pinky tip' };
bar(x, eucloss(:,J)) %, 'BarWidth', 0.5)   %number is example 1:1596
ylim([0 0.05])
xticks([0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15])
xtickangle(90)
set(gca,'xticklabel',names)
title(['eucloss hand ' num2str(J)])


figure;
x=[0:1:15];
bar(x,euclossmean)
names={'Palm', 'Thumb root', 'Thumb mid', 'Thumb tip', 'Index root', 'Index mid', 'Index tip','Middle root', 'Middle mid', 'Middle tip', 'Ring root', 'Ring mid', 'Ring tip', 'Pinky root', 'Pinky mid', 'Pinky tip' };
ylim([0 0.05])
xticks([0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15])
xtickangle(90)
set(gca,'xticklabel',names)
title('euclossmean')


 
 
 %Palm, Thumb root, Thumb mid, Thumb tip, Index root, Index mid, Index tip, Middle root, Middle mid, 
 %Middle tip, Ring root, Ring mid, Ring tip, Pinky root, Pinky mid, Pinky tip.
 