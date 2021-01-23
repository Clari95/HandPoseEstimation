clear
close all
filename = '/home/clari/Schreibtisch/Projekt/Postprocessing/results.txt'
%'/home/clari/Schreibtisch/Projekt/ICVL/Testing/';

temp =importdata(filename);

GroundTruthXYZ = reshape(temp,[length(temp),3,16]);

h5filename1 = '/home/clari/Schreibtisch/Projekt/ICVL/pointdata/Testing/ICVLTestPointCloud_seq1_1.h5';
h5filename2= '/home/clari/Schreibtisch/Projekt/ICVL/pointdata/Testing/ICVLTestPointCloud_seq2_1.h5';


h5read1 = h5read(h5filename1,'/pointCloud'); %,[3,4000,batchIndex]); 
h5read2 = h5read(h5filename2, '/pointCloud');

h5read = h5read1;
h5read(:,:,703:1596)= h5read2(:,:,:);
%h5read =[h5read1; h5read1];
   N=1596;
   
 scatter3(h5read(1,:,N), h5read(2,:,N), h5read(3,:,N), 5,'filled','b');
 hold on;
 scatter3(GroundTruthXYZ(N,1,:), GroundTruthXYZ(N,2,:), GroundTruthXYZ(N,3,:), 80,'filled','r'); %visualize the ground truth pose in red size 80
 view([0,0,-1])
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,1), GroundTruthXYZ(N,1,2)], [GroundTruthXYZ(N,2,1),GroundTruthXYZ(N,2,2)], [GroundTruthXYZ(N,3,1), GroundTruthXYZ(N,3,2)], 'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,2), GroundTruthXYZ(N,1,3)], [GroundTruthXYZ(N,2,2),GroundTruthXYZ(N,2,3)], [GroundTruthXYZ(N,3,2), GroundTruthXYZ(N,3,3)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,3), GroundTruthXYZ(N,1,4)], [GroundTruthXYZ(N,2,3),GroundTruthXYZ(N,2,4)], [GroundTruthXYZ(N,3,3), GroundTruthXYZ(N,3,4)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,1), GroundTruthXYZ(N,1,5)], [GroundTruthXYZ(N,2,1),GroundTruthXYZ(N,2,5)], [GroundTruthXYZ(N,3,1), GroundTruthXYZ(N,3,5)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,5), GroundTruthXYZ(N,1,6)], [GroundTruthXYZ(N,2,5),GroundTruthXYZ(N,2,6)], [GroundTruthXYZ(N,3,5), GroundTruthXYZ(N,3,6)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,6), GroundTruthXYZ(N,1,7)], [GroundTruthXYZ(N,2,6),GroundTruthXYZ(N,2,7)], [GroundTruthXYZ(N,3,6), GroundTruthXYZ(N,3,7)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,1), GroundTruthXYZ(N,1,8)], [GroundTruthXYZ(N,2,1),GroundTruthXYZ(N,2,8)], [GroundTruthXYZ(N,3,1), GroundTruthXYZ(N,3,8)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,8), GroundTruthXYZ(N,1,9)], [GroundTruthXYZ(N,2,8),GroundTruthXYZ(N,2,9)], [GroundTruthXYZ(N,3,8), GroundTruthXYZ(N,3,9)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,9), GroundTruthXYZ(N,1,10)], [GroundTruthXYZ(N,2,9),GroundTruthXYZ(N,2,10)], [GroundTruthXYZ(N,3,9), GroundTruthXYZ(N,3,10)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,1), GroundTruthXYZ(N,1,11)], [GroundTruthXYZ(N,2,1),GroundTruthXYZ(N,2,11)], [GroundTruthXYZ(N,3,1), GroundTruthXYZ(N,3,11)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,11), GroundTruthXYZ(N,1,12)], [GroundTruthXYZ(N,2,11),GroundTruthXYZ(N,2,12)], [GroundTruthXYZ(N,3,11), GroundTruthXYZ(N,3,12)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,12), GroundTruthXYZ(N,1,13)], [GroundTruthXYZ(N,2,12),GroundTruthXYZ(N,2,13)], [GroundTruthXYZ(N,3,12), GroundTruthXYZ(N,3,13)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,1), GroundTruthXYZ(N,1,14)], [GroundTruthXYZ(N,2,1),GroundTruthXYZ(N,2,14)], [GroundTruthXYZ(N,3,1), GroundTruthXYZ(N,3,14)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,14), GroundTruthXYZ(N,1,15)], [GroundTruthXYZ(N,2,14),GroundTruthXYZ(N,2,15)], [GroundTruthXYZ(N,3,14), GroundTruthXYZ(N,3,15)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4
 
 hold on
 plot = plot3([GroundTruthXYZ(N,1,15), GroundTruthXYZ(N,1,16)], [GroundTruthXYZ(N,2,15),GroundTruthXYZ(N,2,16)], [GroundTruthXYZ(N,3,15), GroundTruthXYZ(N,3,16)],'r'); 
 view([0,0,-1])
 plot.LineWidth= 4