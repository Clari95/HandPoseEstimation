function [ u, v ,d ] = xyz2uvd( x,y,z )
%XYZ2UVD Summary of this function goes here
%   Detailed explanation goes here
    u0 = 160;    v0 = 120;
    fx = 240.99;    fy = 240.99;
    u = x*fx./z + u0; 
    v = y*fy./z + v0;
    d = z;

end

