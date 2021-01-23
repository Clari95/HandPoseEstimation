function [ x,y,z ] = uvd2xyz( u,v,d )
%UVD2XYZ Summary of this function goes here
%   Detailed explanation goes here
    u0 = 160;    v0 = 120;
    fx = 240.99;    fy = 240.99;
    x = ((u-0.5-u0)).*d./fx;
    y = ((v-0.5-v0)).*d./fy;
    z = d;
end
