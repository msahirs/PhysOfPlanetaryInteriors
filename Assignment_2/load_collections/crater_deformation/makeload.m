% Make topo file for benchmark

clear all; close all; clc

% Load X,Y,coordinates of the centroids of the elements in the FE model
load elements.txt
no = elements(:,1);
x = elements(:,2)/1000; % x-coordinate in [km]
y = elements(:,3)/1000; % y-coordinate in [km]
figure;
scatter(x,y,10);
xlabel('x-axis [km]','FontSize',20)
ylabel('y-axis [km]','FontSize',20)
set(gca,'FontSize',20)

% Parabolic ice cap 
hb       = 2000;    % maximum ice thickness [m]
rho_ice = 3500;      % density of ice [kg/m3]
g       = 1.7972;   % surface gravity [m/s]
base    = 1000;     % diameter of crater [km]
min_ice = 0;
z = - hb*(x/base).^2 - hb*(y/base).^2 + hb;
z(z<min_ice) = min_ice;

figure(30); scatter(x,y,10,z); colorbar

% Save file
% Element number, load type, magnitude
% Write file
fid = fopen('load','w'); 
for ii = 1:length(x)
    fprintf(fid,'%6i, P2,%12.3E\n',no(ii),z(ii)*g*rho_ice);
end
fclose(fid);









