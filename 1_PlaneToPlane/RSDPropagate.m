clear all;
close all;

lambda = 500e-9; %wavelength of green light (arbitrary choice)
k = 2*pi/lambda;
h = lambda/4; % discretization dx in the x-axis, quarter wavelength to avoid aliasing

%%
% Z-axis is direction of propagation
z_a = 0; % z-location of point source
z_b = 0.025; % an arbitrary midpoint we will use as an intermediate point
z_c = 0.05; % location of the final plane

b_width = 0.2; % plane at Z_B
c_width = 0.004; % 2mm plane at Z_C

% discretize planes b and c
x_b = -b_width/2:h:b_width/2;
x_c = -c_width/2:h:c_width/2;

%% CASE 1:
% for a baseline answer, first, propogate a point source at Z_A 
% directly to a plane 10cm away at Z=C.
r = sqrt(z_c^2 + x_c.^2);
EC1 = exp(1i*k*r) .* (z_c ./ r) ./ r*h;
plot_E_field_and_intensity(x_c, EC1, [num2str(z_c*100), 'cm']);
save('EC1.mat', 'EC1', '-v7');

%% CASE 2:
% first propagate A->B
z_dist = z_b-z_a;
z_dist_sqr = z_dist^2;
r = sqrt(z_dist_sqr + x_b.^2);
EB = exp(1i*k*r) ./ r .* (z_dist ./ r) * h;
plot_E_field_and_intensity(x_b, EB, [num2str(z_b*100), 'cm']);

% next propagate B->C
tic;
EC2 = zeros(1, numel(x_c));
z_dist = (z_c-z_b);
z_dist_sqr = z_dist^2;

% for each point on b, propagate it to every point on c
for n=1:numel(x_b)
    x_cur = x_b(n);    
    r = sqrt(z_dist_sqr + (x_c-x_cur).^2);
    EC2 = EC2 + EB(n) .* exp(1i*k*r)./r .* (z_dist./r) *h;
end

% another, equivalent way to do it:
% each point on c, is the sum of contributions from every point on b
% for n=1:numel(zc_x)
%     x_cur = zc_x(n);
%     r = sqrt(z_dist_sqr + (zb_x-x_cur).^2);
%     EC(n) = sum(EB .* exp(1i*k*r) ./ r .* (z_dist./r) * h);
% end

toc

save('EC2.mat', 'EC2', '-v7');

plot_E_field_and_intensity(x_c, EC2, [num2str(z_c*100), 'cm']);

function plot_E_field_and_intensity(x, E, where_str)
    %% plot the result
    En = E/max(E); % normalized E
    figure; 
    subplot(1,2,1)
    plot_E_field(x, En, where_str);
    subplot(1,2,2)
    plot_intensity(x, En, where_str);
end

function plot_E_field_and_intensity2(x, E, where_str)
    %% plot the result
    En = E/max(E); % normalized E
    figure; 
    subplot(1,3,1)
    plot_E_field_component(x, real(En), where_str, 'Real');
    subplot(1,3,2)
    plot_E_field_component(x, imag(En), where_str, 'Imag');
    subplot(1,3,3)
    plot_intensity(x, En, where_str);
end

function plot_E_field_component(x, E, where_str, comp_str)
    hold on;
    plot(x.*1000,E);
    hold off;
    title([comp_str, ' part of E-Field at ', where_str], 'interpreter', 'latex'); 
    xlabel('x position [millimeters]', 'interpreter', 'latex'); 
    ylabel('Normalized E-Field', 'interpreter', 'latex'); 
end


function plot_E_field(x, E, where_str)
    hold on;
    plot(x.*1000,real(E));
    plot(x.*1000,imag(E));
    hold off;
    title(['E-Field at ', where_str], 'interpreter', 'latex'); 
    xlabel('x position [millimeters]', 'interpreter', 'latex'); 
    ylabel('Normalized E-Field', 'interpreter', 'latex'); 
    legend({'Real Part', 'Imaginary Part'}, 'interpreter', 'latex');
end

function plot_intensity(x, E, where_str)
    plot(x.*1000,abs(E)); 
    title(['Intensity at ', where_str], 'interpreter', 'latex'); 
    xlabel('x position [millimeters]', 'interpreter', 'latex'); 
    ylabel('Normalized Intensity', 'interpreter', 'latex') %plot intensity
end
