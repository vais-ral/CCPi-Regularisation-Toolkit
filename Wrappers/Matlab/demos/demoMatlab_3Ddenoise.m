% Volume (3D) denoising demo using CCPi-RGL
clear
close all
addpath('../mex_compile/installed');
addpath('../../../data/');

N = 512; 
slices = 30;
vol3D = zeros(N,N,slices, 'single');
Im = double(imread('lena_gray_512.tif'))/255;  % loading image
for i = 1:slices
vol3D(:,:,i) = Im + .05*randn(size(Im)); 
end
vol3D(vol3D < 0) = 0;
figure; imshow(vol3D(:,:,15), [0 1]); title('Noisy image');


lambda_reg = 0.03; % regularsation parameter for all methods
%%
fprintf('Denoise a volume using the ROF-TV model (CPU) \n');
tau_rof = 0.0025; % time-marching constant 
iter_rof = 300; % number of ROF iterations
tic; u_rof = ROF_TV(single(vol3D), lambda_reg, iter_rof, tau_rof); toc; 
figure; imshow(u_rof(:,:,15), [0 1]); title('ROF-TV denoised volume (CPU)');
%%
% fprintf('Denoise a volume using the ROF-TV model (GPU) \n');
% tau_rof = 0.0025; % time-marching constant 
% iter_rof = 300; % number of ROF iterations
% tic; u_rofG = ROF_TV_GPU(single(vol3D), lambda_reg, iter_rof, tau_rof); toc;
% figure; imshow(u_rofG(:,:,15), [0 1]); title('ROF-TV denoised volume (GPU)');
%%
fprintf('Denoise a volume using the FGP-TV model (CPU) \n');
iter_fgp = 300; % number of FGP iterations
epsil_tol =  1.0e-05; % tolerance
tic; u_fgp = FGP_TV(single(vol3D), lambda_reg, iter_fgp, epsil_tol); toc; 
figure; imshow(u_fgp(:,:,15), [0 1]); title('FGP-TV denoised volume (CPU)');
%%
% fprintf('Denoise a volume using the FGP-TV model (GPU) \n');
% iter_fgp = 300; % number of FGP iterations
% epsil_tol =  1.0e-05; % tolerance
% tic; u_fgpG = FGP_TV_GPU(single(vol3D), lambda_reg, iter_fgp, epsil_tol); toc; 
% figure; imshow(u_fgpG(:,:,15), [0 1]); title('FGP-TV denoised volume (GPU)');
%%
fprintf('Denoise a volume using the SB-TV model (CPU) \n');
iter_sb = 150; % number of SB iterations
epsil_tol =  1.0e-05; % tolerance
tic; u_sb = SB_TV(single(vol3D), lambda_reg, iter_sb, epsil_tol); toc; 
figure; imshow(u_sb(:,:,15), [0 1]); title('SB-TV denoised volume (CPU)');
%%
% fprintf('Denoise a volume using the SB-TV model (GPU) \n');
% iter_sb = 150; % number of SB iterations
% epsil_tol =  1.0e-05; % tolerance
% tic; u_sbG = SB_TV_GPU(single(vol3D), lambda_reg, iter_sb, epsil_tol); toc; 
% figure; imshow(u_sbG(:,:,15), [0 1]); title('SB-TV denoised volume (GPU)');
%%

%>>>>>>>>>>>>>> MULTI-CHANNEL priors <<<<<<<<<<<<<<< %
fprintf('Denoise a volume using the FGP-dTV model (CPU) \n');

% create another volume (reference) with slightly less amount of noise
vol3D_ref = zeros(N,N,slices, 'single');
for i = 1:slices
vol3D_ref(:,:,i) = Im + .01*randn(size(Im)); 
end
vol3D_ref(vol3D_ref < 0) = 0;
% vol3D_ref = zeros(size(Im),'single'); % pass zero reference (dTV -> TV)

iter_fgp = 300; % number of FGP iterations
epsil_tol =  1.0e-05; % tolerance
eta =  0.2; % Reference image gradient smoothing constant
tic; u_fgp_dtv = FGP_dTV(single(vol3D), single(vol3D_ref), lambda_reg, iter_fgp, epsil_tol, eta); toc; 
figure; imshow(u_fgp_dtv(:,:,15), [0 1]); title('FGP-dTV denoised volume (CPU)');
%%
fprintf('Denoise a volume using the FGP-dTV model (GPU) \n');

% create another volume (reference) with slightly less amount of noise
vol3D_ref = zeros(N,N,slices, 'single');
for i = 1:slices
vol3D_ref(:,:,i) = Im + .01*randn(size(Im)); 
end
vol3D_ref(vol3D_ref < 0) = 0;
% vol3D_ref = zeros(size(Im),'single'); % pass zero reference (dTV -> TV)

iter_fgp = 300; % number of FGP iterations
epsil_tol =  1.0e-05; % tolerance
eta =  0.2; % Reference image gradient smoothing constant
tic; u_fgp_dtv_g = FGP_dTV_GPU(single(vol3D), single(vol3D_ref), lambda_reg, iter_fgp, epsil_tol, eta); toc; 
figure; imshow(u_fgp_dtv_g(:,:,15), [0 1]); title('FGP-dTV denoised volume (GPU)');
%%