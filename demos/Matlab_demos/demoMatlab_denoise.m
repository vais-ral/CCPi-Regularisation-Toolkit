% Image (2D) denoising demo using CCPi-RGL
clear; close all
fsep = '/';

Path2 = sprintf(['..' fsep 'data' fsep], 1i);
Path3 = sprintf(['..' fsep '..' fsep 'src' fsep 'Matlab' fsep 'supp'], 1i);
Path1 = sprintf(['..' fsep '..' fsep 'src' fsep 'Matlab' fsep 'mex_compile' fsep 'installed'], 1i);
addpath(Path1);
addpath(Path2);
addpath(Path3);

Im = double(imread('peppers.tif'))/255;  % loading image
u0 = Im + .05*randn(size(Im)); u0(u0 < 0) = 0;
figure; imshow(u0, [0 1]); title('Noisy image');
%%
fprintf('Denoise using the ROF-TV model (CPU) \n');
lambda_reg = 0.03; % regularsation parameter for all methods
iter_rof = 1500; % number of ROF iterations
tau_rof = 0.003; % time-marching constant 
epsil_tol =  0.0; % tolerance / 1.0e-06
tic; [u_rof,infovec] = ROF_TV(single(u0), lambda_reg, iter_rof, tau_rof, epsil_tol); toc; 
energyfunc_val_rof = TV_energy(single(u_rof),single(u0),lambda_reg, 1);  % get energy function value
rmseROF = (RMSE(u_rof(:),Im(:)));
fprintf('%s %f \n', 'RMSE error for ROF-TV is:', rmseROF);
[ssimval] = ssim(u_rof*255,single(Im)*255);
fprintf('%s %f \n', 'MSSIM error for ROF-TV is:', ssimval);
figure; imshow(u_rof, [0 1]); title('ROF-TV denoised image (CPU)');
%%
%fprintf('Denoise using the ROF-TV model (GPU) \n');
%tic; [u_rofG,infovec]  = ROF_TV_GPU(single(u0), lambda_reg, iter_rof, tau_rof, epsil_tol); toc; 
%figure; imshow(u_rofG, [0 1]); title('ROF-TV denoised image (GPU)');
%%
fprintf('Denoise using the FGP-TV model (CPU) \n');
lambda_reg = 0.03;
iter_fgp = 500; % number of FGP iterations
epsil_tol =  0.0; % tolerance
tic; [u_fgp,infovec] = FGP_TV(single(u0), lambda_reg, iter_fgp, epsil_tol); toc; 
energyfunc_val_fgp = TV_energy(single(u_fgp),single(u0),lambda_reg, 1); % get energy function value
rmseFGP = (RMSE(u_fgp(:),Im(:)));
fprintf('%s %f \n', 'RMSE error for FGP-TV is:', rmseFGP);
[ssimval] = ssim(u_fgp*255,single(Im)*255);
fprintf('%s %f \n', 'MSSIM error for FGP-TV is:', ssimval);
figure; imshow(u_fgp, [0 1]); title('FGP-TV denoised image (CPU)');
%%
% fprintf('Denoise using the FGP-TV model (GPU) \n');
% tic; u_fgpG = FGP_TV_GPU(single(u0), lambda_reg, iter_fgp, epsil_tol); toc; 
% figure; imshow(u_fgpG, [0 1]); title('FGP-TV denoised image (GPU)');
%%
fprintf('Denoise using the PD-TV model (CPU) \n');
lambda_reg = 0.03;
iter_pd = 500; % number of FGP iterations
epsil_tol =  0.0; % tolerance
tic; [u_pd,infovec] = PD_TV(single(u0), lambda_reg, iter_pd, epsil_tol); toc; 
energyfunc_val_pd = TV_energy(single(u_pd),single(u0),lambda_reg, 1); % get energy function value
rmsePD = (RMSE(u_pd(:),Im(:)));
fprintf('%s %f \n', 'RMSE error for PD-TV is:', rmsePD);
[ssimval] = ssim(u_pd*255,single(Im)*255);
fprintf('%s %f \n', 'MSSIM error for PD-TV is:', ssimval);
figure; imshow(u_pd, [0 1]); title('PD-TV denoised image (CPU)');
%%
% fprintf('Denoise using the PD-TV model (GPU) \n');
% tic; u_pdG = PD_TV_GPU(single(u0), lambda_reg, iter_pd, epsil_tol); toc; 
% figure; imshow(u_pdG, [0 1]); title('PD-TV denoised image (GPU)');
%%
fprintf('Denoise using the SB-TV model (CPU) \n');
lambda_reg = 0.03;
iter_sb = 200; % number of SB iterations
epsil_tol =  0.0; % tolerance
tic; [u_sb,infovec] = SB_TV(single(u0), lambda_reg, iter_sb, epsil_tol); toc; 
energyfunc_val_sb = TV_energy(single(u_sb),single(u0),lambda_reg, 1);  % get energy function value
rmseSB = (RMSE(u_sb(:),Im(:)));
fprintf('%s %f \n', 'RMSE error for SB-TV is:', rmseSB);
[ssimval] = ssim(u_sb*255,single(Im)*255);
fprintf('%s %f \n', 'MSSIM error for SB-TV is:', ssimval);
figure; imshow(u_sb, [0 1]); title('SB-TV denoised image (CPU)');
%%
% fprintf('Denoise using the SB-TV model (GPU) \n');
% tic; u_sbG = SB_TV_GPU(single(u0), lambda_reg, iter_sb, epsil_tol); toc; 
% figure; imshow(u_sbG, [0 1]); title('SB-TV denoised image (GPU)');
%%
fprintf('Denoise using Nonlinear-Diffusion model (CPU) \n');
iter_diff = 450; % number of diffusion iterations
lambda_regDiff = 0.025; % regularisation for the diffusivity 
sigmaPar = 0.015; % edge-preserving parameter
tau_param = 0.02; % time-marching constant 
epsil_tol =  0.0; % tolerance
tic; [u_diff,infovec] = NonlDiff(single(u0), lambda_regDiff, sigmaPar, iter_diff, tau_param, 'Huber', epsil_tol); toc; 
rmseDiffus = (RMSE(u_diff(:),Im(:)));
fprintf('%s %f \n', 'RMSE error for Nonlinear Diffusion is:', rmseDiffus);
[ssimval] = ssim(u_diff*255,single(Im)*255);
fprintf('%s %f \n', 'MSSIM error for NDF is:', ssimval);
figure; imshow(u_diff, [0 1]); title('Diffusion denoised image (CPU)');
%%
%fprintf('Denoise using Nonlinear-Diffusion model (GPU) \n');
%tic; u_diff_g = NonlDiff_GPU(single(u0), lambda_regDiff, sigmaPar, iter_diff, tau_param, 'Huber', epsil_tol); toc; 
%figure; imshow(u_diff_g, [0 1]); title('Diffusion denoised image (GPU)');
%%
fprintf('Denoise using the TGV model (CPU) \n');
lambda_TGV = 0.035; % regularisation parameter
alpha1 = 1.0; % parameter to control the first-order term
alpha0 = 2.0; % parameter to control the second-order term
L2 =  12.0; % convergence parameter
iter_TGV = 1200; % number of Primal-Dual iterations for TGV
epsil_tol =  0.0; % tolerance
tic; [u_tgv,infovec] = TGV(single(u0), lambda_TGV, alpha1, alpha0, iter_TGV, L2, epsil_tol); toc; 
figure; imshow(u_tgv, [0 1]); title('TGV denoised image (CPU)');
rmseTGV = (RMSE(u_tgv(:),Im(:)));
fprintf('%s %f \n', 'RMSE error for TGV is:', rmseTGV);
[ssimval] = ssim(u_tgv*255,single(Im)*255);
fprintf('%s %f \n', 'MSSIM error for TGV is:', ssimval);
%%
% fprintf('Denoise using the TGV model (GPU) \n');
% tic; u_tgv_gpu = TGV_GPU(single(u0), lambda_TGV, alpha1, alpha0, iter_TGV, L2, epsil_tol); toc; 
% figure; imshow(u_tgv_gpu, [0 1]); title('TGV denoised image (GPU)');
%%
fprintf('Denoise using the ROF-LLT model (CPU) \n');
lambda_ROF = 0.02; % ROF regularisation parameter
lambda_LLT = 0.015; % LLT regularisation parameter
iter_LLT = 2000; % iterations 
tau_rof_llt = 0.01; % time-marching constant 
epsil_tol = 0.0; % tolerance
tic; [u_rof_llt,infovec]  = LLT_ROF(single(u0), lambda_ROF, lambda_LLT, iter_LLT, tau_rof_llt,epsil_tol); toc; 
rmseROFLLT = (RMSE(u_rof_llt(:),Im(:)));
fprintf('%s %f \n', 'RMSE error for TGV is:', rmseROFLLT);
[ssimval] = ssim(u_rof_llt*255,single(Im)*255);
fprintf('%s %f \n', 'MSSIM error for ROFLLT is:', ssimval);
figure; imshow(u_rof_llt, [0 1]); title('ROF-LLT denoised image (CPU)');
%%
% fprintf('Denoise using the ROF-LLT model (GPU) \n');
% tic; u_rof_llt_g = LLT_ROF_GPU(single(u0), lambda_ROF, lambda_LLT, iter_LLT, tau_rof_llt, epsil_tol); toc; 
% figure; imshow(u_rof_llt_g, [0 1]); title('ROF-LLT denoised image (GPU)');
%%
fprintf('Denoise using Fourth-order anisotropic diffusion model (CPU) \n');
iter_diff = 800; % number of diffusion iterations
lambda_regDiff = 3; % regularisation for the diffusivity 
sigmaPar = 0.03; % edge-preserving parameter
tau_param = 0.0025; % time-marching constant 
epsil_tol =  0.0; % tolerance
tic; [u_diff4,infovec] = Diffusion_4thO(single(u0), lambda_regDiff, sigmaPar, iter_diff, tau_param, epsil_tol); toc; 
rmseDiffHO = (RMSE(u_diff4(:),Im(:)));
fprintf('%s %f \n', 'RMSE error for Fourth-order anisotropic diffusion is:', rmseDiffHO);
[ssimval] = ssim(u_diff4*255,single(Im)*255);
fprintf('%s %f \n', 'MSSIM error for DIFF4th is:', ssimval);
figure; imshow(u_diff4, [0 1]); title('Diffusion 4thO denoised image (CPU)');
%%
%fprintf('Denoise using Fourth-order anisotropic diffusion model (GPU) \n');
%tic; u_diff4_g = Diffusion_4thO_GPU(single(u0), lambda_regDiff, sigmaPar, iter_diff, tau_param); toc; 
%figure; imshow(u_diff4_g, [0 1]); title('Diffusion 4thO denoised image (GPU)');
%%
fprintf('Weights pre-calculation for Non-local TV (takes time on CPU) \n');
SearchingWindow = 7;
PatchWindow = 2;
NeighboursNumber = 20; % the number of neibours to include
h = 0.23; % edge related parameter for NLM
tic; [H_i, H_j, Weights] = PatchSelect(single(u0), SearchingWindow, PatchWindow, NeighboursNumber, h); toc;
%%
fprintf('Denoise using Non-local Total Variation (CPU) \n');
iter_nltv = 3; % number of nltv iterations
lambda_nltv = 0.055; % regularisation parameter for nltv
tic; u_nltv = Nonlocal_TV(single(u0), H_i, H_j, 0, Weights, lambda_nltv, iter_nltv); toc; 
rmse_nltv = (RMSE(u_nltv(:),Im(:)));
fprintf('%s %f \n', 'RMSE error for Non-local Total Variation is:', rmse_nltv);
[ssimval] = ssim(u_nltv*255,single(Im)*255);
fprintf('%s %f \n', 'MSSIM error for NLTV is:', ssimval);
figure; imagesc(u_nltv, [0 1]); colormap(gray); daspect([1 1 1]); title('Non-local Total Variation denoised image (CPU)');
%%
%>>>>>>>>>>>>>> MULTI-CHANNEL priors <<<<<<<<<<<<<<< %
fprintf('Denoise using the FGP-dTV model (CPU) \n');
% create another image (reference) with slightly less amount of noise
u_ref = Im + .01*randn(size(Im)); u_ref(u_ref < 0) = 0;
% u_ref = zeros(size(Im),'single'); % pass zero reference (dTV -> TV)

lambda_reg = 0.04;
iter_fgp = 1000; % number of FGP iterations
epsil_tol =  0.0; % tolerance
eta =  0.2; % Reference image gradient smoothing constant
tic; [u_fgp_dtv,infovec] = FGP_dTV(single(u0), single(u_ref), lambda_reg, iter_fgp, epsil_tol, eta); toc; 
rmse_dTV= (RMSE(u_fgp_dtv(:),Im(:)));
fprintf('%s %f \n', 'RMSE error for Directional Total Variation (dTV) is:', rmse_dTV);
figure; imshow(u_fgp_dtv, [0 1]); title('FGP-dTV denoised image (CPU)');
%%
% fprintf('Denoise using the FGP-dTV model (GPU) \n');
% % create another image (reference) with slightly less amount of noise
% u_ref = Im + .01*randn(size(Im)); u_ref(u_ref < 0) = 0;
% % u_ref = zeros(size(Im),'single'); % pass zero reference (dTV -> TV)
% 
% iter_fgp = 1000; % number of FGP iterations
% epsil_tol =  1.0e-06; % tolerance
% eta =  0.2; % Reference image gradient smoothing constant
% tic; u_fgp_dtvG = FGP_dTV_GPU(single(u0), single(u_ref), lambda_reg, iter_fgp, epsil_tol, eta); toc; 
% figure; imshow(u_fgp_dtvG, [0 1]); title('FGP-dTV denoised image (GPU)');
%%
fprintf('Denoise using the TNV prior (CPU) \n');
slices = 5; N = 512;
vol3D = zeros(N,N,slices, 'single');
for i = 1:slices
vol3D(:,:,i) = Im + .05*randn(size(Im)); 
end
vol3D(vol3D < 0) = 0;

iter_tnv = 200; % number of TNV iterations
tic; u_tnv = TNV(single(vol3D), lambda_reg, iter_tnv); toc; 
figure; imshow(u_tnv(:,:,3), [0 1]); title('TNV denoised stack of channels (CPU)');
