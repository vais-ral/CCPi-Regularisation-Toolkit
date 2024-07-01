% Volume (3D) denoising demo using CCPi-RGL
clear; close all
fsep = '/';


Path1 = sprintf(['..' fsep '..' fsep 'src' fsep 'Matlab' fsep 'mex_compile' fsep 'installed'], 1i);
Path2 = sprintf(['..' fsep 'data' fsep], 1i);
Path3 = sprintf(['..' fsep '..' fsep 'src' fsep 'Matlab' fsep 'supp'], 1i);
addpath(Path1);
addpath(Path2);
addpath(Path3);

N = 512; 
slices = 15;
vol3D = zeros(N,N,slices, 'single');
Ideal3D = zeros(N,N,slices, 'single');
Im = double(imread('peppers.tif'))/255;  % loading image
for i = 1:slices
vol3D(:,:,i) = Im + .05*randn(size(Im)); 
Ideal3D(:,:,i) = Im;
end
vol3D(vol3D < 0) = 0;
figure; imshow(vol3D(:,:,7), [0 1]); title('Noisy image');
%%
fprintf('Denoise a volume using the ROF-TV model (CPU) \n');
lambda_reg = 0.03; % regularsation parameter for all methods
tau_rof = 0.0025; % time-marching constant 
iter_rof = 300; % number of ROF iterations
epsil_tol =  0.0; % tolerance
tic; [u_rof,infovec] = ROF_TV(single(vol3D), lambda_reg, iter_rof, tau_rof, epsil_tol); toc; 
energyfunc_val_rof = TV_energy(single(u_rof),single(vol3D),lambda_reg, 1);  % get energy function value
rmse_rof = (RMSE(Ideal3D(:),u_rof(:)));
fprintf('%s %f \n', 'RMSE error for ROF is:', rmse_rof);
figure; imshow(u_rof(:,:,7), [0 1]); title('ROF-TV denoised volume (CPU)');
%%
% fprintf('Denoise a volume using the ROF-TV model (GPU) \n');
% lambda_reg = 0.03; % regularsation parameter for all methods
% tau_rof = 0.0025; % time-marching constant 
% iter_rof = 300; % number of ROF iterations
% epsil_tol =  0.0; % tolerance
% tic; u_rofG = ROF_TV_GPU(single(vol3D), lambda_reg, iter_rof, tau_rof, epsil_tol); toc;
% rmse_rofG = (RMSE(Ideal3D(:),u_rofG(:)));
% fprintf('%s %f \n', 'RMSE error for ROF is:', rmse_rofG);
% figure; imshow(u_rofG(:,:,7), [0 1]); title('ROF-TV denoised volume (GPU)');
%%
fprintf('Denoise a volume using the FGP-TV model (CPU) \n');
lambda_reg = 0.03; % regularsation parameter for all methods
iter_fgp = 300; % number of FGP iterations
epsil_tol =  0.0; % tolerance
tic; [u_fgp,infovec] = FGP_TV(single(vol3D), lambda_reg, iter_fgp, epsil_tol); toc; 
energyfunc_val_fgp = TV_energy(single(u_fgp),single(vol3D),lambda_reg, 1); % get energy function value
rmse_fgp = (RMSE(Ideal3D(:),u_fgp(:)));
fprintf('%s %f \n', 'RMSE error for FGP-TV is:', rmse_fgp);
figure; imshow(u_fgp(:,:,7), [0 1]); title('FGP-TV denoised volume (CPU)');
%%
fprintf('Denoise a volume using the FGP-TV model (GPU) \n');
% lambda_reg = 0.03; % regularsation parameter for all methods
% iter_fgp = 300; % number of FGP iterations
% epsil_tol =  0.0; % tolerance
% tic; u_fgpG = FGP_TV_GPU(single(vol3D), lambda_reg, iter_fgp, epsil_tol); toc; 
% rmse_fgpG = (RMSE(Ideal3D(:),u_fgpG(:)));
% fprintf('%s %f \n', 'RMSE error for FGP-TV is:', rmse_fgpG);
% figure; imshow(u_fgpG(:,:,7), [0 1]); title('FGP-TV denoised volume (GPU)');
%%
fprintf('Denoise a volume using the PD-TV model (CPU) \n');
lambda_reg = 0.03; % regularsation parameter for all methods
iter_pd = 300; % number of FGP iterations
epsil_tol =  0.0; % tolerance
tic; [u_pd,infovec] = PD_TV(single(vol3D), lambda_reg, iter_pd, epsil_tol); toc; 
energyfunc_val_fgp = TV_energy(single(u_pd),single(vol3D),lambda_reg, 1); % get energy function value
rmse_pd = (RMSE(Ideal3D(:),u_pd(:)));
fprintf('%s %f \n', 'RMSE error for PD-TV is:', rmse_pd);
figure; imshow(u_pd(:,:,7), [0 1]); title('PD-TV denoised volume (CPU)');
%%
% fprintf('Denoise a volume using the PD-TV model (GPU) \n');
% lambda_reg = 0.03; % regularsation parameter for all methods
% iter_pd = 300; % number of FGP iterations
% epsil_tol =  0.0; % tolerance
% tic; u_pdG = PD_TV_GPU(single(vol3D), lambda_reg, iter_pd, epsil_tol); toc; 
% rmse_pdG = (RMSE(Ideal3D(:),u_pdG(:)));
% fprintf('%s %f \n', 'RMSE error for PD-TV is:', rmse_pdG);
% figure; imshow(u_pdG(:,:,7), [0 1]); title('PD-TV denoised volume (GPU)');
%%
fprintf('Denoise a volume using the SB-TV model (CPU) \n');
iter_sb = 150; % number of SB iterations
epsil_tol =  0.0; % tolerance
tic; [u_sb,infovec] = SB_TV(single(vol3D), lambda_reg, iter_sb, epsil_tol); toc; 
energyfunc_val_sb = TV_energy(single(u_sb),single(vol3D),lambda_reg, 1);  % get energy function value
rmse_sb = (RMSE(Ideal3D(:),u_sb(:)));
fprintf('%s %f \n', 'RMSE error for SB-TV is:', rmse_sb);
figure; imshow(u_sb(:,:,7), [0 1]); title('SB-TV denoised volume (CPU)');
%%
% fprintf('Denoise a volume using the SB-TV model (GPU) \n');
% iter_sb = 150; % number of SB iterations
% epsil_tol =  0.0; % tolerance
% tic; u_sbG = SB_TV_GPU(single(vol3D), lambda_reg, iter_sb, epsil_tol); toc; 
% rmse_sbG = (RMSE(Ideal3D(:),u_sbG(:)));
% fprintf('%s %f \n', 'RMSE error for SB-TV is:', rmse_sbG);
% figure; imshow(u_sbG(:,:,7), [0 1]); title('SB-TV denoised volume (GPU)');
%%
fprintf('Denoise a volume using the ROF-LLT model (CPU) \n');
lambda_ROF = lambda_reg; % ROF regularisation parameter
lambda_LLT = lambda_reg*0.35; % LLT regularisation parameter
iter_LLT = 300; % iterations 
tau_rof_llt = 0.0025; % time-marching constant 
epsil_tol =  0.0; % tolerance
tic; [u_rof_llt, infovec] = LLT_ROF(single(vol3D), lambda_ROF, lambda_LLT, iter_LLT, tau_rof_llt, epsil_tol); toc; 
rmse_rof_llt = (RMSE(Ideal3D(:),u_rof_llt(:)));
fprintf('%s %f \n', 'RMSE error for ROF-LLT is:', rmse_rof_llt);
figure; imshow(u_rof_llt(:,:,7), [0 1]); title('ROF-LLT denoised volume (CPU)');
%%
% fprintf('Denoise a volume using the ROF-LLT model (GPU) \n');
% lambda_ROF = lambda_reg; % ROF regularisation parameter
% lambda_LLT = lambda_reg*0.35; % LLT regularisation parameter
% iter_LLT = 300; % iterations 
% tau_rof_llt = 0.0025; % time-marching constant 
% epsil_tol =  0.0; % tolerance
% tic; u_rof_llt_g = LLT_ROF_GPU(single(vol3D), lambda_ROF, lambda_LLT, iter_LLT, tau_rof_llt, epsil_tol); toc; 
% rmse_rof_llt = (RMSE(Ideal3D(:),u_rof_llt_g(:)));
% fprintf('%s %f \n', 'RMSE error for ROF-LLT is:', rmse_rof_llt);
% figure; imshow(u_rof_llt_g(:,:,7), [0 1]); title('ROF-LLT denoised volume (GPU)');
%%
fprintf('Denoise a volume using Nonlinear-Diffusion model (CPU) \n');
iter_diff = 300; % number of diffusion iterations
lambda_regDiff = 0.025; % regularisation for the diffusivity 
sigmaPar = 0.015; % edge-preserving parameter
tau_param = 0.025; % time-marching constant 
epsil_tol =  0.0; % tolerance
tic; [u_diff, infovec]  = NonlDiff(single(vol3D), lambda_regDiff, sigmaPar, iter_diff, tau_param, 'Huber', epsil_tol); toc; 
rmse_diff = (RMSE(Ideal3D(:),u_diff(:)));
fprintf('%s %f \n', 'RMSE error for Diffusion is:', rmse_diff);
figure; imshow(u_diff(:,:,7), [0 1]); title('Diffusion denoised volume (CPU)');
%%
% fprintf('Denoise a volume using Nonlinear-Diffusion model (GPU) \n');
% iter_diff = 300; % number of diffusion iterations
% lambda_regDiff = 0.025; % regularisation for the diffusivity 
% sigmaPar = 0.015; % edge-preserving parameter
% tau_param = 0.025; % time-marching constant 
% tic; u_diff_g = NonlDiff_GPU(single(vol3D), lambda_regDiff, sigmaPar, iter_diff, tau_param, 'Huber', epsil_tol); toc; 
% rmse_diff = (RMSE(Ideal3D(:),u_diff_g(:)));
% fprintf('%s %f \n', 'RMSE error for Diffusion is:', rmse_diff);
% figure; imshow(u_diff_g(:,:,7), [0 1]); title('Diffusion denoised volume (GPU)');
%%
fprintf('Denoise using Fourth-order anisotropic diffusion model (CPU) \n');
iter_diff = 300; % number of diffusion iterations
lambda_regDiff = 3.5; % regularisation for the diffusivity 
sigmaPar = 0.02; % edge-preserving parameter
tau_param = 0.0015; % time-marching constant 
epsil_tol =  0.0; % tolerance
tic; u_diff4 = Diffusion_4thO(single(vol3D), lambda_regDiff, sigmaPar, iter_diff, tau_param, epsil_tol); toc; 
rmse_diff4 = (RMSE(Ideal3D(:),u_diff4(:)));
fprintf('%s %f \n', 'RMSE error for Anis.Diff of 4th order is:', rmse_diff4);
figure; imshow(u_diff4(:,:,7), [0 1]); title('Diffusion 4thO denoised volume (CPU)');
%%
% fprintf('Denoise using Fourth-order anisotropic diffusion model (GPU) \n');
% iter_diff = 300; % number of diffusion iterations
% lambda_regDiff = 3.5; % regularisation for the diffusivity 
% sigmaPar = 0.02; % edge-preserving parameter
% tau_param = 0.0015; % time-marching constant 
% tic; u_diff4_g = Diffusion_4thO_GPU(single(vol3D), lambda_regDiff, sigmaPar, iter_diff, tau_param, epsil_tol); toc; 
% rmse_diff4 = (RMSE(Ideal3D(:),u_diff4_g(:)));
% fprintf('%s %f \n', 'RMSE error for Anis.Diff of 4th order is:', rmse_diff4);
% figure; imshow(u_diff4_g(:,:,7), [0 1]); title('Diffusion 4thO denoised volume (GPU)');
%%
fprintf('Denoise using the TGV model (CPU) \n');
lambda_TGV = 0.03; % regularisation parameter
alpha1 = 1.0; % parameter to control the first-order term
alpha0 = 2.0; % parameter to control the second-order term
L2 =  12.0; % convergence parameter
iter_TGV = 500; % number of Primal-Dual iterations for TGV
epsil_tol =  0.0; % tolerance
tic; u_tgv = TGV(single(vol3D), lambda_TGV, alpha1, alpha0, iter_TGV, L2, epsil_tol); toc; 
rmseTGV = RMSE(Ideal3D(:),u_tgv(:));
fprintf('%s %f \n', 'RMSE error for TGV is:', rmseTGV);
figure; imshow(u_tgv(:,:,3), [0 1]); title('TGV denoised volume (CPU)');
%%
% fprintf('Denoise using the TGV model (GPU) \n');
% lambda_TGV = 0.03; % regularisation parameter
% alpha1 = 1.0; % parameter to control the first-order term
% alpha0 = 2.0; % parameter to control the second-order term
% iter_TGV = 500; % number of Primal-Dual iterations for TGV
% tic; u_tgv_gpu = TGV_GPU(single(vol3D), lambda_TGV, alpha1, alpha0, iter_TGV, L2, epsil_tol); toc; 
% rmseTGV = RMSE(Ideal3D(:),u_tgv_gpu(:));
% fprintf('%s %f \n', 'RMSE error for TGV is:', rmseTGV);
% figure; imshow(u_tgv_gpu(:,:,3), [0 1]); title('TGV denoised volume (GPU)');
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
epsil_tol =  0.0; % tolerance
eta =  0.2; % Reference image gradient smoothing constant
tic; u_fgp_dtv = FGP_dTV(single(vol3D), single(vol3D_ref), lambda_reg, iter_fgp, epsil_tol, eta); toc; 
figure; imshow(u_fgp_dtv(:,:,7), [0 1]); title('FGP-dTV denoised volume (CPU)');
%%
% fprintf('Denoise a volume using the FGP-dTV model (GPU) \n');
% % create another volume (reference) with slightly less amount of noise
% vol3D_ref = zeros(N,N,slices, 'single');
% for i = 1:slices
% vol3D_ref(:,:,i) = Im + .01*randn(size(Im)); 
% end
% vol3D_ref(vol3D_ref < 0) = 0;
% % vol3D_ref = zeros(size(Im),'single'); % pass zero reference (dTV -> TV)
% 
% iter_fgp = 300; % number of FGP iterations
% epsil_tol =  0.0; % tolerance
% eta =  0.2; % Reference image gradient smoothing constant
% tic; u_fgp_dtv_g = FGP_dTV_GPU(single(vol3D), single(vol3D_ref), lambda_reg, iter_fgp, epsil_tol, eta); toc; 
% figure; imshow(u_fgp_dtv_g(:,:,7), [0 1]); title('FGP-dTV denoised volume (GPU)');
%%
