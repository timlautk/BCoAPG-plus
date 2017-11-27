clear all
close all
clc

addpath Tools Algorithms

m = 1000;
n = 5000;

rng('default');
seed = 20;
rng(seed);

%% data generation 
[A,b,eta] = data_gen(m,n);

%% Lasso, min 1/(2*n)*norm(A*x-b)^2+lambda*norm(x,1)
lambda_l1 = 1;
beta_l1 = 0.9;
t_l1 = 0.9;
s_l1 = 5;
fun_l1 = @(x) 1/(2*n)*norm(A*x-b,2)^2+lambda_l1*norm(x,1); % objective function
niter_l1 = 400; % number of iterations for Lasso

disp('Lasso fitting')
disp('APG:')
val_apg_l1 = apg(A,b,lambda_l1,eta,niter_l1,fun_l1,@l1_prox);
disp('APGnc:')
val_apgnc_l1 = apgnc(A,b,lambda_l1,eta,niter_l1,fun_l1,@l1_prox);
disp('APGnc+:')
val_apgncp_l1 = apgncp(A,b,lambda_l1,eta,beta_l1,t_l1,niter_l1,fun_l1,@l1_prox);
disp('randomized BPL:')
val_bpl_l1 = bpl(A,b,lambda_l1,s_l1,niter_l1,fun_l1,@l1_prox);
disp('randomized BCoAPGnc+:')
val_ranbcapgncp_l1 = bcapgncp(A,b,lambda_l1,s_l1,beta_l1,t_l1,niter_l1,fun_l1,@l1_prox,1,@co_update_l1);
disp('GS-r BCoAPGnc+:')
val_bcapgncp_l1 = bcapgncp(A,b,lambda_l1,s_l1,beta_l1,t_l1,niter_l1,fun_l1,@l1_prox,2,@co_update_l1);


%% Grouped Lasso, min 1/(2*n)*norm(A*x-b)^2+lambda*norm(x)
lambda_l1l2 = 1;
s_l1l2 = 5;
N = n/s_l1l2;
beta_l1l2 = 0.8;
t_l1l2 = 0.2;
fun_l1l2 = @(x) 1/(2*n)*norm(A*x-b,2)^2+lambda_l1l2*(norm(x(1:N),2)+norm(x(N+1:2*N),2)+norm(x(2*N+1:3*N),2)+norm(x(3*N+1:4*N),2)+norm(x(4*N+1:end),2));
niter_l1l2 = 300; % number of iterations for grouped Lasso

disp('Grouped Lasso fitting')
disp('randomized BPL:')
val_bpl_l1l2 = bpl_l1l2(A,b,lambda_l1l2,s_l1l2,niter_l1l2,fun_l1l2);
disp('randomized BCoAPGnc+:')
val_ranbcapgncp_l1l2 = bcapgncp_l1l2(A,b,lambda_l1l2,s_l1l2,beta_l1l2,t_l1l2,niter_l1l2,fun_l1l2,1); % last arg: 1 = random update; 2 = GS-r update
disp('GS-r BCoAPGnc+:')
val_bcapgncp_l1l2 = bcapgncp_l1l2(A,b,lambda_l1l2,s_l1l2,beta_l1l2,t_l1l2,niter_l1l2,fun_l1l2,2);


%% Capped l1-regularized least squares, min 1/(2*n)*norm(A*x-b)^2+lambda*sum(min(abs(x),theta))
lambda_cappedl1 = 0.0001;
theta_cappedl1 = 0.1*lambda_cappedl1;
beta_cappedl1 = 0.8;
t_cappedl1 = 0.2;
s_cappedl1 = 10;
fun_cappedl1 = @(x) 1/(2*n)*norm(A*x-b,2)^2+lambda_cappedl1*sum(min(abs(x),theta_cappedl1));
niter_cappedl1 = 600;

disp('Capped l1-regularized least square fitting')
disp('APG:')
val_apg_cappedl1 = apg(A,b,lambda_cappedl1,eta,niter_cappedl1,fun_cappedl1,@cap_l1_prox,theta_cappedl1);
disp('APGnc:')
val_apgnc_cappedl1 = apgnc(A,b,lambda_cappedl1,eta,niter_cappedl1,fun_cappedl1,@cap_l1_prox,theta_cappedl1);
disp('APGnc+:')
val_apgncp_cappedl1 = apgncp(A,b,lambda_cappedl1,eta,beta_cappedl1,t_cappedl1,niter_cappedl1,fun_cappedl1,@cap_l1_prox,theta_cappedl1);
disp('BPL:')
val_bpl_cappedl1 = bpl(A,b,lambda_cappedl1,s_cappedl1,niter_cappedl1,fun_cappedl1,@cap_l1_prox,theta_cappedl1);
disp('randomized BCoAPGnc+:')
val_ranbcapgncp_cappedl1 = bcapgncp(A,b,lambda_cappedl1,s_cappedl1,beta_cappedl1,t_cappedl1,niter_cappedl1,fun_cappedl1,@cap_l1_prox,1,@co_update_capped_l1,theta_cappedl1);
disp('GS-r BCoAPGnc+:')
val_bcapgncp_cappedl1 = bcapgncp(A,b,lambda_cappedl1,s_cappedl1,beta_cappedl1,t_cappedl1,niter_cappedl1,fun_cappedl1,@cap_l1_prox,2,@co_update_capped_l1,theta_cappedl1);


%% Nonconvex Regression (SCAD)
% data generation 
[A_scad,b_scad,eta_scad] = data_gen_scad(m,n); 
lambda_scad = 0.0001;
gamma_scad = 3;
niter_scad = 25;
beta_scad = 0.8;
t_scad = 0.2;
s_scad = 10;

disp('SCAD-regularized least square fitting')
disp('APG:')
val_apg_scad = apg_scad(A_scad,b_scad,lambda_scad,eta_scad,niter_scad,gamma_scad);
disp('APGnc:')
val_apgnc_scad = apgnc_scad(A_scad,b_scad,lambda_scad,eta_scad,niter_scad,gamma_scad);
disp('APGnc+:')
val_apgncp_scad = apgncp_scad(A_scad,b_scad,lambda_scad,eta_scad,beta_scad,t_scad,niter_scad,gamma_scad);
disp('BPL:')
val_bpl_scad = bpl_scad(A_scad,b_scad,lambda_scad,s_scad,niter_scad,gamma_scad);
disp('randomized BCoAPGnc+:')
val_ranbcapgncp_scad = bcapgncp_scad(A_scad,b_scad,lambda_scad,s_scad,beta_scad,t_scad,niter_scad,1,gamma_scad);
disp('GS-r BCoAPGnc+:')
val_bcapgncp_scad = bcapgncp_scad(A_scad,b_scad,lambda_scad,s_scad,beta_scad,t_scad,niter_scad,2,gamma_scad);


%% Plot the results (Objective value - Optimal Value vs no. of iterations)
% Lasso
iter_l1 = 1:niter_l1;
figure(1);
graph1 = semilogy(iter_l1,val_apg_l1-645.314,'-k',iter_l1,val_apgnc_l1-645.314,'--m',iter_l1,val_apgncp_l1-645.314,'-g',iter_l1,val_bpl_l1-645.314,'r',iter_l1,val_ranbcapgncp_l1-645.314,'--',iter_l1,val_bcapgncp_l1-645.314,'-b');
set(graph1,'LineWidth',1.5);
legend('APG','APGnc','APGnc^+','BPL (randomized)','BCoAPGnc^+ (randomized)','BCoAPGnc^+ (GS-r)','interpreter','latex');
% ylabel('Objective value - Optimal value','interpreter','latex')
xlabel('Number of iterations','interpreter','latex')
title('$\ell_1$-regularized Sparse Least Squares','interpreter','latex')


% Grouped Lasso
iter_l1l2 = 1:niter_l1l2;
figure(2);
graph2 = semilogy(iter_l1l2,val_bpl_l1l2-65.0197,'r',iter_l1l2,val_ranbcapgncp_l1l2-65.0197,'--',iter_l1l2,val_bcapgncp_l1l2-65.0197,'-b');
set(graph2,'LineWidth',1.5);
legend('BPL (randomized)','BCoAPGnc^+ (randomized)','BCoAPGnc^+ (GS-r)','interpreter','latex');
% ylabel('Objective value - Optimal value','interpreter','latex')
xlabel('Number of iterations','interpreter','latex')
title('$\ell_1/\ell_2$-regularized Sparse Least Squares','interpreter','latex')


% Capped l1
iter_cappedl1 = 1:niter_cappedl1;
figure(3);
graph3 = semilogy(iter_cappedl1,val_apg_cappedl1-2.8378e-4,'.-k',iter_cappedl1,val_apgnc_cappedl1-2.8378e-4,'--m',iter_cappedl1,val_apgncp_cappedl1-2.8378e-4,'-g',iter_cappedl1,val_bpl_cappedl1-2.8378e-4,'r',iter_cappedl1,val_ranbcapgncp_cappedl1-2.8378e-4,'--',iter_cappedl1,val_bcapgncp_cappedl1-2.8378e-4,'-b');
set(graph3,'LineWidth',1.5);
legend('APG','APGnc','APGnc^+','BPL (randomized)','BCoAPGnc^+ (randomized)','BCoAPGnc^+ (GS-r)','interpreter','latex');
% ylabel('Objective value - Optimal value','interpreter','latex')
xlabel('Number of iterations','interpreter','latex')
title('Capped $\ell_1$-regularized Sparse Least Squares','interpreter','latex')


% SCAD
iter_scad = 1:niter_scad;
figure(4);
graph4 = semilogy(iter_scad,val_apg_scad-8.412e-5,'-k',iter_scad,val_apgnc_scad-8.412e-5,'--m',iter_scad,val_apgncp_scad-8.412e-5,'-g',iter_scad,val_bpl_scad-8.412e-5,'r',iter_scad,val_ranbcapgncp_scad-8.412e-5,'--',iter_scad,val_bcapgncp_scad-8.412e-5,'-b');
set(graph4,'LineWidth',1.5);
legend('APG','APGnc','APGnc^+','BPL (randomized)','BCoAPGnc^+ (randomized)','BCoAPGnc^+ (GS-r)','interpreter','latex');
% ylabel('Objective value - Optimal value','interpreter','latex')
xlabel('Number of iterations','interpreter','latex')
title('Least Squares with SCAD penalty','interpreter','latex')

