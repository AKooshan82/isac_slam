% Figure 9: generic greedy selection and paired SI/power robustness.
QUICK=false;
DISCRETE_TRIALS=250; PHASE_TRIALS=2000;
if QUICK, DISCRETE_TRIALS=20; PHASE_TRIALS=20; end
si_db=linspace(-30,20,101); mismatch_db=linspace(-15,10,101);
cfg=isac.sim_config();

% Panel (a): generic PSD atoms, 12 actions, cardinality 1--5, no rate.
sets=cell(1,5);
for K=1:5, sets{K}=nchoosek(1:12,K); end
ratios=zeros(DISCRETE_TRIALS,5);
for trial=1:DISCRETE_TRIALS
    rng(cfg.master_seed+900000+trial,'twister');
    A=randn(6); Lam=A*A'+2*eye(6); atoms=zeros(6,6,12);
    for r=1:12
        v=randn(6,1); atoms(:,:,r)=(.15+rand())*(v*v')/6;
    end
    baseline=logdet_spd(Lam); current=Lam; chosen=[];
    for K=1:5
        best=-Inf; pick=0;
        for r=setdiff(1:12,chosen)
            value=logdet_spd(current+atoms(:,:,r))-baseline;
            if value>best, best=value; pick=r; end
        end
        chosen=[chosen,pick]; current=current+atoms(:,:,pick);
        greedy=logdet_spd(current)-baseline; oracle=-Inf;
        for row=1:size(sets{K},1)
            ix=sets{K}(row,:);
            oracle=max(oracle,logdet_spd(Lam+sum(atoms(:,:,ix),3))-baseline);
        end
        ratios(trial,K)=greedy/oracle;
    end
end
[ratio_mean,ratio_lo,ratio_hi]=isac.bootstrap_mean(ratios,cfg.master_seed+950000);
ratio_min=min(ratios,[],1);

% Panels (b,c): design with phase-averaged full multipath atoms.
w=isac.sim_beams(cfg.q0,cfg); [Z,~,meta]=isac.efim(cfg.q0,w,cfg);
g=meta.g; Rmin=.5*isac.alloc_rate(isac.alloc_waterfill(g,cfg.Emax),g,cfg);
enom=isac.alloc_solve(cfg.Lambda0,Z,g,Rmin,'joint',cfg);
ordered_db=[-12,-6,0];
ordered=zeros(size(Z,1),size(Z,2),size(Z,3),3);
for k=1:3
    c=cfg; c.env_power(3)=c.env_power(3)*10^(ordered_db(k)/10);
    ordered(:,:,:,k)=isac.efim(c.q0,w,c);
end
erobust=isac.robust_alloc(cfg.Lambda0,ordered,g,Rmin,cfg);
nonordered=zeros(size(Z,1),size(Z,2),size(Z,3),cfg.J);
for j=1:cfg.J
    c=cfg; c.env_power(j)=c.env_power(j)*10^(-12/10);
    nonordered(:,:,:,j)=isac.efim(c.q0,w,c);
end
erobust_nonordered=isac.robust_alloc(cfg.Lambda0,nonordered,g,Rmin,cfg);
si_energies=zeros(cfg.R,numel(si_db));
for k=1:numel(si_db)
    c=cfg; c.noiseS=1+10^(si_db(k)/10);
    Zsi=isac.efim(c.q0,w,c);
    si_energies(:,k)=isac.alloc_solve(cfg.Lambda0,Zsi,g,Rmin,'joint',cfg);
end

% The same held-out phases are reused at every impairment level.
basis=isac.model_basis(cfg.q0,w,cfg);
si_diff=zeros(PHASE_TRIALS,numel(si_db));
power_diff=zeros(PHASE_TRIALS,numel(mismatch_db));
nonordered_diff=zeros(PHASE_TRIALS,numel(mismatch_db));
for trial=1:PHASE_TRIALS
    rng(cfg.master_seed+1200000+trial,'twister');
    gains.env=sqrt(cfg.env_power).*exp(2i*pi*rand(cfg.J,cfg.R));
    gains.bs=sqrt(cfg.bs_power)*exp(2i*pi*rand(1,cfg.R));
    [~,ZS,ZB]=isac.model_from_basis(basis,gains);
    for k=1:numel(si_db)
        actual=ZS/(1+10^(si_db(k)/10))+ZB;
        si_diff(trial,k)=gain(cfg.Lambda0,actual,si_energies(:,k))- ...
                         gain(cfg.Lambda0,actual,enom);
    end
    for k=1:numel(mismatch_db)
        changed=gains; changed.env(3,:)=gains.env(3,:)*10^(mismatch_db(k)/20);
        actual=isac.model_from_basis(basis,changed);
        base=gain(cfg.Lambda0,actual,enom);
        power_diff(trial,k)=gain(cfg.Lambda0,actual,erobust)-base;
        nonordered_diff(trial,k)=gain(cfg.Lambda0,actual,erobust_nonordered)-base;
    end
end
[si_mean,si_lo,si_hi]=isac.bootstrap_mean(si_diff,cfg.master_seed+1299999);
[power_mean,power_lo,power_hi]=isac.bootstrap_mean(power_diff,cfg.master_seed+1299999);

fprintf('Fig. 9a greedy/exhaustive mean ratio by K:'); fprintf(' %.9g',ratio_mean); fprintf('\n');
fprintf('Fig. 9a minimum individual ratio %.9g\n',min(ratios(:)));
fprintf('Fig. 9b SI-aware minus nominal at ISR 20 dB: %.9g nat, CI [%.9g, %.9g]\n', ...
        si_mean(end),si_lo(end),si_hi(end));
for db=[-12,0]
    [~,k]=min(abs(mismatch_db-db));
    fprintf('Fig. 9c robust minus nominal at %g dB: %.9g nat, CI [%.9g, %.9g]\n', ...
            db,power_mean(k),power_lo(k),power_hi(k));
end
fprintf('Fig. 9c maximum ordered versus nonordered robust difference %.9g nat\n', ...
        max(abs(mean(power_diff-nonordered_diff,1))));

col=isac.plot_style(true);
fig=figure('Color','w','Units','inches','Position',[1 1 7.16 2.6]);
tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
nexttile;
errorbar(1:5,ratio_mean,ratio_mean-ratio_lo,ratio_hi-ratio_mean, ...
         '-o','Color',col{1},'LineWidth',1.8,'MarkerSize',3); hold on;
plot(1:5,ratio_min,'--s','Color',col{2},'LineWidth',1.8,'MarkerSize',3);
yline(1-1/exp(1),':','Color',col{4});
xlim([.8 5.2]); ylim([.6 1.04]); xticks(1:5); grid on; title('(a)');
xlabel('Selected actions K'); ylabel('Greedy / optimum gain');
legend('Mean + 95% CI','Minimum','1 - 1/e guarantee','Location','best','FontSize',6);
nexttile;
fill([si_db,fliplr(si_db)],[si_lo,fliplr(si_hi)],'#CCE0EC', ...
     'EdgeColor','none'); hold on;
plot(si_db,si_mean,'Color',col{1},'LineWidth',1.8); yline(0,':','Color',col{4});
xlim([-30 20]); ylim([-.02 .36]); grid on; title('(b)');
xlabel('Residual SI / noise (dB)'); ylabel('SI-aware - nominal gain (nat)');
legend('95% paired CI','Held-out phase mean','Location','northwest','FontSize',6);
nexttile;
fill([mismatch_db,fliplr(mismatch_db)],[power_lo,fliplr(power_hi)],'#F3D9C6', ...
     'EdgeColor','none'); hold on;
plot(mismatch_db,power_mean,'Color',col{2},'LineWidth',1.8);
yline(0,':','Color',col{4}); xline(-12,':','Color',col{4}); xline(0,':','Color',col{4});
xlim([-15 10]); ylim([-.25 .13]); grid on; title('(c)');
xlabel('Landmark-3 power mismatch (dB)'); ylabel('Robust - nominal gain (nat)');
legend('95% paired CI','Both robust designs','Location','northeast','FontSize',6);
exportgraphics(fig,fullfile(fileparts(mfilename('fullpath')),'fig08_discrete_robustness.pdf'),'ContentType','vector');

function v=logdet_spd(A)
% Cholesky log determinant of a positive-definite information matrix.
v=2*sum(log(diag(chol((A+A')/2))));
end

function v=gain(Lam,Z,e)
% Conditional joint information gain against the common prior.
v=logdet_spd(isac.alloc_post(Lam,Z,e))-logdet_spd(Lam);
end
