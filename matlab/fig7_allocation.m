% Figure 7: belief-dependent allocation and rate-feasible policy tradeoffs.
QUICK=false;
TRIALS=2000; if QUICK, TRIALS=20; end
sigma=logspace(log10(.2),log10(12),81);
cfg=isac.sim_config(); w=isac.sim_beams(cfg.q0,cfg);
[Z,~,meta]=isac.efim(cfg.q0,w,cfg);
g=meta.g; Rmax=isac.alloc_rate(isac.alloc_waterfill(g,cfg.Emax),g,cfg);
Rmin=.5*Rmax;

% Full phase-averaged atoms drive every allocation solve.
fractions=zeros(numel(sigma),cfg.R);
for k=1:numel(sigma)
    Lam=cfg.Lambda0; Lam(8:9,8:9)=eye(2)/sigma(k)^2;
    fractions(k,:)=isac.alloc_solve(Lam,Z,g,Rmin,'joint',cfg)'/cfg.Emax;
end
names={'Joint D','Position only','Target conditional blocks','Rate only', ...
       'Uniform/rate mixture','Strongest reflector'};
e=zeros(cfg.R,6);
for p=1:4
    modes={'joint','loc','target','rate'};
    e(:,p)=isac.alloc_solve(cfg.Lambda0,Z,g,Rmin,modes{p},cfg);
end
[e(:,5),mixture]=isac.alloc_fixed(g,Rmin,cfg);
lo=0; hi=cfg.Emax;
for k=1:60
    mid=(lo+hi)/2; candidate=[mid;cfg.Emax-mid;0;0;0];
    if isac.alloc_rate(candidate,g,cfg)>=Rmin, hi=mid; else, lo=mid; end
end
e(:,6)=[hi;cfg.Emax-hi;0;0;0];
nominal=zeros(6,2);
for p=1:6
    [peb,~,meb]=isac.alloc_metrics(isac.alloc_post(cfg.Lambda0,Z,e(:,p)));
    nominal(p,:)=[peb,mean(meb)];
end

% The same held-out phases are used for all six covariance policies.
outcomes=zeros(TRIALS,6,2);
for trial=1:TRIALS
    rng(cfg.master_seed+600000+trial,'twister');
    gains.env=sqrt(cfg.env_power).*exp(2i*pi*rand(cfg.J,cfg.R));
    gains.bs=sqrt(cfg.bs_power)*exp(2i*pi*rand(1,cfg.R));
    Zt=isac.efim(cfg.q0,w,cfg,gains);
    for p=1:6
        [peb,~,meb]=isac.alloc_metrics(isac.alloc_post(cfg.Lambda0,Zt,e(:,p)));
        outcomes(trial,p,:)=[peb,mean(meb)];
    end
end
paired=reshape(outcomes(:,2,:)-outcomes(:,1,:),TRIALS,2);
[pair_mean,pair_lo,pair_hi]=isac.bootstrap_mean(paired,cfg.master_seed+900002);
fprintf('Fig. 7a L3-directed share %.9g -> %.9g\n',fractions(1,4),fractions(end,4));
fprintf('Fig. 7b joint / position-only / feasible-uniform PEB/MEB:\n');
disp(nominal([1,2,5],:));
fprintf('Fig. 7b uniform rate-only mixture %.9g\n',mixture);
fprintf('Fig. 7b held-out position-only minus joint PEB/MEB, paired 95%% CI:\n');
disp([pair_mean;pair_lo;pair_hi]);

col=isac.plot_style(false);
fig=figure('Color','w','Units','inches','Position',[1 1 7.16 2.55]);
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
nexttile;
for r=1:cfg.R
    semilogx(sigma,fractions(:,r),'Color',col{r},'LineWidth',1.4); hold on;
end
xlim([.2 12]); ylim([0 .65]); grid on; title('(a)');
xlabel('Landmark-3 prior coordinate SD (m)'); ylabel('Energy fraction');
legend('BS','Landmark 1','Landmark 2','Landmark 3','Landmark 4', ...
       'Location','northwest','FontSize',7);
nexttile;
for p=1:6
    plot(nominal(p,1),nominal(p,2),'o','Color',col{p},'MarkerSize',4); hold on;
end
xlim([.8 1.32]); ylim([2.70 3.75]); grid on; title('(b)');
xlabel('Position covariance metric PEB (m)');
ylabel('Mean landmark covariance metric (m)');
legend(names,'Location','northwest','FontSize',6);
exportgraphics(fig,fullfile(fileparts(mfilename('fullpath')),'fig06_belief_baselines.pdf'),'ContentType','vector');
