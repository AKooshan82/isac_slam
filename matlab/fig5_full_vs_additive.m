% Figure 5: full multipath atoms versus isolated-landmark addition.
QUICK=false;
TRIALS=2000; if QUICK, TRIALS=20; end
GAPS=[NaN,.05,.5,5];
labels={'Nominal','0.05 m','0.5 m','5 m'};
master=isac.sim_config();

% Rebuild both epoch codebooks and rate target in each geometry.
outcomes=zeros(4,TRIALS,3); absolute_penalty=zeros(4,TRIALS);
for ci=1:4
    cfg=isac.sim_config();
    if ci>1
        p=cfg.q0(1:2); m2=cfg.q0(6:7); u=(m2-p)/norm(m2-p);
        cfg.q0(8:9)=m2+GAPS(ci)*u; cfg.q2(8:9)=cfg.q0(8:9);
    end
    b0=isac.sim_beams(cfg.q0,cfg); b1=isac.sim_beams(cfg.q2,cfg);
    [~,~,m0]=isac.efim(cfg.q0,b0,cfg);
    [~,~,m1]=isac.efim(cfg.q2,b1,cfg);
    Rmin=.5*isac.alloc_rate(isac.alloc_waterfill(m0.g,cfg.Emax),m0.g,cfg);
    for trial=1:TRIALS
        seed=cfg.master_seed+400000+ci*10000+trial;
        rng(seed,'twister');
        gg0.env=sqrt(cfg.env_power).*exp(2i*pi*rand(cfg.J,cfg.R));
        gg0.bs=sqrt(cfg.bs_power)*exp(2i*pi*rand(1,cfg.R));
        gg1.env=sqrt(cfg.env_power).*exp(2i*pi*rand(cfg.J,cfg.R));
        gg1.bs=sqrt(cfg.bs_power)*exp(2i*pi*rand(1,cfg.R));
        [Z0,Za0]=isac.efim(cfg.q0,b0,cfg,gg0);
        [Z1,Za1]=isac.efim(cfg.q2,b1,cfg,gg1);
        S=diag(cfg.state_scale); full=[]; additive=[];
        for r=1:cfg.R
            full=[full,S*Z0(:,:,r)*S]; %#ok<AGROW>
            additive=[additive,S*Za0(:,:,r)*S]; %#ok<AGROW>
        end
        atom_error=norm(additive-full,'fro')/norm(full,'fro');
        ejf=isac.alloc_solve(cfg.Lambda0,Z0,m0.g,Rmin,'joint',cfg);
        eja=isac.alloc_solve(cfg.Lambda0,Za0,m0.g,Rmin,'joint',cfg);
        joint_loss=logdet_gain(cfg.Lambda0,Z0,ejf)-logdet_gain(cfg.Lambda0,Z0,eja);
        [epf0,epf1]=isac.alloc_predictive(cfg.Lambda0,Z0,m0.g,Z1,m1.g, ...
                                         2,Rmin,'loc',cfg);
        [epa0,epa1]=isac.alloc_predictive(cfg.Lambda0,Za0,m0.g,Za1,m1.g, ...
                                         2,Rmin,'loc',cfg);
        pf=terminal_peb(cfg.Lambda0,Z0,epf0,Z1,epf1);
        pa=terminal_peb(cfg.Lambda0,Z0,epa0,Z1,epa1);
        absolute_penalty(ci,trial)=pa-pf;
        outcomes(ci,trial,:)=[atom_error,joint_loss,100*(pa-pf)/pf];
    end
end

means=zeros(4,3); lows=means; highs=means;
rng(master.master_seed+1300,'twister');
for ci=1:4
    values=reshape(outcomes(ci,:,:),TRIALS,3);
    draws=zeros(master.bootstrap_resamples,3);
    for b=1:master.bootstrap_resamples
        ix=randi(TRIALS,TRIALS,1);
        draws(b,:)=mean(values(ix,:),1);
    end
    means(ci,:)=mean(values,1);
    lows(ci,:)=prctile(draws,2.5,1);
    highs(ci,:)=prctile(draws,97.5,1);
    fprintf('Fig. 5 %s: atom error %.9g, conditional joint loss %.9g nat, ', ...
            labels{ci},means(ci,1),means(ci,2));
    fprintf('terminal PEB penalty %.9g m\n',mean(absolute_penalty(ci,:)));
end

col=isac.plot_style(false);
fig=figure('Color','w','Units','inches','Position',[1 1 7.16 2.55]);
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
nexttile;
errorbar(1:4,means(:,1),means(:,1)-lows(:,1),highs(:,1)-means(:,1), ...
         '-o','Color',col{1},'LineWidth',1.4,'MarkerSize',3); grid on;
set(gca,'YScale','log','XTick',1:4,'XTickLabel',labels);
xlim([.6 4.4]); title('(a)');
xlabel('Nominal map / landmark-2--3 separation');
ylabel('Relative full/additive atom error');
legend('Mean and 95% CI','Location','best');
nexttile;
errorbar(1:4,means(:,3),means(:,3)-lows(:,3),highs(:,3)-means(:,3), ...
         '-o','Color',col{2},'LineWidth',1.4,'MarkerSize',3); grid on;
set(gca,'XTick',1:4,'XTickLabel',labels); xlim([.6 4.4]); title('(b)');
xlabel('Nominal map / landmark-2--3 separation'); ylabel('Terminal PEB penalty (%)');
legend('Additive design, full evaluation','Location','best');
exportgraphics(fig,fullfile(fileparts(mfilename('fullpath')),'fig04_separability.pdf'),'ContentType','vector');

function v=logdet_gain(Lam,Z,e)
% Joint D-optimal gain against a shared prior.
v=2*sum(log(diag(chol(isac.alloc_post(Lam,Z,e)))))-2*sum(log(diag(chol(Lam))));
end

function p=terminal_peb(Lam,Z0,e0,Z1,e1)
% Two-epoch PEB, with both designs evaluated under the full atoms.
p=isac.alloc_metrics(isac.alloc_post(isac.alloc_predict( ...
    isac.alloc_post(Lam,Z0,e0),2),Z1,e1));
end
