% Figure 2: single-epoch raw-pilot MAP and idealized feature surrogate.
QUICK=false;
TRIALS=2000; if QUICK, TRIALS=20; end
cfg=isac.sim_config(); snr_db=-10:5:30;
beams=isac.sim_beams(cfg.q0,cfg); % Freeze all five nominal beams.
cfg.q0=[35;20;20*pi/180;45;55]; cfg.env_power=.085; cfg.J=1;
cfg.P0=diag([1;1;6*pi/180;1.5;1.5].^2);
mu=cfg.q0; P=cfg.P0; e=ones(5,1)*cfg.Emax/5;
[~,~,b]=isac.raw_response(mu,beams,cfg);
reference=0;
for r=1:5, reference=reference+e(r)*cfg.bs_power*sum(abs(b{r}).^2); end

% Same three-start, 50-step LM path for QUICK and full trials.
rmse=zeros(9,3,2); low=rmse; high=rmse; local=zeros(9,3);
failure=zeros(9,2); coverage=zeros(9,2); runtime=zeros(9,2);
for setting=1:9
    snr=snr_db(setting); cfg.noiseS=reference/10^(snr/10); cfg.noiseB=cfg.noiseS;
    [~,~,precision]=isac.feature_model(mu,beams,e,cfg);
    squared=zeros(TRIALS,3,2); local_cov=zeros(TRIALS,3);
    failed=false(TRIALS,2); covered=false(TRIALS,2); times=zeros(TRIALS,2);
    for trial=1:TRIALS
        seed=cfg.master_seed+20000000+setting*100000+trial*3;
        rng(seed,'twister'); truth=mu+chol(P,'lower')*randn(5,1);
        [obs,gains,meta]=isac.raw_observe(truth,e,beams,cfg,seed+1);
        raw=isac.raw_fit(mu,P,obs,beams,cfg);
        rng(seed+2,'twister'); h=isac.feature_model(truth,beams,e,cfg);
        z=h+randn(size(h))./sqrt(precision);
        feature=isac.feature_fit(mu,P,z,precision,beams,e,cfg);
        fits={raw,feature};
        for policy=1:2
            fit=fits{policy}; m=isac.trial_metrics(fit.q,truth,fit.P);
            squared(trial,:,policy)=m(1:3);
            failed(trial,policy)=fit.failed;
            covered(trial,policy)=m(4)<=chi2inv(.95,5);
            times(trial,policy)=1000*fit.runtime;
        end
        Z=isac.efim(truth,beams,cfg,gains);
        L=isac.alloc_post(inv(P),Z,e); C=inv(L);
        local_cov(trial,:)=[trace(C(1:2,1:2)),C(3,3),trace(C(4:5,4:5))];
        if meta.cp_violation, error('Trial violates the fixed CP assumption'); end
    end
    [center,lo,hi]=isac.bootstrap_rmse(reshape(squared,TRIALS,6), ...
                                      cfg.master_seed+900+setting);
    rmse(setting,:,:)=reshape(center,3,2);
    low(setting,:,:)=reshape(lo,3,2); high(setting,:,:)=reshape(hi,3,2);
    local(setting,:)=sqrt(mean(local_cov,1));
    failure(setting,:)=[mean(failed(:,1)),mean(failed(:,2))];
    coverage(setting,:)=[mean(covered(:,1)),mean(covered(:,2))];
    runtime(setting,:)=mean(times,1);
end
fprintf('Fig. 2 at 30 dB: raw/feature/local position RMSE %.9g %.9g %.9g m\n', ...
        rmse(end,1,1),rmse(end,1,2),local(end,1));
fprintf('Fig. 2 at 30 dB: raw/feature/local map RMSE %.9g %.9g %.9g m\n', ...
        rmse(end,3,1),rmse(end,3,2),local(end,3));
fprintf('Fig. 2 at 30 dB: raw/feature/local heading RMSE %.9g %.9g %.9g deg\n', ...
        [rmse(end,2,1),rmse(end,2,2),local(end,2)]*180/pi);
fprintf('Fig. 2 raw nonconvergence range %.6g to %.6g\n',min(failure(:,1)),max(failure(:,1)));
fprintf('Fig. 2 at 30 dB: raw/feature joint 95%% coverage %.6g %.6g\n',coverage(end,:));
fprintf('Fig. 2 mean raw/feature solve time ranges (ms): %.6g %.6g / %.6g %.6g\n', ...
        min(runtime(:,1)),max(runtime(:,1)),min(runtime(:,2)),max(runtime(:,2)));
disp('Fig. 2 at 30 dB: Wilson 95% intervals for raw failure and raw coverage:');
disp(isac.wilson(round(failure(end,1)*TRIALS),TRIALS));
disp(isac.wilson(round(coverage(end,1)*TRIALS),TRIALS));

col=isac.plot_style(false);
fig=figure('Color','w','Units','inches','Position',[1 1 7.16 2.55]);
tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
labels={'Position RMSE (m)','Landmark RMSE (m)','Heading RMSE (deg)'};
indices=[1,3,2]; scales=[1,1,180/pi];
for panel=1:3
    nexttile; k=indices(panel); scale=scales(panel);
    for policy=1:2
        lo=low(:,k,policy)*scale; hi=high(:,k,policy)*scale;
        fill([snr_db,fliplr(snr_db)],[lo',fliplr(hi')],col{policy}, ...
             'FaceAlpha',.18,'EdgeColor','none'); hold on;
        plot(snr_db,rmse(:,k,policy)*scale,'-o','Color',col{policy}, ...
             'LineWidth',1.4,'MarkerSize',2.5);
    end
    plot(snr_db,local(:,k)*scale,'--','Color',col{3},'LineWidth',1.4);
    set(gca,'YScale','log'); grid on; title(['(',char('a'+panel-1),')']);
    xlabel('BS integrated pilot SNR (dB)'); ylabel(labels{panel});
    legend('','Raw MAP','','Feature surrogate','Raw local covariance', ...
           'Location','best','FontSize',7);
end
exportgraphics(fig,fullfile(fileparts(mfilename('fullpath')),'fig01_estimator_bounds.pdf'),'ContentType','vector');
