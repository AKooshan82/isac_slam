% Figure 8: ideal-rate frontier and fixed-model two-epoch covariance design.
cfg=isac.sim_config(); fractions=linspace(0,.95,41);
position_sd=linspace(.25,5,31); bs_powers=[1.2,.08];

% Panel (a): every point solves the same joint D-optimal problem.
w=isac.sim_beams(cfg.q0,cfg); [Z,~,m]=isac.efim(cfg.q0,w,cfg);
g=m.g; Rmax=isac.alloc_rate(isac.alloc_waterfill(g,cfg.Emax),g,cfg);
Rmin=.5*Rmax; frontier=zeros(numel(fractions),3);
for k=1:numel(fractions)
    e=isac.alloc_solve(cfg.Lambda0,Z,g,fractions(k)*Rmax,'joint',cfg);
    [peb,~,meb]=isac.alloc_metrics(isac.alloc_post(cfg.Lambda0,Z,e));
    frontier(k,:)=[fractions(k)*Rmax,peb,mean(meb)];
end

% Panel (b): same absolute rate target and frozen coefficients for all policies.
rowsout=zeros(2*numel(position_sd)*6,7); row=0;
for bi=1:2
    cc=isac.sim_config(); cc.bs_power=bs_powers(bi);
    [Z0,~,m0]=isac.efim(cc.q0,isac.sim_beams(cc.q0,cc),cc);
    [Z1,~,m1]=isac.efim(cc.q2,isac.sim_beams(cc.q2,cc),cc);
    target=.5*isac.alloc_rate(isac.alloc_waterfill(m0.g,cc.Emax),m0.g,cc);
    myopic_loc=isac.alloc_solve(cc.Lambda0,Z0,m0.g,target,'loc',cc);
    myopic_joint=isac.alloc_solve(cc.Lambda0,Z0,m0.g,target,'joint',cc);
    uniform0=isac.alloc_fixed(m0.g,target,cc);
    uniform1=isac.alloc_fixed(m1.g,target,cc);
    for qstd=position_sd
        e0=zeros(cc.R,6); e1=e0;
        e0(:,1)=myopic_loc; e0(:,2)=myopic_joint;
        e1(:,1)=isac.alloc_solve(isac.alloc_predict(isac.alloc_post(cc.Lambda0,Z0,e0(:,1)),qstd), ...
                              Z1,m1.g,target,'loc',cc);
        e1(:,2)=isac.alloc_solve(isac.alloc_predict(isac.alloc_post(cc.Lambda0,Z0,e0(:,2)),qstd), ...
                              Z1,m1.g,target,'joint',cc);
        [e0(:,3),e1(:,3)]=isac.alloc_predictive(cc.Lambda0,Z0,m0.g,Z1,m1.g, ...
                                              qstd,target,'loc',cc);
        [e0(:,4),e1(:,4)]=isac.alloc_predictive(cc.Lambda0,Z0,m0.g,Z1,m1.g, ...
                                              qstd,target,'joint',cc);
        [e0(:,5),e1(:,5)]=isac.alloc_predictive(cc.Lambda0(1:3,1:3),Z0(1:3,1:3,:), ...
                            m0.g,Z1(1:3,1:3,:),m1.g,qstd,target,'loc',cc);
        e0(:,6)=uniform0; e1(:,6)=uniform1;
        for policy=1:6
            if policy==5
                Lam=cc.Lambda0(1:3,1:3); A0=Z0(1:3,1:3,:); A1=Z1(1:3,1:3,:);
            else
                Lam=cc.Lambda0; A0=Z0; A1=Z1;
            end
            Lplus=isac.alloc_post(Lam,A0,e0(:,policy));
            Lterm=isac.alloc_post(isac.alloc_predict(Lplus,qstd),A1,e1(:,policy));
            [peb,~,meb]=isac.alloc_metrics(Lterm);
            ablation=NaN;
            if policy~=5
                Lab=isac.alloc_ablate(Lplus,Lam);
                ablation=isac.alloc_metrics(isac.alloc_post( ...
                    isac.alloc_predict(Lab,qstd),A1,e1(:,policy)))-peb;
            end
            row=row+1;
            rowsout(row,:)=[cc.bs_power,qstd,policy,peb,mean(meb), ...
                            e0(4,policy)/cc.Emax,ablation];
        end
    end
end

nominal=rowsout(rowsout(:,1)==1.2,:);
weak=rowsout(rowsout(:,1)==.08,:);
fprintf('Fig. 8a rate-only max / default half target %.9g %.9g bit/s/Hz\n',Rmax,Rmin);
fprintf('Fig. 8a PEB at zero / half / 95%% target: %.9g %.9g %.9g m\n', ...
        frontier(1,2),frontier(find(abs(fractions-.5)==min(abs(fractions-.5)),1),2),frontier(end,2));
fprintf('Fig. 8a mean MEB endpoints %.9g %.9g m\n',frontier(1,3),frontier(end,3));
fprintf('Fig. 8b nominal / weak BS predictive gain %% ranges:\n');
disp([gain_range(nominal);gain_range(weak)]);
fprintf('Fig. 8b first-epoch landmark-3 directed share range: %.9g %.9g\n', ...
        min(rowsout(rowsout(:,3)==3,6)),max(rowsout(rowsout(:,3)==3,6)));
fprintf('Fig. 8b nominal / weak map-ablation PEB increase ranges (m):\n');
disp([min(nominal(:,7),[],'omitnan'),max(nominal(:,7),[],'omitnan'); ...
      min(weak(:,7),[],'omitnan'),max(weak(:,7),[],'omitnan')]);

col=isac.plot_style(false);
fig=figure('Color','w','Units','inches','Position',[1 1 7.16 2.55]);
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
nexttile;
plot(frontier(:,1)*1000,frontier(:,2),'Color',col{1},'LineWidth',1.4); hold on;
plot(frontier(:,1)*1000,frontier(:,3),'--','Color',col{2},'LineWidth',1.4);
ylim([.7 3.35]); grid on; title('(a)');
xlabel('Ideal-CSI rate target (10^{-3} bit/s/Hz)'); ylabel('Covariance metric (m)');
legend('PEB','Mean MEB','Location','west');
nexttile;
names={'Myopic position','Myopic joint D','Terminal position P2', ...
       'Terminal joint D','Known-map oracle'};
for policy=1:5
    data=nominal(nominal(:,3)==policy,:);
    plot(data(:,2),data(:,4),'Color',col{policy},'LineWidth',1.4); hold on;
end
grid on; title('(b) nominal BS power 1.2');
xlabel('Position process SD (m)'); ylabel('Terminal PEB (m)');
legend(names,'Location','southeast','FontSize',6);
exportgraphics(fig,fullfile(fileparts(mfilename('fullpath')),'fig07_comm_predictive.pdf'),'ContentType','vector');

function range=gain_range(data)
% Relative PEB reduction against matched sequential myopic localization.
base=data(data(:,3)==1,4); pred=data(data(:,3)==3,4);
gain=100*(base-pred)./base;
range=[min(gain),max(gain)];
end
