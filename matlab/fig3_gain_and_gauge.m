% Figure 3: unknown-gain range information and the fixed-BS gauge.
cfg=isac.sim_config();
B=logspace(5,9,401)'; Gamma=100; N=128;

% Conditional range CRBs and the corrected full-atom gauge spectrum.
known=zeros(size(B)); unknown=known;
for k=1:numel(B)
    f=(-N/2:N/2-1)'*B(k)/N;
    known(k)=cfg.c/2/sqrt(8*pi^2*Gamma*mean((cfg.fc+f).^2));
    unknown(k)=cfg.c/2/sqrt(8*pi^2*Gamma*mean((f-mean(f)).^2));
end
w=isac.sim_beams(cfg.q0,cfg);
[Z,~,meta]=isac.efim(cfg.q0,w,cfg);
e=ones(cfg.R,1)*cfg.Emax/cfg.R;
env=zeros(numel(cfg.q0)); total=env;
for r=1:cfg.R
    env=env+e(r)*meta.Zsense(:,:,r);
    total=total+e(r)*Z(:,:,r);
end
S=diag(cfg.state_scale);
eigvals=[sort(eig(S*env*S)),sort(eig(S*total*S)), ...
         sort(eig(S*(total+cfg.Lambda0)*S))];
ranks=sum(eigvals>cfg.rank_tol.*max(eigvals,[],1),1);
fprintf('Fig. 3b: environment / +BS / +prior ranks: %d %d %d\n',ranks);
fprintf('Fig. 3a at 200 MHz: known/unknown gain range CRB %.9g %.9g m\n', ...
        interp1(B,known,200e6),interp1(B,unknown,200e6));

% Original palette, log axes, displayed 1e-12 eigenvalue floor.
col=isac.plot_style(true);
fig=figure('Color','w','Units','inches','Position',[1 1 7.16 2.6]);
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
nexttile;
loglog(B/1e6,known,'Color',col{1},'LineWidth',1.8); hold on;
loglog(B/1e6,unknown,'--','Color',col{2},'LineWidth',1.8);
xlim([.1 1000]); ylim([1e-5 100]); grid on;
xlabel('Occupied bandwidth (MHz)'); ylabel('Range bound (m)'); title('(a)');
legend('Known complex gain','Unknown complex gain','Location','southwest');
nexttile;
for j=1:3
    style={'-o','--s',':^'};
    semilogy(1:11,max(eigvals(:,j),1e-12),style{j}, ...
             'Color',col{j},'LineWidth',1.8,'MarkerSize',3); hold on;
end
xlim([.7 11.3]); ylim([1e-13 100]); xticks(1:2:11); grid on;
xlabel('Ordered eigenvalue index'); ylabel('Scaled information eigenvalue'); title('(b)');
text(1.2,1e-11,'Plot floor: 10^{-12}','FontSize',7);
legend('Sensing (rank 8)','+ BS (rank 10)','+ prior (rank 11)','Location','southeast');
exportgraphics(fig,fullfile(fileparts(mfilename('fullpath')),'fig02_identifiability.pdf'),'ContentType','vector');
