% Figure 6: current and persistent value of static-map precision.
cinfo=logspace(-4,4,401)';
Q=diag([1.2^2,1.2^2,(8*pi/180)^2]);

% This is a frozen one-landmark Gaussian illustration, not estimator RMSE.
pebs=zeros(size(cinfo));
for k=1:numel(cinfo)
    [L,~]=small_map(cinfo(k));
    pebs(k)=isac.alloc_metrics(L);
end
per_decade=-gradient(pebs,log10(cinfo));
[L1,Z]=small_map(1/9); L2=L1; L2(4:5,4:5)=L2(4:5,4:5)+3*eye(2);
persistence=zeros(11,2);
for k=0:10
    persistence(k+1,:)=[isac.alloc_metrics(L1),isac.alloc_metrics(L2)];
    if k<10
        L1=predict_small(L1,Q)+Z;
        L2=predict_small(L2,Q)+Z;
    end
end
fprintf('Fig. 6a PEB endpoints %.9g -> %.9g m\n',pebs(1),pebs(end));
fprintf('Fig. 6b largest better-map minus base-map PEB %.9g m\n', ...
        max(persistence(:,2)-persistence(:,1)));

col=isac.plot_style(true);
fig=figure('Color','w','Units','inches','Position',[1 1 7.16 2.6]);
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
nexttile;
yyaxis left; semilogx(cinfo,pebs,'Color',col{1},'LineWidth',1.8);
ylim([1.4 2.2]); ylabel('Local position scale (m)');
yyaxis right; semilogx(cinfo,per_decade,'--','Color',col{2},'LineWidth',1.8);
ylim([0 .4]); ylabel('Reduction per decade (m)');
xlim([1e-4 1e4]); grid on; xlabel('Map prior precision (m^{-2})'); title('(a)');
legend('Position scale','Per-decade drop','Location','northeast');
nexttile;
plot(0:10,persistence(:,1),'-o','Color',col{1},'LineWidth',1.8,'MarkerSize',3); hold on;
plot(0:10,persistence(:,2),'--s','Color',col{2},'LineWidth',1.8,'MarkerSize',3);
xlim([0 10]); ylim([1.3 4.5]); grid on;
xlabel('Prediction / update epoch'); ylabel('Local position scale (m)'); title('(b)');
legend('Base map','Map precision + 3I','Location','northwest');
exportgraphics(fig,fullfile(fileparts(mfilename('fullpath')),'fig05_map_value.pdf'),'ContentType','vector');

function [L,Z]=small_map(c)
% Local range/bearing information for a single point landmark.
p=[35;20]; m=[45;55]; d=m-p; r=norm(d); u=d/r; up=[-u(2);u(1)];
H=[-u',0,u';-up'/r,-1,up'/r];
F=diag([1/.2^2,1/(1.5*pi/180)^2]); Z=H'*F*H;
L=blkdiag(diag(1./[1.5^2,1.5^2,(10*pi/180)^2]),c*eye(2))+Z;
end

function L=predict_small(L,Q)
% Frozen transition-information Schur complement, Section IV.
Qi=Q\eye(3); A=L(1:3,1:3); B=L(1:3,4:5); C=L(4:5,4:5); M=A+Qi;
L=[Qi-Qi*(M\Qi),Qi*(M\B);B'*(M\Qi),C-B'*(M\B)];
L=(L+L')/2;
end
