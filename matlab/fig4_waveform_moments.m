% Figure 4: within-action spectral and angular moments.
cfg=isac.sim_config();
offsets=(0:56)'; theta=15*pi/180; angle_deg=linspace(0,10,301)';

% Both 16-tone clusters remain within the declared 128-tone band.
spectral=zeros(size(offsets));
for k=1:numel(offsets)
    m=offsets(k); bins=[(-8:7)-m,(-8:7)+m]';
    f=bins*cfg.delta_f;
    spectral(k)=mean((f-mean(f)).^2)/cfg.bandwidth^2;
end
[at,dat]=isac.sim_ula(cfg.Nt,theta);
[ar,dar]=isac.sim_ula(cfg.Nr,theta);
rx=real((dar-ar*(ar'*dar))'*(dar-ar*(ar'*dar)));
angular=zeros(numel(angle_deg),4); % normalized, total, receive, transmit
for k=1:numel(angle_deg)
    off=angle_deg(k)*pi/180;
    w=[isac.sim_ula(cfg.Nt,theta-off),isac.sim_ula(cfg.Nt,theta+off)];
    c=at'*w; cp=dat'*w; power=abs(c).^2;
    gain=mean(power); weight=power/sum(power); chi=cp./c;
    tx=real(sum(weight.*abs(chi-sum(weight.*chi)).^2));
    factor=2*cfg.pilot_useful_fraction*gain;
    angular(k,:)=[rx+tx,factor*(rx+tx),factor*rx,factor*tx];
end
fprintf('Fig. 4b normalized factor %.9g -> %.9g; fixed-energy information %.9g -> %.9g\n', ...
        angular(1,1),angular(end,1),angular(1,2),angular(end,2));
fprintf('Fig. 4a maximum cluster offset/B = %.9g\n',offsets(end)/cfg.N);

col=isac.plot_style(true);
fig=figure('Color','w','Units','inches','Position',[1 1 7.16 2.6]);
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
nexttile;
plot(offsets/cfg.N,spectral,'Color',col{1},'LineWidth',1.8); hold on;
yline(.25,':','Color',col{4});
xlim([0 .45]); ylim([0 .27]); grid on;
xlabel('Cluster-center offset / bandwidth'); ylabel('Frequency variance / bandwidth^2'); title('(a)');
legend('Within-band compound action','Support upper bound','Location','northeast');
nexttile;
plot(angle_deg,angular(:,2),'Color',col{1},'LineWidth',1.8); hold on;
plot(angle_deg,angular(:,3),'--','Color',col{2},'LineWidth',1.8);
plot(angle_deg,angular(:,4),':','Color',col{3},'LineWidth',1.8);
xlim([0 10]); ylim([0 15]); grid on;
xlabel('Within-action beam offset (deg)'); ylabel('Angular information per energy'); title('(b)');
legend('Total (fixed energy)','Receive aperture','Transmit diversity','Location','west');
exportgraphics(fig,fullfile(fileparts(mfilename('fullpath')),'fig03_waveform_moments.pdf'),'ContentType','vector');
