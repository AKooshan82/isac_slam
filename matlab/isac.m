classdef isac
% Shared physical model, numerical methods, and plotting utilities.
% Fixed beams are inputs to every information and estimation solve.
  methods (Static)
    function cfg = sim_config()
    % Fixed synthetic geometry, waveform, prior, and noise configuration.
      cfg.version = 'corrected-physical-v1';
      cfg.master_seed = 20260921;
      cfg.c = 299792458; cfg.fc = 28e9; cfg.bandwidth = 200e6;
      cfg.Nt = 8; cfg.Nr = 8; cfg.NB = 8;
      cfg.N = 128; cfg.Ncp = 100; cfg.symbols = 4; cfg.payload_symbols = 3;
      cfg.delta_f = cfg.bandwidth/cfg.N;
      cfg.f = (-cfg.N/2:cfg.N/2-1)'*cfg.delta_f;
      cfg.Tuseful = 1/cfg.delta_f; cfg.Tcp = cfg.Ncp/cfg.bandwidth;
      cfg.bs = [0;0]; cfg.bs_heading = 0;
      cfg.landmarks = [20,35;55,22;45,55;75,50];
      cfg.J = isac.rows(cfg.landmarks); cfg.R = 1+cfg.J;
      cfg.q0 = [35;20;20*pi/180;reshape(cfg.landmarks',[],1)];
      cfg.q2 = [38;42;35*pi/180;reshape(cfg.landmarks',[],1)];
      cfg.prior_std = [1.5;1.5;10*pi/180;repelem([0.2;0.8;8;1.5],2)];
      cfg.P0 = diag(cfg.prior_std.^2); cfg.Lambda0 = diag(1./cfg.prior_std.^2);
      cfg.env_power = [0.28;0.14;0.085;0.055]; cfg.bs_power = 1.2;
      cfg.noiseS = 1; cfg.noiseB = 1; cfg.Emax = 10;
      cfg.rank_tol = 1e-10;
      cfg.state_scale = [1;1;0.1;ones(2*cfg.J,1)];
      cfg.bootstrap_resamples = 2000;
      cfg.snr_db = -10:5:30;
      % Even-length Chu sequence has constant time-envelope AND flat spectrum.
      % The unitary FFT has exact Parseval energy. Its last Ncp samples are CP.
      n = (0:cfg.N-1)';
      useful_time = exp(1i*pi*n.^2/cfg.N)/sqrt(cfg.N);
      pilot_unscaled = fftshift(fft(useful_time)/sqrt(cfg.N));
      cfg.cp_useful_ratio = sum(abs(useful_time(end-cfg.Ncp+1:end)).^2);
      cfg.action_energy_factor = cfg.symbols*(1+cfg.cp_useful_ratio);
      cfg.pilot = pilot_unscaled/sqrt(cfg.action_energy_factor);
      cfg.pilot_time = useful_time/sqrt(cfg.action_energy_factor);
      cfg.pilot_useful_fraction = sum(abs(cfg.pilot).^2);
      cfg.pilot_cp_fraction = sum(abs(cfg.pilot_time(end-cfg.Ncp+1:end)).^2);
      cfg.payload_tone_energy = 1/(cfg.symbols*(cfg.N+cfg.Ncp));
      cfg.rate_prefactor = cfg.payload_symbols*cfg.N/(cfg.R*cfg.symbols*(cfg.N+cfg.Ncp));
      cfg.frame_duration = cfg.R*cfg.symbols*(cfg.Tuseful+cfg.Tcp);
    end

    function geom = sim_geometry(q,cfg)
    % Monostatic and fixed-frame BS geometric Jacobians.
    % Analytic delay/bearing Jacobians. BS AoA is in the FIXED BS frame.
      nq=numel(q); J=(nq-3)/2; maps=reshape(q(4:end),2,J); p=q(1:2);
      geom.Htau=zeros(J,nq); geom.Htheta=zeros(J,nq);
      geom.tau=zeros(J,1); geom.theta=zeros(J,1);
      for j=1:J
        d=maps(:,j)-p; r=norm(d); assert(r>1e-6,'Coincident landmark and mobile');
        u=d'/r; a=[-d(2),d(1)]/r^2; cols=3+(2*j-1:2*j);
        geom.tau(j)=2*r/cfg.c; geom.theta(j)=atan2(d(2),d(1))-q(3);
        geom.Htau(j,1:2)=-2*u/cfg.c; geom.Htau(j,cols)=2*u/cfg.c;
        geom.Htheta(j,1:2)=-a; geom.Htheta(j,3)=-1; geom.Htheta(j,cols)=a;
      end
      d=p-cfg.bs; r=norm(d); assert(r>1e-6,'Coincident BS and mobile');
      u=d'/r; a=[-d(2),d(1)]/r^2;
      geom.tauB=r/cfg.c; geom.thetaB=atan2(d(2),d(1))-cfg.bs_heading;
      geom.thetaT=atan2(-d(2),-d(1))-q(3);
      geom.HtauB=zeros(1,nq); geom.HthetaB=zeros(1,nq); geom.HthetaT=zeros(1,nq);
      geom.HtauB(1:2)=u/cfg.c; geom.HthetaB(1:2)=a;
      geom.HthetaT(1:2)=a; geom.HthetaT(3)=-1;
      geom.cp_violation=any(geom.tau>cfg.Tcp) || geom.tauB>cfg.Tcp;
      geom.max_delay=max([geom.tau;geom.tauB]);
    end

    function [a,da] = sim_ula(N,theta)
    % Unit-norm half-wavelength array steering derivative.
    % Unit-norm half-wavelength ULA, theta measured from broadside.
      n=(0:N-1)'; a=exp(1i*pi*n*sin(theta))/sqrt(N);
      if nargout>1, da=1i*pi*n*cos(theta).*a; end
    end

    function [beams,angles] = sim_beams(q,cfg)
    % Freeze these columns during an optimization/estimation solve.
      J=(numel(q)-3)/2; p=q(1:2); maps=reshape(q(4:end),2,J);
      directions=[cfg.bs-p,maps-p];
      angles=atan2(directions(2,:),directions(1,:))-q(3);
      beams=zeros(cfg.Nt,J+1);
      for r=1:J+1, beams(:,r)=isac.sim_ula(cfg.Nt,angles(r)); end
    end

    function c = sim_components(q,beams,cfg)
    % Eqs. (2)--(3): known-pilot channel response components.
    % Compact analytic response components shared by raw responses and EFIMs.
    % Observation ordering: antenna varies fastest within each frequency tone.
      c.geom=isac.sim_geometry(q,cfg); J=(numel(q)-3)/2; R=isac.columns(beams);
      c.U=zeros(cfg.Nr*cfg.N,J); c.Utau=c.U; c.Utheta=c.U;
      c.tx=zeros(J,R); c.txprime=c.tx;
      for j=1:J
        [ar,dar]=isac.sim_ula(cfg.Nr,c.geom.theta(j));
        [at,dat]=isac.sim_ula(cfg.Nt,c.geom.theta(j));
        wave=cfg.pilot.*exp(-1i*2*pi*cfg.f*c.geom.tau(j));
        grid=ar*wave.'; gridt=ar*((-1i*2*pi*cfg.f).*wave).'; grida=dar*wave.';
        c.U(:,j)=grid(:); c.Utau(:,j)=gridt(:); c.Utheta(:,j)=grida(:);
        c.tx(j,:)=at'*beams; c.txprime(j,:)=dat'*beams;
      end
      [ar,dar]=isac.sim_ula(cfg.NB,c.geom.thetaB); [at,dat]=isac.sim_ula(cfg.Nt,c.geom.thetaT);
      wave=cfg.pilot.*exp(-1i*2*pi*cfg.f*c.geom.tauB);
      grid=ar*wave.'; gridt=ar*((-1i*2*pi*cfg.f).*wave).'; grida=dar*wave.';
      c.UB=grid(:); c.UtauB=gridt(:); c.UthetaB=grida(:);
      c.txB=at'*beams; c.txprimeB=dat'*beams;
    end

    function [Y,rankG,svals] = sim_project(G,X,tol)
    % Eq. (12): orthogonal nuisance-gain projection.
    % Stable nuisance orthogonal projection; no square observation projector.
      [U,S,~]=svd(G,'econ'); svals=diag(S);
      if isempty(svals) || svals(1)==0, rankG=0; Y=X; return; end
      rankG=sum(svals>tol*svals(1)); U=U(:,1:rankG);
      Y=X-U*(U'*X);
    end

    function [G,dG,b,db,meta] = raw_response(q,beams,cfg)
    % Eqs. (2)--(3): raw pilot templates and state derivatives.
    % Unwhitened PILOT-only responses per total action energy including CP.
    % G{r}: Nr*N x J. dG{r}: Nr*N x nstate x J. b/db: BS equivalents.
    % Beams are fixed inputs: their dependence on the belief is NOT differentiated.
      c=isac.sim_components(q,beams,cfg); nq=numel(q); J=(nq-3)/2; R=isac.columns(beams);
      G=cell(1,R); dG=cell(1,R); b=cell(1,R); db=cell(1,R);
      for r=1:R
        G{r}=bsxfun(@times,c.U,c.tx(:,r).');
        dG{r}=zeros(isac.rows(c.U),nq,J);
        for j=1:J
          dt=c.Utau(:,j)*c.tx(j,r);
          da=c.Utheta(:,j)*c.tx(j,r)+c.U(:,j)*c.txprime(j,r);
          dG{r}(:,:,j)=dt*c.geom.Htau(j,:)+da*c.geom.Htheta(j,:);
        end
        b{r}=c.UB*c.txB(r);
        db{r}=c.UtauB*c.txB(r)*c.geom.HtauB+c.UthetaB*c.txB(r)*c.geom.HthetaB+ ...
              c.UB*c.txprimeB(r)*c.geom.HthetaT;
      end
      meta=c.geom; meta.beam_gain=abs(c.tx).^2; meta.bs_beam_gain=abs(c.txB).^2;
      meta.pilot_useful_fraction=cfg.pilot_useful_fraction;
    end

    function [obs,gains,meta] = raw_observe(q,e,beams,cfg,seed)
    % Received pilots live only in memory. A seed plus inputs exactly regenerates.
      [G,~,b,~,meta]=isac.raw_response(q,beams,cfg);
      R=numel(G); J=isac.columns(G{1}); obs.e=e;
      rng(seed,'twister'); gains.env=sqrt(cfg.env_power(:)).*exp(2i*pi*rand(J,R));
      gains.bs=sqrt(cfg.bs_power)*exp(2i*pi*rand(1,R));
      for r=1:R
        % Fixed observation coordinates pair noise across different allocations.
        z=randn(isac.rows(G{r}),2); nb=randn(isac.rows(b{r}),2);
        obs.env{r}=sqrt(e(r))*G{r}*gains.env(:,r)+sqrt(cfg.noiseS/2)*(z(:,1)+1i*z(:,2));
        obs.bs{r}=sqrt(e(r))*b{r}*gains.bs(r)+sqrt(cfg.noiseB/2)*(nb(:,1)+1i*nb(:,2));
      end
    end

    function out = raw_fit(mu, P, obs, beams, cfg)
    % Eq. (65): concentrated MAP with fixed LM starts and damping.
    % Concentrated action-specific complex-gain MAP; no true gains used.
    % Frozen, outcome-independent multistart/stopping rules.
      started=tic; n=numel(mu); sc=sqrt(diag(P)); W=inv(P);
      offsets=zeros(n,3); offsets(1:3,2)=[.5;-.5;.5]; offsets(:,3)=-offsets(:,2);
      best=Inf; allcost=zeros(1,3); allstatus=zeros(1,3);
      for st=1:3
        q=mu+sc.*offsets(:,st); damping=1e-3; status=0; its=0;
        [cost,grad,H,fitg]=isac.evaluate(q,mu,W,obs,beams,cfg);
        for it=1:50
          its=it; gs=sc.*grad; Hs=(sc*sc').*H;
          if norm(gs,Inf)<1e-5, status=1; break; end
          step=-(Hs+damping*diag(max(diag(Hs),1)))\gs;
          if norm(step)>2, step=step*(2/norm(step)); end
          trial=q+sc.*step;
          [ct,gt,Ht,fg]=isac.evaluate(trial,mu,W,obs,beams,cfg);
          if ct<cost
            rel=(cost-ct)/max(1,abs(cost)); q=trial; cost=ct; grad=gt; H=Ht; fitg=fg;
            damping=max(damping/3,1e-9);
            if rel<1e-9 && norm(step)<1e-5, status=1; break; end
          else
            damping=damping*10;
            if damping>1e12, status=2; break; end
          end
        end
        allcost(st)=cost; allstatus(st)=status;
        if cost<best
          best=cost; out.q=q; out.P=inv(H); out.gains=fitg;
          out.status=status; out.iterations=its; out.scaled_gradient=norm(sc.*grad,Inf);
        end
      end
      out.cost=best; out.start_costs=allcost; out.start_status=allstatus;
      out.runtime=toc(started); out.failed=(out.status~=1 || ~all(isfinite(out.q)));
    end

    function [cost,grad,H,gains]=evaluate_slow(q,mu,W,obs,beams,cfg)
      [G,dG,b,db]=isac.raw_response(q,beams,cfg);
      delta=q-mu; delta(3)=atan2(sin(delta(3)),cos(delta(3)));
      cost=.5*delta'*W*delta; grad=W*delta; H=W;
      R=numel(G); J=isac.columns(G{1}); gains.env=zeros(J,R); gains.bs=zeros(1,R);
      for r=1:R
        if obs.e(r)<1e-10, continue; end
        ss=sqrt(obs.e(r)/cfg.noiseS); sb=sqrt(obs.e(r)/cfg.noiseB);
        Ge=ss*G{r}; ye=obs.env{r}/sqrt(cfg.noiseS);
        [U,S,V]=svd(Ge,'econ'); s=diag(S); keep=s>max(s)*1e-10;
        U=U(:,keep); a=V(:,keep)*((U'*ye)./s(keep)); residual=ye-Ge*a;
        D=zeros(isac.rows(Ge),numel(q));
        for j=1:J, D=D+ss*dG{r}(:,:,j)*a(j); end
        Dp=D-U*(U'*D);
        cost=cost+real(residual'*residual); grad=grad-2*real(D'*residual);
        H=H+2*real(Dp'*Dp); gains.env(:,r)=a;
        gb=sb*b{r}; yb=obs.bs{r}/sqrt(cfg.noiseB); normb=real(gb'*gb);
        if normb>1e-24
          beta=(gb'*yb)/normb; rb=yb-gb*beta; D=sb*db{r}*beta;
          Dp=D-gb*((gb'*D)/normb);
          cost=cost+real(rb'*rb); grad=grad-2*real(D'*rb); H=H+2*real(Dp'*Dp);
          gains.bs(r)=beta;
        else
          cost=cost+real(yb'*yb);
        end
      end
      H=(H+H')/2;
    end

    function [cost,grad,H,gains]=evaluate(q,mu,W,obs,beams,cfg)
    % Exact shared-subspace reduction. Independent unconstrained action gains
    % absorb nonzero transmit scalars; observation coordinates remain unchanged.
      c=isac.sim_components(q,beams,cfg); active=find(obs.e>=1e-10);
      tx_active=abs(c.tx(:,active));
      if any(tx_active(:)<1e-9) || any(abs(c.txB(active))<1e-9)
        [cost,grad,H,gains]=isac.evaluate_slow(q,mu,W,obs,beams,cfg);return;
      end
      delta=q-mu;delta(3)=atan2(sin(delta(3)),cos(delta(3)));
      cost=.5*delta'*W*delta;grad=W*delta;H=W;
      J=isac.columns(c.U); R=isac.columns(beams); gains.env=zeros(J,R);gains.bs=zeros(1,R);
      if isempty(active),return;end
      Y=cell2mat(obs.env(active))/sqrt(cfg.noiseS);
      [U,S,V]=svd(c.U,'econ');s=diag(S);keep=s>max(s)*cfg.rank_tol;U=U(:,keep);
      aa=V(:,keep)*((U'*Y)./s(keep));res=Y-c.U*aa;
      D=[c.Utau,c.Utheta];Dp=D-U*(U'*D); Hg=[c.geom.Htau;c.geom.Htheta];
      ad=[aa;aa];z=sum(conj(ad).*(D'*res),2);
      cost=cost+sum(abs(res(:)).^2);grad=grad-2*real(Hg'*z);
      H=H+2*real(Hg'*((Dp'*Dp).*(conj(ad)*ad.'))*Hg);
      gains.env(:,active)=sqrt(cfg.noiseS)*aa./(c.tx(:,active).*sqrt(obs.e(active)'));
      Y=cell2mat(obs.bs(active))/sqrt(cfg.noiseB); ub=c.UB;nb=real(ub'*ub);
      aa=ub'*Y/nb;res=Y-ub*aa;D=[c.UtauB,c.UthetaB];Dp=D-ub*((ub'*D)/nb);
      Hg=[c.geom.HtauB;c.geom.HthetaB];ad=[aa;aa];z=sum(conj(ad).*(D'*res),2);
      cost=cost+sum(abs(res(:)).^2);grad=grad-2*real(Hg'*z);
      H=H+2*real(Hg'*((Dp'*Dp).*(conj(ad)*ad.'))*Hg);
      gains.bs(active)=sqrt(cfg.noiseB)*aa./(c.txB(active).*sqrt(obs.e(active)'));
      H=(H+H')/2;
    end

    function [h,H,w]=feature_model(q,beams,e,cfg)
    % Section VII-A: independent range/bearing feature surrogate.
    % Idealized independent range/bearing surrogate, not a raw-data extractor.
    % If requested, precision is isolated-path projected curvature at this q.
      c=isac.sim_components(q,beams,cfg); J=(numel(q)-3)/2; n=numel(q);
      h=zeros(2*(J+1),1); H=zeros(2*(J+1),n); w=h;
      h(1:2)=[c.geom.tauB*cfg.c;c.geom.thetaB];
      H(1:2,:)=[c.geom.HtauB*cfg.c;c.geom.HthetaB];
      gamma=cfg.bs_power*cfg.pilot_useful_fraction*sum(e(:)'.*abs(c.txB).^2)/cfg.noiseB;
      varf=sum(abs(cfg.pilot).^2.*(cfg.f-sum(abs(cfg.pilot).^2.*cfg.f)/cfg.pilot_useful_fraction).^2)/cfg.pilot_useful_fraction;
      w(1:2)=2*gamma*[(2*pi/cfg.c)^2*varf;pi^2*cos(c.geom.thetaB)^2*(cfg.NB^2-1)/12];
      for j=1:J
        k=2*j+(1:2); h(k)=[c.geom.tau(j)*cfg.c/2;c.geom.theta(j)];
        H(k,:)=[c.geom.Htau(j,:)*cfg.c/2;c.geom.Htheta(j,:)];
        gamma=cfg.env_power(j)*cfg.pilot_useful_fraction*sum(e(:)'.*abs(c.tx(j,:)).^2)/cfg.noiseS;
        w(k)=2*gamma*[(4*pi/cfg.c)^2*varf;pi^2*cos(c.geom.theta(j))^2*(cfg.Nr^2-1)/12];
      end
    end

    function out=feature_fit(mu,P,z,w,beams,e,cfg)
      started=tic; n=numel(mu); sc=sqrt(diag(P)); W=inv(P);
      offsets=zeros(n,3); offsets(1:3,2)=[.5;-.5;.5]; offsets(:,3)=-offsets(:,2);
      best=Inf;
      for st=1:3
        q=mu+sc.*offsets(:,st); damping=1e-3; status=0;
        [cost,grad,H]=isac.feval_at(q,mu,W,z,w,beams,e,cfg);
        for it=1:50
          gs=sc.*grad; Hs=(sc*sc').*H;
          if norm(gs,Inf)<1e-5, status=1; break; end
          step=-(Hs+damping*diag(max(diag(Hs),1)))\gs;
          if norm(step)>2, step=step*(2/norm(step)); end
          qt=q+sc.*step; [ct,gt,Ht]=isac.feval_at(qt,mu,W,z,w,beams,e,cfg);
          if ct<cost
            rel=(cost-ct)/max(1,abs(cost)); q=qt; cost=ct; grad=gt; H=Ht; damping=max(damping/3,1e-9);
            if rel<1e-9 && norm(step)<1e-5, status=1; break; end
          else
            damping=damping*10; if damping>1e12,status=2;break;end
          end
        end
        if cost<best
          best=cost; out.q=q; out.P=inv(H); out.status=status; out.iterations=it;
          out.scaled_gradient=norm(sc.*grad,Inf);
        end
      end
      out.cost=best; out.runtime=toc(started); out.failed=(out.status~=1 || ~all(isfinite(out.q)));
    end

    function [cost,grad,H]=feval_at(q,mu,W,z,w,beams,e,cfg)
      [h,A]=isac.feature_model(q,beams,e,cfg); d=q-mu; d(3)=atan2(sin(d(3)),cos(d(3)));
      r=h-z; r(2:2:end)=atan2(sin(r(2:2:end)),cos(r(2:2:end)));
      cost=.5*(d'*W*d+sum(w.*r.^2)); grad=W*d+A'*(w.*r); H=W+A'*(w.*A);
    end

    function [Z,Zadd,meta] = efim(q,beams,cfg,gains)
    % Eq. (32): analytic phase average of full nuisance-projected atoms.
    % Full and isolated-path EFIMs, with the SAME BS atom in both.
    % Omitted gains: analytic average of conditional EFIM over independent phases.
    % gains.env (J x R), gains.bs (1 x R): conditional complex-gain experiment.
    % Averaging conditional EFIM is a DESIGN SURROGATE, not a Bayesian EFIM.
      phase_average=(nargin<4 || isempty(gains));
      c=isac.sim_components(q,beams,cfg); nq=numel(q); J=(nq-3)/2; R=isac.columns(beams);
      assert(numel(cfg.env_power)==J,'cfg.env_power must match number of landmarks');
      Z=zeros(nq,nq,R); Zadd=Z; meta.Zbs=Z; meta.Zsense=Z; meta.Zsense_add=Z;
      meta.nuisance_rank=zeros(1,R); meta.nuisance_condition=zeros(1,R);
      meta.signal_overlap=zeros(1,R); meta.derivative_overlap=zeros(1,R);
      meta.g=cfg.bs_power*abs(c.txB).^2*cfg.payload_tone_energy/cfg.noiseB;
      meta.rate_prefactor=cfg.payload_symbols*cfg.N/(R*cfg.symbols*(cfg.N+cfg.Ncp));
      meta.cp_violation=c.geom.cp_violation; meta.max_delay=c.geom.max_delay;
      meta.beam_gain=abs(c.tx).^2; meta.bs_beam_gain=abs(c.txB).^2;
      for r=1:R
        G=bsxfun(@times,c.U,c.tx(:,r).');
        Us=cell(1,J); Dsum=zeros(isac.rows(G),nq);
        full=zeros(nq); isolated=full;
        [U,S,~]=svd(G,'econ'); sv=diag(S);
        if sv(1)>0, rk=sum(sv>cfg.rank_tol*sv(1)); else, rk=0; end
        U=U(:,1:rk); meta.nuisance_rank(r)=rk;
        if rk>0, meta.nuisance_condition(r)=sv(1)/sv(rk); end
        for j=1:J
          dt=c.Utau(:,j)*c.tx(j,r);
          da=c.Utheta(:,j)*c.tx(j,r)+c.U(:,j)*c.txprime(j,r);
          D=dt*c.geom.Htau(j,:)+da*c.geom.Htheta(j,:);
          Pj=isac.sim_project(G(:,j),D,cfg.rank_tol);
          if phase_average
            Pfull=D-U*(U'*D);
            full=full+2*cfg.env_power(j)*real(Pfull'*Pfull)/cfg.noiseS;
            isolated=isolated+2*cfg.env_power(j)*real(Pj'*Pj)/cfg.noiseS;
          else
            Dsum=Dsum+gains.env(j,r)*D;
            isolated=isolated+2*abs(gains.env(j,r))^2*real(Pj'*Pj)/cfg.noiseS;
          end
          % Rescale delay derivative before numerical subspace-rank decisions.
          W=[G(:,j),dt*1e-8,da]; wn=sqrt(sum(abs(W).^2,1));
          W=W(:,wn>0); wn=wn(wn>0); W=bsxfun(@rdivide,W,wn);
          [u,s,~]=svd(W,'econ'); ss=diag(s); Us{j}=u(:,ss>cfg.rank_tol*ss(1));
        end
        if ~phase_average
          Pfull=Dsum-U*(U'*Dsum); full=2*real(Pfull'*Pfull)/cfg.noiseS;
        end
        for i=1:J
          for j=i+1:J
            denom=norm(G(:,i))*norm(G(:,j));
            if denom>0, meta.signal_overlap(r)=max(meta.signal_overlap(r),abs(G(:,i)'*G(:,j))/denom); end
            meta.derivative_overlap(r)=max(meta.derivative_overlap(r),norm(Us{i}'*Us{j},2));
          end
        end
        % Fixed-frame BS closed-form projected atom; tx derivative eliminated.
        if phase_average, powerB=cfg.bs_power; else, powerB=abs(gains.bs(r))^2; end
        w=abs(cfg.pilot).^2; wf=sum(w); fmean=sum(w.*cfg.f)/wf;
        fvar=sum(w.*(cfg.f-fmean).^2)/wf;
        anglevar=pi^2*cos(c.geom.thetaB)^2*(cfg.NB^2-1)/12;
        rho=powerB*abs(c.txB(r))^2*wf/cfg.noiseB;
        HB=[c.geom.HtauB;c.geom.HthetaB];
        ZB=HB'*diag([8*pi^2*rho*fvar,2*rho*anglevar])*HB;
        meta.Zbs(:,:,r)=(ZB+ZB')/2; meta.Zsense(:,:,r)=(full+full')/2;
        meta.Zsense_add(:,:,r)=(isolated+isolated')/2;
        Z(:,:,r)=meta.Zsense(:,:,r)+meta.Zbs(:,:,r);
        Zadd(:,:,r)=meta.Zsense_add(:,:,r)+meta.Zbs(:,:,r);
      end
      meta.phase_averaged=phase_average;
    end

    function basis=model_basis(q,beams,cfg)
    % Eq. (32): full conditional-atom projected derivative cross-Grams.
    % Fixed-geometry projected derivative cross-Grams for exact gain rebuilding.
    % Does not store received samples. Phase cross terms are retained in full.
      [G,dG,b,db]=isac.raw_response(q,beams,cfg); nq=numel(q); R=numel(G); J=isac.columns(G{1});
      basis.nq=nq; basis.J=J; basis.R=R; basis.cross=zeros(nq*nq,J*J,R);
      basis.bs=zeros(nq,nq,R); basis.noiseS=cfg.noiseS; basis.noiseB=cfg.noiseB;
      for r=1:R
        projected=isac.sim_project(G{r},reshape(dG{r},isac.rows(G{r}),nq*J),cfg.rank_tol);
        for i=1:J
          Di=projected(:,(i-1)*nq+(1:nq));
          for j=1:J
            Dj=projected(:,(j-1)*nq+(1:nq)); C=Di'*Dj;
            basis.cross(:,i+(j-1)*J,r)=C(:)/cfg.noiseS;
          end
        end
        PB=isac.sim_project(b{r},db{r},cfg.rank_tol);
        basis.bs(:,:,r)=2*real(PB'*PB)/cfg.noiseB;
      end
    end

    function [Z,ZS,ZB]=model_from_basis(basis,gains,sensing_variance_factor)
    % Eq. (32): rebuild all conditional cross-path phase terms.
    % Exactly rebuild conditional full atoms, including gain cross terms.
    % Variance factor divides the fixed baseline sensing covariance, not BS.
      if nargin<3,sensing_variance_factor=1;end
      nq=basis.nq; R=basis.R; Z=zeros(nq,nq,R); ZS=Z;ZB=Z;
      for r=1:R
        a=gains.env(:,r); crossweights=conj(a)*a.';
        S=2*real(reshape(basis.cross(:,:,r)*crossweights(:),nq,nq))/sensing_variance_factor;
        ZS(:,:,r)=(S+S')/2;ZB(:,:,r)=abs(gains.bs(r))^2*basis.bs(:,:,r);
        Z(:,:,r)=ZS(:,:,r)+ZB(:,:,r);
      end
    end

    function e = alloc_waterfill(g,Emax)
      g=g(:);
      % Water filling with exact active-set search, stable at very small gains.
      [gs,idx]=sort(g,'descend'); e=zeros(size(g));
      for k=1:numel(g)
        if gs(k)<=0, break; end
        level=(Emax+sum(1./gs(1:k)))/k;
        if k==numel(g) || gs(k+1)==0 || level<=1/gs(k+1)
          e(idx(1:k))=max(0,level-1./gs(1:k)); break;
        end
      end
      if sum(e)>0, e=e*Emax/sum(e); end
    end

    function [e,fraction,feasible] = alloc_fixed(g,Rmin,cfg)
      E=cfg.Emax; r=numel(g); u=ones(r,1)*E/r;
      er=isac.alloc_waterfill(g,E); rmax=isac.alloc_rate(er,g,cfg);
      feasible=(rmax>=Rmin-1e-12);
      if ~feasible, e=er; fraction=1; return; end
      if isac.alloc_rate(u,g,cfg)>=Rmin, e=u; fraction=0; return; end
      lo=0; hi=1;
      for k=1:60
        mid=(lo+hi)/2;
        if isac.alloc_rate((1-mid)*u+mid*er,g,cfg)>=Rmin, hi=mid; else lo=mid; end
      end
      fraction=hi; e=(1-hi)*u+hi*er;
    end

    function L = alloc_post(Lam,Z,e)
    % Eq. (36): affine pose--map information update.
      L=Lam;
      for r=1:numel(e), L=L+e(r)*Z(:,:,r); end
      L=(L+L')/2;
    end

    function [peb,heb,meb,P] = alloc_metrics(L)
    % Section IV: local covariance PEB/MEB, not nonlinear estimator bounds.
      P=L\eye(isac.rows(L)); P=(P+P')/2;
      peb=sqrt(trace(P(1:2,1:2))); heb=sqrt(P(3,3));
      J=(isac.rows(L)-3)/2; meb=zeros(J,1);
      for j=1:J, ii=3+(2*j-1:2*j); meb(j)=sqrt(trace(P(ii,ii))); end
    end

    function Lnext = alloc_predict(Lplus,qstd,headingstd)
    % Section IV: frozen identity-transition static-map covariance prediction.
      if nargin<3, headingstd=20*pi/180; end
      % Static map, additive independent pose process, identity frozen transition.
      P=Lplus\eye(isac.rows(Lplus));
      P(1:3,1:3)=P(1:3,1:3)+diag([qstd^2,qstd^2,headingstd^2]);
      Lnext=P\eye(isac.rows(P)); Lnext=(Lnext+Lnext')/2;
    end

    function [Lab,checks] = alloc_ablate(Lplus,Lminus)
      A=Lplus(1:3,1:3); B=Lplus(1:3,4:end); C=Lplus(4:end,4:end);
      Cm=Lminus(4:end,4:end); J=A-B*(C\B'); U=[C\B',eye(isac.rows(C))];
      Lab=blkdiag(J,zeros(isac.rows(C)))+U'*Cm*U; Lab=(Lab+Lab')/2;
      P=Lplus\eye(isac.rows(Lplus)); Pa=Lab\eye(isac.rows(Lab));
      checks.pose_cov_abs_error=norm(P(1:3,1:3)-Pa(1:3,1:3),'fro');
      checks.cross_cov_abs_error=norm(P(1:3,4:end)-Pa(1:3,4:end),'fro');
      checks.removed_info_min_eig=min(eig((Lplus-Lab+Lplus'-Lab')/2));
      checks.identity_abs_error=norm(Lplus-Lab-U'*(C-Cm)*U,'fro');
    end

    function out=trial_metrics(est,truth,P)
      err=est-truth; err(3)=atan2(sin(err(3)),cos(err(3))); J=(numel(truth)-3)/2;
      out=[sum(err(1:2).^2),err(3)^2,sum(err(4:end).^2)/J, ...
           err'*(P\err),trace(P(1:2,1:2)),P(3,3),trace(P(4:end,4:end))/J];
    end

    function value = alloc_rate(e,g,cfg)
    % Eq. (9): ideal-CSI spectral efficiency with pilot and CP overhead.
      value = cfg.rate_prefactor * sum(log1p(g(:).*e(:)))/log(2);
    end

    function [e,cert] = alloc_solve(Lam,Z,g,Rmin,mode,cfg)
    % Joint D, position trace, target proxy or rate allocation, Section V.
    % CVX's log-det and semidefinite cones are solved by MOSEK.
      R=numel(g); n=size(Lam,1); g=g(:);
      cvx_begin quiet
        cvx_solver mosek
        variable e(R) nonnegative
        expression K(n,n)
        K=Lam;
        for r=1:R, K=K+e(r)*Z(:,:,r); end
        if strcmp(mode,'joint')
          maximize(log_det(K))
        elseif strcmp(mode,'loc')
          variable T(2,2) symmetric
          minimize(trace(T))
        elseif strcmp(mode,'target')
          expression target_value
          target_value=0;
          for j=1:(n-3)/2
            ix=3+(2*j-1:2*j);
            target_value=target_value+log_det(K(ix,ix));
          end
          maximize(target_value)
        elseif strcmp(mode,'rate')
          maximize(sum(log(1+g.*e)))
        else
          error('Unknown allocation objective');
        end
        subject to
          sum(e)==cfg.Emax;
          cfg.rate_prefactor*sum(log(1+g.*e))/log(2)>=Rmin;
          if strcmp(mode,'loc')
            U=eye(n,2);
            [K,U;U',T]>=0;
          end
      cvx_end
      cert=struct('accepted',contains(cvx_status,'Solved'), ...
                  'status',cvx_status,'objective',cvx_optval, ...
                  'rate',isac.alloc_rate(e,g,cfg), ...
                  'energy_residual',abs(sum(e)-cfg.Emax));
      if ~cert.accepted, error('CVX/MOSEK allocation failed: %s',cvx_status); end
    end

    function [e0,e1,cert] = alloc_predictive(Lam,Z0,g0,Z1,g1,qstd,Rmin,mode,cfg)
    % Fixed-model two-epoch SDP, Proposition 3; no nonlinear feedback.
      n=size(Lam,1); R=numel(g0); old=[1:3,7:n+3]; new=4:n+3;
      K0=zeros(n+3); K0(old,old)=Lam;
      Qi=diag(1./[qstd^2,qstd^2,(20*pi/180)^2]);
      K0(1:3,1:3)=K0(1:3,1:3)+Qi;
      K0(4:6,4:6)=Qi;
      K0(1:3,4:6)=-Qi; K0(4:6,1:3)=-Qi;
      cvx_begin quiet
        cvx_solver mosek
        variable e0(R) nonnegative
        variable e1(R) nonnegative
        expression K(n+3,n+3)
        K=K0;
        for r=1:R
          A=zeros(n+3); B=A;
          A(old,old)=Z0(:,:,r); B(new,new)=Z1(:,:,r);
          K=K+e0(r)*A+e1(r)*B;
        end
        if strcmp(mode,'loc')
          variable T(2,2) symmetric
          minimize(trace(T))
        elseif strcmp(mode,'joint')
          variable S(n,n) symmetric
          maximize(log_det(S))
        else
          error('Unknown predictive objective');
        end
        subject to
          sum(e0)==cfg.Emax; sum(e1)==cfg.Emax;
          cfg.rate_prefactor*sum(log(1+g0(:).*e0))/log(2)>=Rmin;
          cfg.rate_prefactor*sum(log(1+g1(:).*e1))/log(2)>=Rmin;
          if strcmp(mode,'loc')
            U=eye(n+3); U=U(:,4:5);
            [K,U;U',T]>=0;
          else
            [K(old,old),K(old,new);K(new,old),K(new,new)-S]>=0;
          end
      cvx_end
      cert=struct('accepted',contains(cvx_status,'Solved'), ...
                  'status',cvx_status,'objective',cvx_optval);
      if ~cert.accepted, error('CVX/MOSEK predictive SDP failed: %s',cvx_status); end
    end

    function [e,cert] = robust_alloc(Lam,Zscenarios,g,Rmin,cfg)
    % Finite-scenario max-min log determinant, Section V-D.
      R=numel(g); S=size(Zscenarios,4); n=size(Lam,1); g=g(:);
      cvx_begin quiet
        cvx_solver mosek
        variable e(R) nonnegative
        variable t
        expression K(n,n)
        maximize(t)
        subject to
          sum(e)==cfg.Emax;
          cfg.rate_prefactor*sum(log(1+g.*e))/log(2)>=Rmin;
          for s=1:S
            K=Lam;
            for r=1:R, K=K+e(r)*Zscenarios(:,:,r,s); end
            log_det(K)>=t;
          end
      cvx_end
      cert=struct('accepted',contains(cvx_status,'Solved'), ...
                  'status',cvx_status,'objective',cvx_optval);
      if ~cert.accepted, error('CVX/MOSEK robust allocation failed: %s',cvx_status); end
    end

    function [centre,lo,hi] = bootstrap_mean(x,seed,B)
    % Percentile CI over whole independent trials, paired when x is a difference.
      if nargin<3, B=2000; end
      N=size(x,1); rng(seed,'twister'); draws=zeros(B,size(x,2));
      for b=1:B, ix=randi(N,N,1); draws(b,:)=mean(x(ix,:),1); end
      centre=mean(x,1); lo=prctile(draws,2.5,1); hi=prctile(draws,97.5,1);
    end

    function [centre,lo,hi] = bootstrap_rmse(x,seed,B)
    % Whole-trial bootstrap of root mean squared error.
      if nargin<3, B=2000; end
      N=size(x,1); rng(seed,'twister'); draws=zeros(B,size(x,2));
      for b=1:B, ix=randi(N,N,1); draws(b,:)=sqrt(mean(x(ix,:),1)); end
      centre=sqrt(mean(x,1)); lo=prctile(draws,2.5,1); hi=prctile(draws,97.5,1);
    end

    function interval=wilson(k,N)
    % Wilson 95% interval for nonconvergence and coverage proportions.
      z=1.959963984540054; p=k/N; den=1+z^2/N;
      centre=(p+z^2/(2*N))/den;
      half=z*sqrt(p*(1-p)/N+z^2/(4*N^2))/den;
      interval=[p,centre-half,centre+half];
    end

    function colors=plot_style(diagnostic)
    % Match the two palettes used by the original publication plots.
      if nargin<1, diagnostic=false; end
      if diagnostic
        colors={'#176A9D','#D17028','#368B68','#747D86'};
      else
        colors={'#0072B2','#D55E00','#009E73','#CC79A7','#666666','#56B4E9'};
      end
    end

    function n=rows(x)
      n=size(x,1);
    end
    function n=columns(x)
      n=size(x,2);
    end
  end
end
