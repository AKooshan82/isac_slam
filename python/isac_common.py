"""Shared physical model, numerical methods, and plotting utilities.

Experiment-specific calculations stay in their respective scripts. State
arrays use pose--map order [x, y, heading, landmark_1_x, landmark_1_y, ...].
"""
from pathlib import Path
import time

import cvxpy as cp
import matplotlib as mpl
import numpy as np


MASTER_SEED = 20260921


def defaults():
    """Return the shared synthetic geometry, waveform, and noise settings."""
    c = dict(speed=299792458., fc=28e9, bandwidth=200e6, Nt=8, Nr=8,
             NB=8, N=128, Ncp=100, symbols=4, payload_symbols=3,
             bs=np.zeros(2), bs_heading=0., Emax=10., noiseS=1., noiseB=1.,
             bs_power=1.2, env_power=np.array([.28, .14, .085, .055]),
             rank_tol=1e-10, bootstrap=2000, master_seed=MASTER_SEED,
             snr_db=np.arange(-10, 31, 5))
    landmarks = np.array([[20., 35.], [55., 22.], [45., 55.], [75., 50.]])
    c['landmarks'] = landmarks
    c['J'], c['R'] = 4, 5
    c['q0'] = np.r_[35., 20., np.deg2rad(20.), landmarks.ravel()]
    c['q2'] = np.r_[38., 42., np.deg2rad(35.), landmarks.ravel()]
    sd = np.r_[1.5, 1.5, np.deg2rad(10.), np.repeat([.2, .8, 8., 1.5], 2)]
    c['prior_std'] = sd
    c['P0'], c['Lambda0'] = np.diag(sd**2), np.diag(sd**-2)
    c['state_scale'] = np.r_[1., 1., .1, np.ones(8)]
    c['delta_f'] = c['bandwidth'] / c['N']
    c['f'] = np.arange(-64, 64) * c['delta_f']
    c['Tuseful'] = 1 / c['delta_f']
    c['Tcp'] = c['Ncp'] / c['bandwidth']
    n = np.arange(c['N'])
    useful = np.exp(1j * np.pi * n**2 / c['N']) / np.sqrt(c['N'])
    pilot = np.fft.fftshift(np.fft.fft(useful) / np.sqrt(c['N']))
    cp_ratio = np.sum(np.abs(useful[-c['Ncp']:])**2)
    c['pilot'] = pilot / np.sqrt(c['symbols'] * (1 + cp_ratio))
    c['pilot_useful_fraction'] = float(np.sum(np.abs(c['pilot'])**2))
    c['payload_tone_energy'] = 1 / (c['symbols'] * (c['N'] + c['Ncp']))
    c['rate_prefactor'] = c['payload_symbols'] * c['N'] / (c['R'] * c['symbols'] * (c['N'] + c['Ncp']))
    return c


def ula(n, theta):
    """Unit-norm half-wavelength ULA and its angle derivative, Section II."""
    k = np.arange(n)
    a = np.exp(1j * np.pi * k * np.sin(theta)) / np.sqrt(n)
    return a, 1j * np.pi * k * np.cos(theta) * a


def geometry(q, cfg):
    """Monostatic and fixed-BS delay/bearing Jacobians, Section II-B."""
    n, jmax = len(q), (len(q) - 3) // 2
    maps, p = q[3:].reshape(jmax, 2), q[:2]
    Htau, Htheta = np.zeros((jmax, n)), np.zeros((jmax, n))
    tau, theta = np.zeros(jmax), np.zeros(jmax)
    for j, m in enumerate(maps):
        d = m - p
        r = np.linalg.norm(d)
        if r <= 1e-6:
            raise ValueError('Coincident landmark and mobile')
        u, a = d / r, np.array([-d[1], d[0]]) / r**2
        cols = slice(3 + 2*j, 5 + 2*j)
        tau[j], theta[j] = 2*r/cfg['speed'], np.arctan2(d[1], d[0])-q[2]
        Htau[j, :2], Htau[j, cols] = -2*u/cfg['speed'], 2*u/cfg['speed']
        Htheta[j, :2], Htheta[j, 2], Htheta[j, cols] = -a, -1, a
    d = p - cfg['bs']
    r = np.linalg.norm(d)
    u, a = d/r, np.array([-d[1], d[0]])/r**2
    HtauB, HthetaB, HthetaT = np.zeros((3, n))
    HtauB[:2], HthetaB[:2], HthetaT[:2], HthetaT[2] = u/cfg['speed'], a, a, -1
    tauB = r/cfg['speed']
    return dict(tau=tau, theta=theta, Htau=Htau, Htheta=Htheta,
                tauB=tauB, thetaB=np.arctan2(d[1], d[0])-cfg['bs_heading'],
                thetaT=np.arctan2(-d[1], -d[0])-q[2], HtauB=HtauB,
                HthetaB=HthetaB, HthetaT=HthetaT,
                cp_violation=bool(np.any(tau > cfg['Tcp']) or tauB > cfg['Tcp']))


def beams(q, cfg):
    """Belief-directed fixed beams; their derivatives are excluded from EFIMs."""
    p, maps = q[:2], q[3:].reshape(-1, 2)
    directions = np.vstack((cfg['bs'], maps)) - p
    angles = np.arctan2(directions[:, 1], directions[:, 0]) - q[2]
    return np.column_stack([ula(cfg['Nt'], a)[0] for a in angles])


def components(q, w, cfg):
    """Pilot responses shared by raw MAP and nuisance-projected EFIMs."""
    geo, J, R = geometry(q, cfg), (len(q)-3)//2, w.shape[1]
    U, Ut, Ua = [np.zeros((cfg['Nr']*cfg['N'], J), complex) for _ in range(3)]
    tx, txp = np.zeros((J, R), complex), np.zeros((J, R), complex)
    for j in range(J):
        ar, dar = ula(cfg['Nr'], geo['theta'][j])
        at, dat = ula(cfg['Nt'], geo['theta'][j])
        wave = cfg['pilot'] * np.exp(-2j*np.pi*cfg['f']*geo['tau'][j])
        U[:, j] = (ar[:, None]*wave).reshape(-1, order='F')
        Ut[:, j] = (ar[:, None]*(-2j*np.pi*cfg['f']*wave)).reshape(-1, order='F')
        Ua[:, j] = (dar[:, None]*wave).reshape(-1, order='F')
        tx[j], txp[j] = at.conj()@w, dat.conj()@w
    ar, dar = ula(cfg['NB'], geo['thetaB'])
    at, dat = ula(cfg['Nt'], geo['thetaT'])
    wave = cfg['pilot'] * np.exp(-2j*np.pi*cfg['f']*geo['tauB'])
    UB = (ar[:, None]*wave).reshape(-1, order='F')
    UtB = (ar[:, None]*(-2j*np.pi*cfg['f']*wave)).reshape(-1, order='F')
    UaB = (dar[:, None]*wave).reshape(-1, order='F')
    return dict(geo=geo, U=U, Ut=Ut, Ua=Ua, tx=tx, txp=txp,
                UB=UB, UtB=UtB, UaB=UaB, txB=at.conj()@w, txpB=dat.conj()@w)


def project(G, X, tol=1e-10):
    """Nuisance orthogonal projection in the raw EFIM, Eq. (12)."""
    u, s, _ = np.linalg.svd(G, full_matrices=False)
    rank = int(np.sum(s > tol*s[0])) if len(s) and s[0] else 0
    u = u[:, :rank]
    return X - u@(u.conj().T@X), rank


def raw_response(q, w, cfg):
    """Unwhitened known-pilot responses and derivatives, Eqs. (2)--(3)."""
    c = components(q, w, cfg)
    n, J, R = len(q), c['U'].shape[1], w.shape[1]
    G, dG, b, db = [], [], [], []
    for r in range(R):
        G.append(c['U']*c['tx'][:, r])
        deriv = np.zeros((len(c['U']), n, J), complex)
        for j in range(J):
            dt = c['Ut'][:, j]*c['tx'][j, r]
            da = c['Ua'][:, j]*c['tx'][j, r] + c['U'][:, j]*c['txp'][j, r]
            deriv[:, :, j] = np.outer(dt, c['geo']['Htau'][j]) + np.outer(da, c['geo']['Htheta'][j])
        dG.append(deriv)
        b.append(c['UB']*c['txB'][r])
        db.append(np.outer(c['UtB']*c['txB'][r], c['geo']['HtauB'])
                  + np.outer(c['UaB']*c['txB'][r], c['geo']['HthetaB'])
                  + np.outer(c['UB']*c['txpB'][r], c['geo']['HthetaT']))
    return G, dG, b, db, c['geo']


def efim_atoms(q, w, cfg, gains=None):
    """Full projected and isolated-path atoms; phase average is Eq. (32).

    With gains=None, independent uniform phases are averaged analytically,
    while the full multipath nuisance projector is retained.
    """
    c = components(q, w, cfg)
    n, J, R = len(q), c['U'].shape[1], w.shape[1]
    Z, Za, Zs = [np.zeros((n, n, R)) for _ in range(3)]
    overlaps = np.zeros((R, 2))
    g = cfg['bs_power']*abs(c['txB'])**2*cfg['payload_tone_energy']/cfg['noiseB']
    for r in range(R):
        G = c['U']*c['tx'][:, r]
        u, s, _ = np.linalg.svd(G, full_matrices=False)
        rank = int(np.sum(s > cfg['rank_tol']*s[0])) if s[0] else 0
        u = u[:, :rank]
        full, isolated, Dsum, spaces = np.zeros((n, n)), np.zeros((n, n)), np.zeros((len(G), n), complex), []
        for j in range(J):
            dt = c['Ut'][:, j]*c['tx'][j, r]
            da = c['Ua'][:, j]*c['tx'][j, r] + c['U'][:, j]*c['txp'][j, r]
            D = np.outer(dt, c['geo']['Htau'][j]) + np.outer(da, c['geo']['Htheta'][j])
            Pi, _ = project(G[:, [j]], D, cfg['rank_tol'])
            if gains is None:
                Pf = D-u@(u.conj().T@D)
                full += 2*cfg['env_power'][j]*np.real(Pf.conj().T@Pf)/cfg['noiseS']
                isolated += 2*cfg['env_power'][j]*np.real(Pi.conj().T@Pi)/cfg['noiseS']
            else:
                Dsum += gains['env'][j, r]*D
                isolated += 2*abs(gains['env'][j, r])**2*np.real(Pi.conj().T@Pi)/cfg['noiseS']
            W = np.column_stack((G[:, j], dt*1e-8, da))
            norms = np.linalg.norm(W, axis=0)
            W = W[:, norms > 0]/norms[norms > 0]
            uu, ss, _ = np.linalg.svd(W, full_matrices=False)
            spaces.append(uu[:, ss > cfg['rank_tol']*ss[0]])
        if gains is not None:
            Pf = Dsum-u@(u.conj().T@Dsum)
            full = 2*np.real(Pf.conj().T@Pf)/cfg['noiseS']
        for i in range(J):
            for j in range(i+1, J):
                den = np.linalg.norm(G[:, i])*np.linalg.norm(G[:, j])
                if den:
                    overlaps[r, 0] = max(overlaps[r, 0], abs(np.vdot(G[:, i], G[:, j]))/den)
                overlaps[r, 1] = max(overlaps[r, 1], np.linalg.norm(spaces[i].conj().T@spaces[j], 2))
        powerB = cfg['bs_power'] if gains is None else abs(gains['bs'][r])**2
        pw = abs(cfg['pilot'])**2
        fm = np.dot(pw, cfg['f'])/np.sum(pw)
        fv = np.dot(pw, (cfg['f']-fm)**2)/np.sum(pw)
        av = np.pi**2*np.cos(c['geo']['thetaB'])**2*(cfg['NB']**2-1)/12
        rho = powerB*abs(c['txB'][r])**2*np.sum(pw)/cfg['noiseB']
        HB = np.vstack((c['geo']['HtauB'], c['geo']['HthetaB']))
        ZB = HB.T@np.diag([8*np.pi**2*rho*fv, 2*rho*av])@HB
        Zs[:, :, r] = (full+full.T)/2
        Z[:, :, r] = Zs[:, :, r]+(ZB+ZB.T)/2
        Za[:, :, r] = (isolated+isolated.T)/2+(ZB+ZB.T)/2
    return Z, Za, dict(g=g, Zsense=Zs, overlaps=overlaps, cp_violation=c['geo']['cp_violation'])


def model_basis(q, w, cfg):
    """Full conditional-atom cross-Grams for held-out phase trials, Eq. (32)."""
    G, dG, b, db, _ = raw_response(q, w, cfg)
    n, J, R = len(q), G[0].shape[1], len(G)
    cross = np.zeros((n, n, J, J, R), complex)
    BS = np.zeros((n, n, R))
    for r in range(R):
        D = np.column_stack([dG[r][:, :, j] for j in range(J)])
        P, _ = project(G[r], D, cfg['rank_tol'])
        for i in range(J):
            for j in range(J):
                cross[:, :, i, j, r] = P[:, i*n:(i+1)*n].conj().T@P[:, j*n:(j+1)*n]/cfg['noiseS']
        PB, _ = project(b[r][:, None], db[r], cfg['rank_tol'])
        BS[:, :, r] = 2*np.real(PB.conj().T@PB)/cfg['noiseB']
    return cross, BS


def atoms_from_basis(basis, gains, sensing_variance_factor=1.):
    """Rebuild conditional full atoms with all relative-phase cross terms."""
    cross, BS = basis
    n, _, J, _, R = cross.shape
    Zs, Zb = np.zeros((n, n, R)), np.zeros((n, n, R))
    for r in range(R):
        a = gains['env'][:, r]
        for i in range(J):
            for j in range(J):
                Zs[:, :, r] += 2*np.real(np.conj(a[i])*a[j]*cross[:, :, i, j, r])/sensing_variance_factor
        Zs[:, :, r] = (Zs[:, :, r]+Zs[:, :, r].T)/2
        Zb[:, :, r] = abs(gains['bs'][r])**2*BS[:, :, r]
    return Zs+Zb, Zs, Zb


def observe(q, e, w, cfg, seed):
    """Received pilots, Eq. (6); arrays exist in memory only."""
    G, _, b, _, geo = raw_response(q, w, cfg)
    rng = np.random.Generator(np.random.MT19937(seed))
    J, R = G[0].shape[1], len(G)
    gains = dict(env=np.sqrt(cfg['env_power'])[:, None]*np.exp(2j*np.pi*rng.random((J, R))),
                 bs=np.sqrt(cfg['bs_power'])*np.exp(2j*np.pi*rng.random(R)))
    env, bs = [], []
    for r in range(R):
        z = rng.standard_normal((G[r].shape[0], 2))
        nb = rng.standard_normal((len(b[r]), 2))
        env.append(np.sqrt(e[r])*G[r]@gains['env'][:, r]
                   + np.sqrt(cfg['noiseS']/2)*(z[:, 0]+1j*z[:, 1]))
        bs.append(np.sqrt(e[r])*b[r]*gains['bs'][r]
                  + np.sqrt(cfg['noiseB']/2)*(nb[:, 0]+1j*nb[:, 1]))
    return dict(env=env, bs=bs, e=np.asarray(e)), gains, geo


def _raw_evaluate_slow(q, mu, W, obs, w, cfg):
    """Reference projected MAP evaluator for near-zero beam responses."""
    G, dG, b, db, _ = raw_response(q, w, cfg)
    delta = q-mu
    delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
    cost, grad, H = .5*delta@W@delta, W@delta, W.copy()
    J, R = G[0].shape[1], len(G)
    gains = dict(env=np.zeros((J, R), complex), bs=np.zeros(R, complex))
    for r in range(R):
        if obs['e'][r] < 1e-10:
            continue
        ss, sb = np.sqrt(obs['e'][r]/cfg['noiseS']), np.sqrt(obs['e'][r]/cfg['noiseB'])
        Ge, ye = ss*G[r], obs['env'][r]/np.sqrt(cfg['noiseS'])
        u, s, vh = np.linalg.svd(Ge, full_matrices=False)
        keep = s > s.max()*1e-10
        u = u[:, keep]
        a = vh.conj().T[:, keep]@((u.conj().T@ye)/s[keep])
        residual = ye-Ge@a
        D = sum((ss*dG[r][:, :, j]*a[j] for j in range(J)))
        Dp = D-u@(u.conj().T@D)
        cost += np.vdot(residual, residual).real
        grad -= 2*np.real(D.conj().T@residual)
        H += 2*np.real(Dp.conj().T@Dp)
        gains['env'][:, r] = a
        gb, yb = sb*b[r], obs['bs'][r]/np.sqrt(cfg['noiseB'])
        normb = np.vdot(gb, gb).real
        if normb > 1e-24:
            beta = np.vdot(gb, yb)/normb
            rb, D = yb-gb*beta, sb*db[r]*beta
            Dp = D-np.outer(gb, gb.conj()@D/normb)
            cost += np.vdot(rb, rb).real
            grad -= 2*np.real(D.conj().T@rb)
            H += 2*np.real(Dp.conj().T@Dp)
            gains['bs'][r] = beta
        else:
            cost += np.vdot(yb, yb).real
    return cost, grad, (H+H.T)/2, gains


def _raw_evaluate(q, mu, W, obs, w, cfg):
    """Shared-subspace concentrated MAP evaluator, Eq. (65).

    This follows the production evaluator's operations and falls back to
    the full response when an active transmit scalar is near zero.
    """
    c = components(q, w, cfg)
    active = np.flatnonzero(obs['e'] >= 1e-10)
    if np.any(abs(c['tx'][:, active]) < 1e-9) or np.any(abs(c['txB'][active]) < 1e-9):
        return _raw_evaluate_slow(q, mu, W, obs, w, cfg)
    delta = q-mu
    delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
    cost, grad, H = .5*delta@W@delta, W@delta, W.copy()
    J, R = c['U'].shape[1], w.shape[1]
    gains = dict(env=np.zeros((J, R), complex), bs=np.zeros(R, complex))
    if not len(active):
        return cost, grad, H, gains

    Y = np.column_stack([obs['env'][r] for r in active])/np.sqrt(cfg['noiseS'])
    u, s, vh = np.linalg.svd(c['U'], full_matrices=False)
    keep = s > s.max()*cfg['rank_tol']
    u = u[:, keep]
    aa = vh.conj().T[:, keep]@((u.conj().T@Y)/s[keep, None])
    residual = Y-c['U']@aa
    D = np.column_stack((c['Ut'], c['Ua']))
    Dp = D-u@(u.conj().T@D)
    Hg = np.vstack((c['geo']['Htau'], c['geo']['Htheta']))
    ad = np.vstack((aa, aa))
    z = np.sum(np.conj(ad)*(D.conj().T@residual), axis=1)
    cost += np.sum(abs(residual)**2)
    grad -= 2*np.real(Hg.T@z)
    H += 2*np.real(Hg.T@((Dp.conj().T@Dp)*(np.conj(ad)@ad.T))@Hg)
    gains['env'][:, active] = (np.sqrt(cfg['noiseS'])*aa
                               /(c['tx'][:, active]*np.sqrt(obs['e'][active])[None, :]))

    Y = np.column_stack([obs['bs'][r] for r in active])/np.sqrt(cfg['noiseB'])
    ub = c['UB']
    nb = np.vdot(ub, ub).real
    aa = (ub.conj()@Y)/nb
    residual = Y-ub[:, None]*aa
    D = np.column_stack((c['UtB'], c['UaB']))
    Dp = D-np.outer(ub, (ub.conj()@D)/nb)
    Hg = np.vstack((c['geo']['HtauB'], c['geo']['HthetaB']))
    ad = np.vstack((aa, aa))
    z = np.sum(np.conj(ad)*(D.conj().T@residual), axis=1)
    cost += np.sum(abs(residual)**2)
    grad -= 2*np.real(Hg.T@z)
    H += 2*np.real(Hg.T@((Dp.conj().T@Dp)*(np.conj(ad)@ad.T))@Hg)
    gains['bs'][active] = (np.sqrt(cfg['noiseB'])*aa
                            /(c['txB'][active]*np.sqrt(obs['e'][active])))
    return cost, grad, (H+H.T)/2, gains


def lm_fit(mu, P, evaluate):
    """The original 3-start, 50-iteration LM loop, line by line.

    Stopping tolerances, damping updates, and scaled step cap are unchanged.
    This function is shared by raw MAP and the idealized feature surrogate.
    """
    started = time.perf_counter()
    n, sc, W = len(mu), np.sqrt(np.diag(P)), np.linalg.inv(P)
    offsets = np.zeros((n, 3))
    offsets[:3, 1] = [.5, -.5, .5]
    offsets[:, 2] = -offsets[:, 1]
    best, start_costs, start_status = np.inf, np.zeros(3), np.zeros(3, int)
    result = None
    for st in range(3):
        q, damping, status, its = mu+sc*offsets[:, st], 1e-3, 0, 0
        cost, grad, H, fitted = evaluate(q, mu, W)
        for it in range(1, 51):
            its = it
            gs, Hs = sc*grad, (sc[:, None]*sc[None, :])*H
            if np.linalg.norm(gs, np.inf) < 1e-5:
                status = 1
                break
            step = -np.linalg.solve(Hs+damping*np.diag(np.maximum(np.diag(Hs), 1)), gs)
            if np.linalg.norm(step) > 2:
                step *= 2/np.linalg.norm(step)
            trial = q+sc*step
            ct, gt, Ht, fg = evaluate(trial, mu, W)
            if ct < cost:
                rel = (cost-ct)/max(1, abs(cost))
                q, cost, grad, H, fitted = trial, ct, gt, Ht, fg
                damping = max(damping/3, 1e-9)
                if rel < 1e-9 and np.linalg.norm(step) < 1e-5:
                    status = 1
                    break
            else:
                damping *= 10
                if damping > 1e12:
                    status = 2
                    break
        start_costs[st], start_status[st] = cost, status
        if cost < best:
            best = cost
            result = dict(q=q, P=np.linalg.inv(H), gains=fitted, status=status,
                          iterations=its, scaled_gradient=np.linalg.norm(sc*grad, np.inf))
    result.update(cost=best, start_costs=start_costs, start_status=start_status,
                  runtime=time.perf_counter()-started)
    result['failed'] = bool(result['status'] != 1 or not np.isfinite(result['q']).all())
    return result


def raw_fit(mu, P, obs, w, cfg):
    """Concentrated raw-pilot MAP, Eq. (65)."""
    return lm_fit(mu, P, lambda q, m, W: _raw_evaluate(q, m, W, obs, w, cfg))


def feature_model(q, w, e, cfg):
    """Independent range/bearing surrogate; it does not extract raw features."""
    c = components(q, w, cfg)
    geo, J, n = c['geo'], (len(q)-3)//2, len(q)
    h, H, precision = np.zeros(2*(J+1)), np.zeros((2*(J+1), n)), np.zeros(2*(J+1))
    h[:2] = [geo['tauB']*cfg['speed'], geo['thetaB']]
    H[:2] = np.vstack((geo['HtauB']*cfg['speed'], geo['HthetaB']))
    gamma = cfg['bs_power']*cfg['pilot_useful_fraction']*np.dot(e, abs(c['txB'])**2)/cfg['noiseB']
    fmean = np.dot(abs(cfg['pilot'])**2, cfg['f'])/cfg['pilot_useful_fraction']
    varf = np.dot(abs(cfg['pilot'])**2, (cfg['f']-fmean)**2)/cfg['pilot_useful_fraction']
    precision[:2] = 2*gamma*np.array([(2*np.pi/cfg['speed'])**2*varf,
                                      np.pi**2*np.cos(geo['thetaB'])**2*(cfg['NB']**2-1)/12])
    for j in range(J):
        ix = slice(2*j+2, 2*j+4)
        h[ix] = [geo['tau'][j]*cfg['speed']/2, geo['theta'][j]]
        H[ix] = np.vstack((geo['Htau'][j]*cfg['speed']/2, geo['Htheta'][j]))
        gamma = cfg['env_power'][j]*cfg['pilot_useful_fraction']*np.dot(e, abs(c['tx'][j])**2)/cfg['noiseS']
        precision[ix] = 2*gamma*np.array([(4*np.pi/cfg['speed'])**2*varf,
                                           np.pi**2*np.cos(geo['theta'][j])**2*(cfg['Nr']**2-1)/12])
    return h, H, precision


def feature_fit(mu, P, z, precision, w, e, cfg):
    """Wrapped-residual feature MAP surrogate with the same LM loop."""
    def evaluate(q, m, W):
        h, A, _ = feature_model(q, w, e, cfg)
        d, r = q-m, h-z
        d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))
        r[1::2] = np.arctan2(np.sin(r[1::2]), np.cos(r[1::2]))
        cost = .5*(d@W@d+np.dot(precision, r*r))
        return cost, W@d+A.T@(precision*r), W+A.T@(precision[:, None]*A), None
    return lm_fit(mu, P, evaluate)


def trial_metrics(est, truth, P):
    """Squared errors, joint ellipsoid statistic, and fitted variances."""
    err = est-truth
    err[2] = np.arctan2(np.sin(err[2]), np.cos(err[2]))
    J = (len(truth)-3)//2
    return np.array([err[:2]@err[:2], err[2]**2, err[3:]@err[3:]/J,
                     err@np.linalg.solve(P, err), np.trace(P[:2, :2]),
                     P[2, 2], np.trace(P[3:, 3:])/J])


def post(Lambda, Z, e):
    """Affine pose--map information update, Eq. (36)."""
    L = Lambda+np.einsum('r,ijr->ij', e, Z)
    return (L+L.T)/2


def metrics(L):
    """Local position, heading, and landmark covariance metrics."""
    P = np.linalg.inv(L)
    J = (len(L)-3)//2
    meb = np.array([np.sqrt(np.trace(P[3+2*j:5+2*j, 3+2*j:5+2*j])) for j in range(J)])
    return np.sqrt(np.trace(P[:2, :2])), np.sqrt(P[2, 2]), meb


def rate(e, g, cfg):
    """Ideal-CSI spectral efficiency, Eq. (9)."""
    return cfg['rate_prefactor']*np.log1p(g*np.asarray(e)).sum()/np.log(2)


def waterfill(g, E):
    """Exact rate-only water filling over the finite action codebook."""
    g = np.asarray(g)
    idx = np.argsort(-g)
    gs, e = g[idx], np.zeros(len(g))
    for k in range(1, len(g)+1):
        if gs[k-1] <= 0:
            break
        level = (E+np.sum(1/gs[:k]))/k
        if k == len(g) or gs[k] == 0 or level <= 1/gs[k]:
            e[idx[:k]] = np.maximum(0, level-1/gs[:k])
            break
    return e*E/e.sum() if e.sum() else e


def feasible_uniform(g, Rmin, cfg):
    """Uniform/rate-only mixture when pure uniform violates the rate target."""
    u = np.full(len(g), cfg['Emax']/len(g))
    er = waterfill(g, cfg['Emax'])
    if rate(u, g, cfg) >= Rmin:
        return u, 0.
    lo, hi = 0., 1.
    for _ in range(60):
        mid = (lo+hi)/2
        if rate((1-mid)*u+mid*er, g, cfg) >= Rmin:
            hi = mid
        else:
            lo = mid
    return (1-hi)*u+hi*er, hi


def solve(Lambda, Z, g, Rmin, mode, cfg):
    """Finite-codebook convex allocation, Section V, via CVXPY + MOSEK."""
    R = len(g)
    e = cp.Variable(R, nonneg=True)
    K = Lambda+sum(e[r]*Z[:, :, r] for r in range(R))
    constraints = [cp.sum(e) == cfg['Emax'],
                   cfg['rate_prefactor']*cp.sum(cp.log1p(cp.multiply(g, e)))/np.log(2) >= Rmin]
    if mode == 'joint':
        objective = cp.Maximize(cp.log_det(K))
    elif mode == 'loc':
        T = cp.Variable((2, 2), symmetric=True)
        U = np.eye(len(Lambda))[:, :2]
        constraints.append(cp.bmat([[K, U], [U.T, T]]) >> 0)
        objective = cp.Minimize(cp.trace(T))
    elif mode == 'target':
        objective = cp.Maximize(sum(cp.log_det(K[3+2*j:5+2*j, 3+2*j:5+2*j])
                                    for j in range((len(Lambda)-3)//2)))
    elif mode == 'rate':
        objective = cp.Maximize(cp.sum(cp.log1p(cp.multiply(g, e))))
    else:
        raise ValueError(mode)
    cp.Problem(objective, constraints).solve(solver=cp.MOSEK, verbose=False)
    if e.value is None:
        raise RuntimeError(f'{mode} allocation failed')
    return np.asarray(e.value).ravel()


def solve_robust(Lambda, scenarios, g, Rmin, cfg):
    """Finite-scenario max-min joint log determinant, Section V-D."""
    R = len(g)
    e, t = cp.Variable(R, nonneg=True), cp.Variable()
    constraints = [cp.sum(e) == cfg['Emax'],
                   cfg['rate_prefactor']*cp.sum(cp.log1p(cp.multiply(g, e)))/np.log(2) >= Rmin]
    for Z in scenarios:
        K = Lambda+sum(e[r]*Z[:, :, r] for r in range(R))
        constraints.append(cp.log_det(K) >= t)
    cp.Problem(cp.Maximize(t), constraints).solve(solver=cp.MOSEK, verbose=False)
    if e.value is None:
        raise RuntimeError('Robust allocation failed')
    return np.asarray(e.value).ravel()


def predict(Lplus, qstd, headingstd=np.deg2rad(20.)):
    """Static-map covariance prediction under frozen identity transition."""
    P = np.linalg.inv(Lplus)
    P[:3, :3] += np.diag([qstd**2, qstd**2, headingstd**2])
    L = np.linalg.inv(P)
    return (L+L.T)/2


def solve_predictive(Lambda, Z0, g0, Z1, g1, qstd, Rmin, mode, cfg):
    """Two-epoch SDP, Proposition 3, via CVXPY + MOSEK."""
    n, R = len(Lambda), len(g0)
    old = np.r_[np.arange(3), np.arange(6, n+3)]
    new = np.arange(3, n+3)
    K0 = np.zeros((n+3, n+3))
    K0[np.ix_(old, old)] = Lambda
    Qi = np.diag([qstd**-2, qstd**-2, np.deg2rad(20.)**-2])
    K0[:3, :3] += Qi
    K0[3:6, 3:6] += Qi
    K0[:3, 3:6] -= Qi
    K0[3:6, :3] -= Qi
    e0, e1 = cp.Variable(R, nonneg=True), cp.Variable(R, nonneg=True)
    K = K0.copy()
    for r in range(R):
        A, B = np.zeros_like(K0), np.zeros_like(K0)
        A[np.ix_(old, old)] = Z0[:, :, r]
        B[np.ix_(new, new)] = Z1[:, :, r]
        K = K+e0[r]*A+e1[r]*B
    constraints = []
    for e, g in [(e0, g0), (e1, g1)]:
        constraints += [cp.sum(e) == cfg['Emax'],
                        cfg['rate_prefactor']*cp.sum(cp.log1p(cp.multiply(g, e)))/np.log(2) >= Rmin]
    if mode == 'loc':
        T = cp.Variable((2, 2), symmetric=True)
        U = np.eye(n+3)[:, [3, 4]]
        constraints.append(cp.bmat([[K, U], [U.T, T]]) >> 0)
        objective = cp.Minimize(cp.trace(T))
    elif mode == 'joint':
        S = cp.Variable((n, n), symmetric=True)
        constraints.append(cp.bmat([[K[np.ix_(old, old)], K[np.ix_(old, new)]],
                                    [K[np.ix_(new, old)], K[np.ix_(new, new)]-S]]) >> 0)
        objective = cp.Maximize(cp.log_det(S))
    else:
        raise ValueError(mode)
    cp.Problem(objective, constraints).solve(solver=cp.MOSEK, verbose=False)
    if e0.value is None or e1.value is None:
        raise RuntimeError('Predictive SDP failed')
    return np.asarray(e0.value).ravel(), np.asarray(e1.value).ravel()


def ablate_map(Lplus, Lminus):
    """Conditional-map-precision reset preserving pose/cross covariance."""
    A, B, C, Cm = Lplus[:3, :3], Lplus[:3, 3:], Lplus[3:, 3:], Lminus[3:, 3:]
    J = A-B@np.linalg.solve(C, B.T)
    U = np.column_stack((np.linalg.solve(C, B.T), np.eye(len(C))))
    top = np.zeros_like(Lplus)
    top[:3, :3] = J
    L = top+U.T@Cm@U
    return (L+L.T)/2


def bootstrap_mean(x, seed, B=2000):
    """Whole-trial percentile interval; preserve paired comparisons in x."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    draws = np.array([x[rng.integers(n, size=n)].mean(axis=0) for _ in range(B)])
    return x.mean(axis=0), np.quantile(draws, .025, axis=0), np.quantile(draws, .975, axis=0)


def bootstrap_rmse(squared, seed, B=2000):
    """Whole-trial bootstrap of root mean squared error."""
    rng = np.random.default_rng(seed)
    n = len(squared)
    draws = np.array([np.sqrt(squared[rng.integers(n, size=n)].mean(axis=0)) for _ in range(B)])
    return np.sqrt(squared.mean(axis=0)), np.quantile(draws, .025, axis=0), np.quantile(draws, .975, axis=0)


def wilson(k, n):
    """Wilson 95% binomial interval for coverage and nonconvergence."""
    z = 1.959963984540054
    p, den = k/n, 1+z*z/n
    centre = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return p, centre-half, centre+half


def plot_style(diagnostic=False):
    """Original palette, Times/Helvetica text, light grid and PDF dimensions."""
    mpl.rcParams.update({'font.family': 'sans-serif' if diagnostic else 'serif',
                         'font.sans-serif': ['Helvetica', 'Arial'],
                         'font.serif': ['Times New Roman', 'Times'],
                         'font.size': 8, 'axes.linewidth': .6,
                         'grid.color': '#E6EAEE' if diagnostic else '#dddddd',
                         'grid.linewidth': .45 if diagnostic else .4,
                         'xtick.direction': 'out', 'ytick.direction': 'out',
                         'savefig.format': 'pdf'})
    return (['#176A9D', '#D17028', '#368B68', '#747D86'] if diagnostic
            else ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#666666', '#56B4E9'])


def save_pdf(fig, script, stem):
    """Save beside the script, without any archive or manifest machinery."""
    target = Path(script).with_name(stem+'.pdf')
    fig.savefig(target)
    return target
