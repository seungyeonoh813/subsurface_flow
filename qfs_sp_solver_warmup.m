function out = qfs_sp_solver_warmup(rate_mult, dt_days, days_per_window, is_reset, map_idx, base_rate, p_soft_bar, clear_carry)
% qfs_sp_solver_warmup
% ------------------------------------------------------------------------
% Single-phase, incompressible water flow in a 128 x 128 quarter five-spot
% (MRST, ad-core). Called from Python (qfs_ppo.py) one control window at a time.
%
%   is_reset = true  : build grid / rock / fluid / wells for permeability map
%                      map_idx, store them in the MATLAB base workspace
%                      (qfs_model, qfs_state, qfs_rock, qfs_W, qfs_pcarry),
%                      return the initial state. No simulation.
%   is_reset = false : read the stored state, set the injector rate to
%                      base_rate * rate_mult, simulate one window of
%                      days_per_window days with steps of dt_days, store the
%                      new state and return the window quantities below.
%
% Wells: rate-controlled injector in cell (1,1), producer in cell (128,128)
% at a fixed bottom-hole pressure of 100 bar. Initial pressure 200 bar.
% ------------------------------------------------------------------------
% INPUTS
%   rate_mult       : injection-rate multiplier for THIS window (e.g. 0.5..1.5)
%   dt_days         : timestep length [days] inside the window (reference = 1)
%   days_per_window : window length   [days] (reference = 30)
%   is_reset        : true -> build env & return initial state (no simulation)
%   map_idx         : permeability slice index into kmap (1..n_unique), used on reset
%   base_rate       : base injection rate [m^3/day] (reference = 5)
%   p_soft_bar      : (optional) soft pressure limit P_soft [bar] used in out.pviol.
%                     Omitted/empty -> 230 bar. The Python env always passes its
%                     own value so the penalty uses the same limit as the reward.
%   clear_carry     : (optional, default false) true -> drop the trapz carry left by
%                     the warm-up windows, so the first CONTROLLED window starts its
%                     pressure penalty from a clean left boundary. The Python env sets
%                     this on its first step() after reset; warm-up prod/pviol then
%                     never leak into the reward or into J.
%
% OUTPUT (struct 'out')
%   out.pressure : final pressure field [Pa], column-major flattened (nx*ny x 1)
%   out.k_mD     : permeability field [mD], flattened
%   out.p_max    : max pressure over the window [bar]
%   out.prod     : produced volume  this window [m^3]
%   out.inj      : injected volume  this window [m^3]
%   out.pviol    : pressure-violation penalty this window: trapezoid rule over the
%                  daily samples of max(p_max - Plim, 0)^2 [bar^2 day] (see below)
%   out.inj_rate : actual injection rate this window [m^3/day]
%   out.failed   : true if the MRST nonlinear solver failed this window
%                  (state is left unchanged; QoIs are returned as 0)
% ------------------------------------------------------------------------

    % ---- constants (identical to the single-phase reference) ------------
    nx = 128; ny = 128;
    Lx = 128; Ly = 128;                 % metres (1 m cells)
    if nargin >= 7 && ~isempty(p_soft_bar)
        Pmax_limit = double(p_soft_bar);    % bar, soft limit (from caller)
    else
        Pmax_limit = 230;                   % bar, soft limit (default)
    end
    KMAP_FILE  = 'k128x128_unique.mat'; % must sit in the run directory / MATLAB path
    KMAP_VAR   = 'kmap';                % 128 x 128 x n_unique (mD), from make_unique_kmap.m

    % Floor for permeability. Some kmap slices contain a handful of
    % (near-)zero cells (e.g. slice #38 has one exact zero). With
    % incompressible flow the diagonal of the pressure equation for such
    % a cell is built ENTIRELY from transmissibilities (no accumulation
    % term to fall back on), so a zero/near-zero perm cell makes that row
    % of the Jacobian singular -> "Linear solver produced non-finite
    % values" in simulateScheduleAD. Flooring is a numerical regularization
    % (not a physics change): it keeps a cell "nearly impermeable" instead
    % of "exactly singular".
    PERM_FLOOR = 1e-2;                  % mD

    % ---- one-time setup, cached per MATLAB engine -----------------------
    persistent MRST_READY KMAP
    if isempty(MRST_READY)
        mrstModule add ad-core ad-props       % single-phase: no ad-blackoil
        MRST_READY = true;
    end
    if isempty(KMAP)
        S    = load(KMAP_FILE, KMAP_VAR);
        KMAP = S.(KMAP_VAR);
    end

    if is_reset
        % ================= RESET: build env, no simulation ===============
        G = cartGrid([nx, ny], [Lx*meter, Ly*meter]);
        G = computeGeometry(G);

        nmaps = size(KMAP, 3);
        idx   = round(map_idx);
        if ~isfinite(map_idx) || idx < 1 || idx > nmaps
            error('qfs:mapIdx', 'map_idx = %g is out of range 1..%d (%s)', ...
                  map_idx, nmaps, KMAP_FILE);
        end
        permField = KMAP(:, :, idx);                 % 128 x 128, mD

        % ---- perm floor: guard against (near-)zero cells --------------
        nbad = sum(~isfinite(permField(:))) + sum(permField(:) < PERM_FLOOR);
        if nbad > 0
            warning('qfs:perm', ...
                'slice %d: %d cell(s) non-finite or below floor -> clamped to %g mD', ...
                idx, nbad, PERM_FLOOR);
        end
        permField(~isfinite(permField)) = PERM_FLOOR;
        permField = max(permField, PERM_FLOOR);

        rock.perm = reshape(permField, [], 1) * milli * darcy;
        rock.poro = 0.2 * ones(G.cells.num, 1);

        fluid = initSimpleADIFluid('phases', 'W', ...
                                   'mu',  1*centi*poise, ...
                                   'rho', 1000*kilogram/meter^3);
        model = WaterModel(G, rock, fluid);

        state0 = initResSol(G, 200*barsa);           % single-phase, no saturation

        % Quarter five-spot well template
        injCell  = sub2ind([nx, ny], 1,  1);
        prodCell = sub2ind([nx, ny], nx, ny);
        W = [];
        W = addWell(W, G, rock, injCell,  'Type', 'rate', ...
                    'Val', base_rate * meter^3/day, 'Comp_i', 1, 'Name', 'Injector');
        W = addWell(W, G, rock, prodCell, 'Type', 'bhp', ...
                    'Val', 100 * barsa,             'Comp_i', 1, 'Name', 'Producer');

        assignin('base', 'qfs_model', model);
        assignin('base', 'qfs_state', state0);
        assignin('base', 'qfs_G',     G);
        assignin('base', 'qfs_rock',  rock);
        assignin('base', 'qfs_W',     W);
        assignin('base', 'qfs_pcarry', []);   % previous window's last viol^2 (trapz carry)

        out.pressure = state0.pressure;
        out.k_mD     = reshape(permField, [], 1);
        out.p_max    = max(state0.pressure) / barsa;
        out.prod     = 0;
        out.inj      = 0;
        out.pviol    = 0;
        out.inj_rate = 0;
        out.failed   = false;

    else
        % ================= STEP: apply multiplier, simulate 1 window =====
        model = evalin('base', 'qfs_model');
        state = evalin('base', 'qfs_state');
        rock  = evalin('base', 'qfs_rock');
        W     = evalin('base', 'qfs_W');

        % ---- guard against non-finite / non-positive actions ------------
        if ~isfinite(rate_mult) || rate_mult <= 0
            warning('qfs:badAction', 'rate_mult=%g invalid -> window skipped', rate_mult);
            out.pressure = state.pressure;
            out.k_mD     = rock.perm / (milli * darcy);
            out.p_max    = max(state.pressure) / barsa;
            out.prod     = 0;
            out.inj      = 0;
            out.pviol    = 0;
            out.inj_rate = base_rate * rate_mult;
            out.failed   = true;
            return
        end

        W(1).val = base_rate * rate_mult * meter^3/day;   % injector rate (SI)

        nsub     = max(1, round(days_per_window / dt_days));
        dt       = ones(nsub, 1) * (dt_days * day);
        schedule = simpleSchedule(dt, 'W', W);

        % ---- guarded simulation: never let a solver failure kill the
        % calling Python process. On failure, leave state unchanged and
        % report zero QoIs with out.failed = true so the Python env can end
        % the episode instead of crashing the worker.
        ok = true;
        try
            [~, wellSols, states, ~] = evalc('simulateScheduleAD(state, model, schedule)');
        catch ME
            ok = false;
            warning('qfs:solverFailed', 'simulateScheduleAD failed: %s', ME.message);
        end

        if ~ok
            out.pressure = state.pressure;
            out.k_mD     = rock.perm / (milli * darcy);
            out.p_max    = max(state.pressure) / barsa;
            out.prod     = 0;
            out.inj      = 0;
            out.pviol    = 0;
            out.inj_rate = base_rate * rate_mult;
            out.failed   = true;
            return
        end

        % ---- window QoIs (same definitions as extractQuarterFiveSpotQoI) --
        nt      = numel(states);
        q_prod  = zeros(nt, 1);
        q_inj   = zeros(nt, 1);
        p_max_t = zeros(nt, 1);
        for n = 1:nt
            p_max_t(n) = max(states{n}.pressure) / barsa;
            ws         = wellSols{n};
            q_inj(n)   =  ws(1).qWs / (meter^3/day);   % injected water
            q_prod(n)  = -ws(2).qWs / (meter^3/day);   % produced water
        end

        dtd  = dt / day;
        prod = sum(q_prod .* dtd);      % rectangle sum == reference cumProduced
        inj  = sum(q_inj  .* dtd);

        % --- pressure penalty: trapezoidal, matching the reference
        %     pressurePenalty = trapz(time, viol.^2).
        % Carry the previous window's last viol^2 so that summing the per-window
        % contributions reproduces ONE global trapz over the whole concatenated
        % series EXACTLY ( == the reference value when dt_days = 1 ).
        h     = dt_days;                          % uniform day-spacing
        viol  = max(p_max_t - Pmax_limit, 0);
        y     = viol.^2;
        m     = numel(y);
        
        % clear_carry is optional (default false)
        if nargin < 8 || isempty(clear_carry)
            clear_carry = false;
        end
        
        carry = evalin('base', 'qfs_pcarry');
        
        % First control window after the warm-up: forget the warm-up's last
        % violation value, so the warm-up does not contribute to the penalty.
        if clear_carry
            carry = [];
        end
        
        if isempty(carry)
        
            % first window: plain trapz over its own samples (no left neighbour)
            if m >= 2
                pviol = h * ( y(1)/2 + sum(y(2:end-1)) + y(end)/2 );
            else
                pviol = 0;   % lone sample; its trapezoid joins the next window
            end
        else
            % boundary trapezoid (carry -> y(1)) + interior; last half-weighted
            if m >= 2
                pviol = h * ( carry/2 + sum(y(1:end-1)) + y(end)/2 );
            else
                pviol = h * ( carry + y(1) ) / 2;
            end
        end
        assignin('base', 'qfs_pcarry', y(end));

        next_state = states{end};
        assignin('base', 'qfs_state', next_state);

        out.pressure = next_state.pressure;
        out.k_mD     = rock.perm / (milli * darcy);
        out.p_max    = max(p_max_t);
        out.prod     = prod;
        out.inj      = inj;
        out.pviol    = pviol;
        out.inj_rate = base_rate * rate_mult;
        out.failed   = false;
    end
end