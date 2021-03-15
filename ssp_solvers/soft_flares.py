import numpy as np

def soft_labeling_rtdp(M, L, w, eps, n, theta):
	distances = -1. * np.inf * np.ones(len(M.states))
	for i in range(n):
		state = M.init
		visited = []
		while True:
			visited.append(state)
			if state in MDP.goals:
				break
			action = M.greedy_action(state)
			state = sample_from_sigma(state, action, M.transitions, L, distances)
			if np.random.binomial(1, L(s)) == 1:
				break
		while len(visited) > 0:
			state = visited.pop(-1)
			dist = estimate_eps_distance(M, state, w, theta)
			if np.random.binomial(1, L(s)) == 0:
				break
	return M.greedy_action(M.init)


def estimate_eps_distance(M, s, w, eps, psi, t):
	_no_high_res = True
	_open = []
	_closed = []
	_all = True

	z = np.random.binomial(1, psi)
	h = (z == 0) * t + (z == 1) * np.inf

	if np.random.binomial(1, L(s)) == 1:
		_open.append((s,0))

	while len(_open) > 0:
		s_d = _open.pop(-1)
		if s_d[1] > 2 * h:
			_all = False
			continue
		_closed.append(s_d)
		if M.residual(s) > eps:
			_no_high_res = False
		a = M.greedy_action(s)
		for sp in range(len(M.states)):
			if T[s][a][sp] > 0.0:
				if (np.random.binomial(1, L(s)) == 0 or h == np.inf) and sp not in _closed:
					_open.append((sp, s_d[1] + w(s, sp)))
				elif distances[sp] != np.inf and sp not in _closed:
					_all == False

	if _no_high_res:
		for sp_d in _closed:
			if _all:
				distances[sp_d[0]] = np.inf
			elif sp_d[1] <= t:
				distances[sp_d[0]] = t - sp_d[1]
	else:
		while len(_closed) > 0:
			sp_d = _closed.pop(-1)
			M.bellman_update(state)