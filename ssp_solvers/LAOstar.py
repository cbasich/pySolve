import numpy as np

class LAOStarSolver():
	def __init__(self, mdp, dead_end_cost=1000, eps = 0.001):
		self.mdp = mdp
		self.dead_end_cost = dead_end_cost

		self.visited = []
		self.expanded = []


	def sovlve(self, state):
		total_expanded = 0
		count_expanded = 0

		error = self.dead_end_cost

		while(True):
			count_expanded = self.expand(state)
			total_expanded += count_expanded

			while count_expanded != 0:
				self.visited.clear()
				count_expanded = self.expand(state)
				total_expanded += count_expanded

			while(True):
				self.visited.clear()
				error = self.test_convergence(state)
				if error < self.eps:
					return self.get_best_action(state)
				elif error > self.dead_end_cost:
					break


	def expand(self, state):
		if state in self.visited: return 0

		count = 0
		action = self.get_best_action(state)
		if action is None:
			self.bellmanUpdate(problem, state, weight)
			return 1
		else:
			for statePrime in self.get_successors(state, action):
				count += self.expand(statePrime)

		self.bellmanUpdate(problem, state, weight)
		return count


	def test_convergence(self, state):
		error = 0.

		if state in visited:
			return 0.

		action = self.get_best_action(state)
		if action is None:
			return DEAD_END_COST + 1
		else:
			for statePrime in self.get_successors(state, action):
				error = max(error, self.test_convergence(statePrime))

		error = max(error, self.bellmanUpdate(problem, state, weight))
		if action == self.get_best_action(state):
			return error
		return DEAD_END_COST + 1