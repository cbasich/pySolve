import collections
import numpy as np

P_UCT = 5.0                 # This is a global parameter value the should probably be tuned
                            # as a "good value" will vary wildly between domains. 

class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)

class MCTSNode(object):
    def __init__(self, state, move, parent=None, running_avg=False):
        self.state = state
        self.move = move
        self.is_expanded = False
        self.is_terminal = False
        self.parent = parent
        self.number_visits = 0
        self.total_value = 0.0
        self.children = {}
        self.child_priors = np.zeros([len(self.children.keys())], dtype=np.float32)
        self.child_total_value = np.zeros([len(self.children.keys())], dtype=np.float32)
        self.child_number_visits = np.zeros([len(self.children.keys())], dtype=np.float32)

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return P_UCT * self.child_priors * np.sqrt(np.log(self.number_visits + 1) / (1 + self.child_number_visits))

    def best_child(self):
        if self.state.done:
            return None, None, None
        q = self.child_Q()
        u = self.child_U()
        values = q + u

        legal_moves = self.state.get_legal_moves
        illegal_moves = np.ones(values.shape, dtype=bool)
        illegal_moves[legal_moves] = False
        values[illegal_moves] = -np.inf
        best = np.random.choice(np.flatnonzero(np.isclose(values, values.max())))  # This should randomly draw from tied best
        return best, values[best], values

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def add_child(self, move):
        # Value will be 0 unless the game is over
        new_state = self.state.take_action(move)
        self.children[move] = UCTNode(new_state, move = move, parent=self, running_avg=self.running_avg)


class MCTSUCT():
    def __init__(self, state, num_rollouts):
        self.root = state
        self.num_rollouts = num_rollouts

    def rollout(self, node=None):
        node = self.root
        while node.is_expanded:
            node.number_visits += 1
            if node.parent is not None:
                node.parent.child_number_visits[node.move] += 1
            move, _, _ = node.best_child()
            if move is None:
                node.is_terminal = True
                break
            if move not in node.children:
                node.add_child(move)
            node = node.children[move]

        return node

    def backup(self, node, value_estimate):
        node.number_visits += 1
        node.total_value += value_estimate
        if node.parent is not None:
            node.parent.child_number_visits[node.move] += 1
            node.parent.child_total_value[node.move] += value_estimate
            node = node.parent
        while node.parent is not None:
            node.total_value += value_estimate
            node.parent.child_total_value[node.move] += value_estimate
            node = node.parent


def MCTS (state, num_rollouts):
    root = MCTSNode(state, move = None, parent = DummyNode())
    MCTS = MCTS(root)
    for _ in range(num_rollouts):
        leaf = UCT.rollout()
        child_priors, value_estimate = nn.evaluate(leaf.state)
        leaf.expand(child_priors)
        UCT.backup(leaf, value_estimate)
    return np.argmax(root.child_number_visits)