class Transition:
    def __init__(self, state, action, reward, next_state, value=0):
        self.state = state
        self.action = action
        self.reward = reward
        self.value = value
        self.next_state = next_state
        self.is_end = True if next_state[0] is None else False
        self.is_end_val = 0 if next_state[0] is None else 1

    def __repr__(self):
        if type(self.next_state) is list:
            return str(self.state.shape) + ' with ' + str(self.action) + ' to ' + str(len(self.next_state)) + ' ' +  str(self.next_state[0].shape) + ' r: ' + str(self.reward) + ' v: ' + str(self.value)
        return str(self.state.shape) + ' with ' + str(self.action) + ' to ' + str(self.next_state.shape) + ' r: ' + str(self.reward) + ' v: ' + str(self.value)
