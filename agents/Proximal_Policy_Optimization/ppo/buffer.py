"""A single on-policy rollout, with distinct bootstrap and recursion masks."""
import torch


class RolloutBuffer:
    def __init__(self):
        self.rows = []
        self.consumed = False

    def add(self, obs, action, old_logp, old_value, reward, terminated, truncated, next_value):
        self.rows.append(tuple(torch.as_tensor(x).detach().clone() for x in
                               (obs, action, old_logp, old_value, reward,
                                terminated, truncated, next_value)))

    def compute_gae(self, gamma=0.99, lam=0.95):
        if not self.rows:
            raise ValueError('Empty rollout')
        (self.obs, self.actions, self.logp_old, self.values_old, self.rewards,
         self.terminated, self.truncated, self.next_values) = map(torch.stack, zip(*self.rows))
        self.advantages = torch.zeros_like(self.rewards)
        carry = torch.zeros_like(self.rewards[0])
        for t in reversed(range(len(self.rows))):
            delta = (self.rewards[t] + gamma * (~self.terminated[t]).float() *
                     self.next_values[t] - self.values_old[t])
            continuation = ~(self.terminated[t] | self.truncated[t])
            carry = delta + gamma * lam * continuation.float() * carry
            self.advantages[t] = carry
        self.returns = self.advantages + self.values_old
        return self.advantages, self.returns

    def minibatches(self, batch_size, epochs):
        if self.consumed:
            raise RuntimeError('Rollout already consumed; collect fresh transitions')
        self.consumed = True
        fields = [self.obs.flatten(0, 1), self.actions.flatten(), self.logp_old.flatten(),
                  self.values_old.flatten(), self.advantages.flatten(), self.returns.flatten()]
        fields[4] = (fields[4] - fields[4].mean()) / (fields[4].std(unbiased=False) + 1e-8)
        for _ in range(epochs):
            indices = torch.randperm(len(fields[1]), device=fields[0].device)
            for start in range(0, len(indices), batch_size):
                selection = indices[start:start + batch_size]
                yield tuple(field[selection] for field in fields)
