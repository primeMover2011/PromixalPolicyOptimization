import torch
import torch.nn as nn

#def init_weights(m):
#    if isinstance(m, nn.Linear):
#        nn.init.normal_(m.weight, mean=0., std=1.141)
#        nn.init.constant_(m.bias, 0.1)


class PPOActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(PPOActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        # std   = self.log_std.exp().squeeze(0).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist, value


