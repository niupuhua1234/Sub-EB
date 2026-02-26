from typing import List, Optional
import torch
from torchtyping import TensorType
from copy import  deepcopy
from src.gfn.containers import States, Trajectories
from src.gfn.envs import Env
from src.gfn.samplers.actions_samplers import ActionsSampler, BackwardActionsSampler

# Typing
StatesTensor = TensorType["n_trajectories", "state_shape", torch.float]
ActionsTensor = TensorType["n_trajectories", torch.long]
LogProbsTensor = TensorType["n_trajectories", torch.float]
DonesTensor = TensorType["n_trajectories", torch.bool]
ForwardMasksTensor = TensorType["n_trajectories", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["n_trajectories", "n_actions - 1", torch.bool]


class TrajectoriesSampler:
    def __init__(
            self,
            env: Env,
            actions_sampler: ActionsSampler,
    ):
        """Sample complete trajectories, or sub-trajectories from a given batch states, using actions_sampler.

        Args:
            env (Env): Environment to sample trajectories from.
            actions_sampler (ActionsSampler): Sampler of actions.
        """
        self.env = env
        self.actions_sampler = actions_sampler
        self.is_backward = isinstance(actions_sampler, BackwardActionsSampler)

    def sample(
            self,
            n_trajectories: Optional[int] = None,
            states: Optional[States] = None,
            is_random=False,
            fill_value=0.
    ) -> Trajectories:
        """
        Args:
            states (States,optional):      Starting states of trajectories , ``state.tensor :(n_trajectories,-1)``
            n_trajectories(int,optional):  the number of trajectories
        """
        if states is None:
            assert (n_trajectories is not None),"Either states or n_trajectories should be specified"
            if self.is_backward | is_random:
                states = self.env.reset(batch_shape=n_trajectories,random=True)
            else:
                states=  self.env.reset(batch_shape=n_trajectories,random=False)
            #If not given starting states, Create a n_traj State objects all initialized to s_0 as the starting states.
        else:
            assert (len(states.batch_shape) == 1),  "States should be a linear batch of states"
            n_trajectories = states.batch_shape[0]
            #If staring states are given, make the n_traj is conistent with the number  of starting states
        device = states.states_tensor.device

        trajectories_states: List[StatesTensor] = [states.states_tensor]    #list of T time-length +1 for s_f, each element is n_traj vector
        trajectories_fmasks: List[ForwardMasksTensor] = [states.forward_masks]
        trajectories_bmasks: List[BackwardMasksTensor] = [states.backward_masks]
        trajectories_actions: List[ActionsTensor] = []                      #list of T time-length, each element is n_traj vector, recording the actions that leads to the current states,
        trajectories_logprobs: List[LogProbsTensor] = []
        trajectories_dones = torch.zeros( n_trajectories, dtype=torch.long, device=device) # recording at which step each traj  stop, n_traj vector
        #trajectories_log_rewards = torch.zeros( n_trajectories, dtype=torch.float, device=device) # recording at which step each traj  stop, n_traj vector

        dones = states.is_initial_state if self.is_backward else states.is_sink_state
        step = 0
        #iterating one time step by step, all  trajectories to be saved end in sf
        while not all(dones):
            actions   = torch.full( (n_trajectories,),fill_value=-1,dtype=torch.long,device=device)        # action.shape= n_traj
            log_probs = torch.full((n_trajectories,), fill_value=fill_value, dtype=torch.float, device=device)
            with torch.no_grad():
                actions_log_probs, valid_actions = self.actions_sampler.sample(states[~dones])
            if torch.isnan(actions_log_probs).any(): raise ValueError("prbs is nan")
            actions[~dones] = valid_actions
            log_probs[~dones] = actions_log_probs
            trajectories_actions += [actions]
            trajectories_logprobs += [log_probs]
            ####################################
            ####################################
            new_states = self.env.backward_step(states, actions) if self.is_backward \
                else  self.env.step(states, actions)    # move forward by taking one action obtain one states object for n_traj trajectories
            new_dones = new_states.is_initial_state & ~dones if self.is_backward \
                else new_states.is_sink_state  & ~dones #  among undones in the past, the dones currently  sink_tate: val=-1
            trajectories_states += [new_states.states_tensor]
            trajectories_fmasks += [new_states.forward_masks]
            trajectories_bmasks += [new_states.backward_masks]
            #######################################
            #######################################
            #trajectories_log_rewards[new_dones]= self.env.log_reward(states[new_dones])
            step += 1
            states = new_states
            dones = dones | new_dones # {dones in the past} ∪ {new_dones currently}
            trajectories_dones[new_dones] = step

        trajectories = Trajectories(
            env=self.env,
            states=self.env.States(states_tensor=torch.stack(trajectories_states, dim=0),
                                   forward_masks=torch.stack(trajectories_fmasks,dim=0),
                                   backward_masks=torch.stack(trajectories_bmasks, dim=0)), # to simplifyied
            actions=torch.stack(trajectories_actions, dim=0) if trajectories_actions !=[] else None,
            when_is_done=trajectories_dones,
            is_backward=self.is_backward,
            log_probs=torch.stack(trajectories_logprobs, dim=0) if trajectories_logprobs !=[] else None,
            #log_rewards =trajectories_log_rewards,
            )

        return trajectories


    def sample_T(
            self,
            n_trajectories: Optional[int] = None,
            states: Optional[States] = None,
            is_random=False,
    ) -> States:
        """
        Args:
            states (States,optional):      Starting states of trajectories , ``state.tensor :(n_trajectories,-1)``
            n_trajectories(int,optional):  the number of terminating states
        """
        if states is None:
            assert (n_trajectories is not None),"Either states or n_trajectories should be specified"
            if self.is_backward | is_random:
                states = self.env.reset(batch_shape=n_trajectories,random=True)
            else:
                states=  self.env.reset(batch_shape=n_trajectories,random=False)
            #If not given starting states, Create a n_traj State objects all initialized to s_0 as the starting states.
        else:
            assert (len(states.batch_shape) == 1),  "States should be a linear batch of states"
            n_trajectories = states.batch_shape[0]
            #If staring states are given, make the n_traj is conistent with the number  of starting states
        device = states.states_tensor.device

        dones = states.is_initial_state if self.is_backward else states.is_sink_state
        trajectories_states= states.states_tensor
        step = 0
        while not all(dones):
            actions   = torch.full( (n_trajectories,),fill_value=-1,dtype=torch.long,device=device)
            with torch.no_grad():
                actions_log_probs, valid_actions = self.actions_sampler.sample(states[~dones])
            if torch.isnan(actions_log_probs).any(): raise ValueError("prbs is nan")
            actions[~dones] = valid_actions
            ####################################
            ####################################
            new_states = self.env.backward_step(states, actions) if self.is_backward \
                else  self.env.step(states, actions)    # move forward by taking one action obtain one states object for n_traj trajectories
            new_dones = new_states.is_initial_state & ~dones if self.is_backward \
                else new_states.is_sink_state  & ~dones #  among undones in the past, the dones currently  sink_tate: val=-1
            #######################################
            #######################################
            dones = dones | new_dones # {dones in the past} ∪ {new_dones currently}
            step += 1
            trajectories_states[new_dones] = states.states_tensor[new_dones]
            states = new_states
            #print('\r','sampling step....No.'+str(step),end="")
        return self.env.States(states_tensor=trajectories_states)

class CompleteTrajectoriesSampler():
    def __init__(
            self,
            env: Env,
            actions_sampler: ActionsSampler,
            bctions_sampler: BackwardActionsSampler,
    ):
        """Sample complete trajectories from a given batch states, using actions_sampler, and bctions_sampler
        trajectories_log_prob is not recorded, as a part of complete trajectories is sampled in forward way, and the rest
        is sampled in backward manner

        Args:
            env (Env): Environment to sample trajectories from.
            actions_sampler (ActionsSampler): Sampler of forward  actions.
            bctions_sampler (BackwardActionsSampler): Sampler of backward actions.
        """
        self.env = env
        self.actions_sampler = actions_sampler
        self.bctions_sampler= bctions_sampler

    def sample(self,
            n_trajectories: Optional[int] = None,
            init_states: Optional[States] = None, backward_only=False) -> Trajectories:
        """
        Args:
            states (States,optional):      Starting states of trajectories , ``state.tensor :(n_trajectories,-1)``
            n_trajectories(int,optional):  the number of trajectories
        """
        if init_states is None:
            init_states=  self.env.reset(batch_shape=n_trajectories,random=False)
            #If not given starting states, Create a n_traj State objects all initialized to s_0 as the starting states.
        else:
            assert (len(init_states.batch_shape) == 1),  "States should be a linear batch of states"
            n_trajectories = init_states.batch_shape[0]
            #If staring states are given, make the n_traj is conistent with the number  of starting states
        device         = init_states.states_tensor.device

        trajectories_states: List[StatesTensor] = [init_states.states_tensor]    #list of T time-length +1 for s_f, each element is n_traj vector
        trajectories_fmasks: List[ForwardMasksTensor] = [init_states.forward_masks]
        trajectories_bmasks: List[BackwardMasksTensor] = [init_states.backward_masks]
        trajectories_actions: List[ActionsTensor] = []                      #list of T time-length, each element is n_traj vector, recording the actions that leads to the current states,
        trajectories_logprobs: List[LogProbsTensor] = []
        trajectories_dones = torch.zeros( n_trajectories, dtype=torch.long, device=device) # recording at which step each traj  stop, n_traj vector
        #trajectories_log_rewards = torch.zeros( n_trajectories, dtype=torch.float, device=device) # recording at which step each traj  stop, n_traj vector
        #######################
        #forward sampling
        #######################
        if backward_only == False:
            step = 0
            dones = init_states.is_sink_state
            states= deepcopy(init_states)
            while not all(dones):
                actions= torch.full( (n_trajectories,),fill_value=-1,dtype=torch.long,device=device)        # action.shape= n_traj
                log_probs = torch.full((n_trajectories,), fill_value=0, dtype=torch.float, device=device)
                with torch.no_grad():
                    actions_log_probs, valid_actions = self.actions_sampler.sample(states[~dones])
                if torch.isnan(actions_log_probs).any(): raise ValueError("prbs is nan")
                actions[~dones] = valid_actions
                log_probs[~dones] = actions_log_probs
                trajectories_actions += [actions]
                trajectories_logprobs += [log_probs]
                ####################################
                ####################################
                new_states = self.env.step(states, actions)
                new_dones =  new_states.is_sink_state  & ~dones
                trajectories_states += [new_states.states_tensor]
                trajectories_fmasks += [new_states.forward_masks]
                trajectories_bmasks += [new_states.backward_masks]
                #######################################
                #trajectories_log_rewards[new_dones]= self.env.log_reward(states[new_dones])
                step += 1
                states = new_states
                dones = dones | new_dones # {dones in the past} ∪ {new_dones currently}
                trajectories_dones[new_dones] = step
        #######################
        #backward sampling
        #######################
        backward_trajectories_dones = torch.zeros(n_trajectories, dtype=torch.long, device=device)
        step = 0
        dones = init_states.is_initial_state
        states= deepcopy(init_states)
        while not all(dones):
            actions= torch.full( (n_trajectories,),fill_value=-1,dtype=torch.long,device=device)        # action.shape= n_traj
            log_probs = torch.full((n_trajectories,), fill_value=0, dtype=torch.float, device=device)
            with torch.no_grad():
                actions_log_probs, valid_actions = self.bctions_sampler.sample(states[~dones])
            if torch.isnan(actions_log_probs).any(): raise ValueError("prbs is nan")
            actions[~dones] = valid_actions
            log_probs[~dones] = actions_log_probs
            trajectories_actions.insert(0,self.env.bction2action(states,actions))
            trajectories_logprobs.insert(0,log_probs)
            ####################################
            ####################################
            new_states =  self.env.backward_step(states, actions)
            new_dones = new_states.is_initial_state & ~dones
            trajectories_states.insert(0,new_states.states_tensor)
            trajectories_fmasks.insert(0,new_states.forward_masks)
            trajectories_bmasks.insert(0,new_states.backward_masks)
            #######################################
            step += 1
            states = new_states
            dones = dones | new_dones # {dones in the past} ∪ {new_dones currently}
            backward_trajectories_dones[new_dones] = step

        indices= backward_trajectories_dones.max()-backward_trajectories_dones
        max_length =(backward_trajectories_dones+trajectories_dones+1).max() #-1 for sf

        states_tensor= torch.stack([state[torch.arange(i, i + max_length).clamp(max=len(trajectories_states)-1)]
                                    for i,state in zip(indices,torch.stack(trajectories_states, dim=1))],dim=1)
        forward_masks= torch.stack([state[torch.arange(i, i + max_length).clamp(max=len(trajectories_fmasks)-1)]
                                    for i,state in zip(indices,torch.stack(trajectories_fmasks, dim=1))],dim=1)
        backward_masks= torch.stack([state[torch.arange(i, i + max_length).clamp(max=len(trajectories_bmasks)-1)]
                                    for i,state in zip(indices, torch.stack(trajectories_bmasks, dim=1))],dim=1)
        actions= torch.stack([state[torch.arange(i, i + max_length-1).clamp(max=len(trajectories_actions)-1)]
                                    for i,state in zip(indices, torch.stack(trajectories_actions, dim=1))],dim=1) if trajectories_actions != [] else None
        log_probs= torch.stack([state[torch.arange(i, i + max_length-1).clamp(max=len(trajectories_actions)-1)]
                                    for i,state in zip(indices, torch.stack(trajectories_logprobs, dim=1))],dim=1) if trajectories_logprobs != [] else None

        trajectories = Trajectories(
            env=self.env,
            states=self.env.States(states_tensor=states_tensor,forward_masks=forward_masks,backward_masks=backward_masks),
            actions=actions,
            when_is_done=trajectories_dones+backward_trajectories_dones,
            is_backward=None,
            log_probs= log_probs,
            #log_rewards =trajectories_log_rewards,
            )
        return trajectories

