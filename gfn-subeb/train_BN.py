import torch
import wandb
from datetime import  datetime
from Config import Config,SamplerConfig,args_process
from src.gfn.containers.replay_buffer import Replay_x
from src.gfn.utils import validate_dist,validate_mode

from argparse import ArgumentParser
from simple_parsing.helpers.serialization import encode
from tqdm import tqdm, trange
timestamp = datetime.now().strftime('-%Y%m%d_%H%M%S')
parser = ArgumentParser(description='DAG-GFlowNet')
parser.add_argument('--project',default='DAG_BN10_Eval')
# Environment
environment = parser.add_argument_group('Environment')
environment.add_argument('--Env',default='BayesianNetwork', choices=['BayesianNetwork'])
bn = parser.add_argument_group('Type of graph')
bn.add_argument('--graph', default='erdos_renyi_lingauss', choices=['erdos_renyi_lingauss', 'sachs_continuous'])
bn.add_argument('--prior', type=str, default='erdos_renyi', choices=['uniform', 'erdos_renyi', 'edge', 'fair'])
bn.add_argument('--prior_kwargs', type=dict, default={},help='Arguments of the prior over graphs.')
bn.add_argument('--scorer_kwargs', type=dict, default={},help='Arguments of the scorer.')
bn.add_argument('--num_samples', type=int, default=1000)
bn.add_argument('--num_variables', type=int, default=10, help='design the number of nodes in erdos graph')
bn.add_argument('--num_edges', type=int, default=10, help='design the number of edges in erdos graph')
bn.add_argument('--max_edges', type=int, default=14, help='design the maximum number of edges in generated graph')
bn.add_argument('--alpha', type=float, default=10.)
bn.add_argument('--reward_type', default='log',  choices=['log', 'vanila'])

#Model
optimization = parser.add_argument_group('Method')
optimization.add_argument("--PB_parameterized",default=False)
optimization.add_argument("--lamb",type=float,default=0.99)
optimization.add_argument("--lamda",type=float,default=0.9)
optimization.add_argument("--weighing",default='geometric_within')
optimization.add_argument("--epsilon_decay",type=float,default=0.99)
optimization.add_argument("--epsilon_start",type=float,default=0.0)
optimization.add_argument("--epsilon_end",type=float,default=0.0)
# Optimization
optimization = parser.add_argument_group('Optimization')
optimization.add_argument("--train_mode",default='forward_phrase', choices=['forward_phrase','backward_phrase','two_phrase'])
optimization.add_argument('--Loss',default='RLEval', choices=['DB','TB','RL','TRPO','Sub_TB','RLEval'])
optimization.add_argument("--seed", type=int, default=0)
optimization.add_argument("--optim",default={'lr':0.001,'lr_Z':1.0})
optimization.add_argument("--evaloptim",default={'lr_V':0.001})
optimization.add_argument("--log_reward_clip_min",type=float,default=-12.)
optimization.add_argument("--trpo_delta",type=float,default=0.01)
optimization.add_argument("--no_Z",type=bool,default=False)
optimization.add_argument("--GFNModuleConfig",default={'module_name': "NeuralNet",
                                                    'n_hidden_layers': 4,
                                                    'hidden_dim': 256})
optimization.add_argument("--VModuleConfig",default={'module_name': "NeuralNet",
                                                    'n_hidden_layers': 2,
                                                    'hidden_dim': 256})
optimization.add_argument("--batch_size", type=int, default=128)
optimization.add_argument("--n_iterations", type=int, default=2000)
optimization.add_argument("--device_str",default='cuda',choices=['cpu','cuda'])
# Replay buffer
replay = parser.add_argument_group('Replay Buffer')
replay.add_argument("--replay_buffer_size", type=int, default=int(1e10))
replay.add_argument("--replay_type", default='LocalSearch', choices=['Prioritized','LocalSearch','None'])
replay.add_argument("--ls_k",type=float,default=0.25)
replay.add_argument("--ls_i",type=int,default=7)
# Miscellaneous
misc = parser.add_argument_group('Miscellaneous')
misc.add_argument("--use_wandb", type=bool, default=False)
misc.add_argument("--validation_interval", type=int, default=10)
misc.add_argument("--validation_samples", type=int,default=128)
########################################
args = parser.parse_args()
torch.manual_seed(args.seed)
args = args_process(args)
env,parametrization,loss_fn=Config(args)
print(loss_fn)
trajectories_sampler,B_trajectories_sampler,localsearch_sampler=SamplerConfig(env,parametrization,k=args.ls_k,i=args.ls_i)
replay_x=Replay_x(env,args.replay_buffer_size)
replay_x_validation=Replay_x(env,args.replay_buffer_size)
########################################
if args.Loss in ['TRPO','RL','RLEval']:
    assert args.epsilon_start == 0.0, 'epsilon_start should be 0.0 for on-policy method!'
name = args.Loss + '-B' if args.PB_parameterized else args.Loss + '-U'

if args.use_wandb:
    wandb.init(project=args.project)
    arg_code = encode(args)
    arg_code['lamb'] = str(arg_code['lamb'])
    arg_code['lamda'] = str(arg_code['lamda'])
    wandb.config.update(arg_code)

epsilon=args.epsilon_start
states_visited = 0
for i in trange(args.n_iterations):
    training_samples = trajectories_sampler.sample(n_trajectories=args.batch_size) \
        if not args.replay_type=='LocalSearch' else localsearch_sampler.sample(n_trajectories=args.batch_size)
    replay_x.add(training_samples.last_states.states_tensor, training_samples.log_rewards)
    ##################################################################################################3
    states_visited += len(training_samples)
    epsilon = args.epsilon_end + (epsilon - args.epsilon_end) * args.epsilon_decay
    trajectories_sampler.actions_sampler.epsilon = epsilon
    training_samples.to_device(args.device_str)
    to_log = {"states_visited": states_visited}
    if args.train_mode in ['forward_phrase', 'two_phrase']:
        backward_update = args.train_mode == 'forward_phrase' and args.PB_parameterized
        loss = loss_fn.update_model(training_samples, backward_update=backward_update)
        to_log["loss"] = loss.item()
    #########################################
    if args.train_mode in ['backward_phrase','two_phrase']:
        last_states = env.States(replay_x.sample_biased(args.batch_size)) \
            if args.replay_type in ['Prioritized', 'LocalSearch'] else training_samples.last_states
        assert len(parametrization.logit_PB.parameters()), 'Backward policy is not parameterized'
        forward_update  = args.train_mode== 'backward_phrase'
        B_training_samples = B_trajectories_sampler.sample(n_trajectories=args.batch_size, states=last_states)
        B_training_samples.to_device(args.device_str)
        B_loss = loss_fn.B_update_model(B_training_samples,forward_update=forward_update)
        to_log["B_loss"] = B_loss.item()
    #
    if args.use_wandb: wandb.log(to_log, step=i)
    if (i+1) % (args.validation_interval*2) == 0 or i==0:
        validation_dist,_  = validate_dist(env,parametrization, trajectories_sampler,args.validation_samples,exact=False,B_sampler=B_trajectories_sampler)
        validation_mode    = validate_mode(env, parametrization, trajectories_sampler,replay_x)
        validation_info    =   validation_dist | validation_mode
        if args.use_wandb:
            wandb.log(validation_info, step=i)
        to_log.update(validation_info)
        tqdm.write(f"{i}: {to_log}")
    if (i+1) % (args.validation_interval*10) == 0 and i != 0:
        parametrization.save_state_dict('./scripts', '{}_{}_'.format(name,i))
        if  args.use_wandb:
            artifact = wandb.Artifact('{}-{}'.format(name,timestamp), type='model')
            artifact.add_file('./scripts/{}_{}_logit_PF.pt'.format(name,i))
            wandb.log_artifact(artifact)
if args.use_wandb: wandb.run.name=name+timestamp