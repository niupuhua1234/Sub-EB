from src.gfn.estimators import (
    LogEdgeFlowEstimator,
    LogitPBEstimator,
    LogitPFEstimator,
    LogStateFlowEstimator,
    LogZEstimator,
)
from src.gfn.losses import (
    Loss,
    DetailedBalance,
    FlowMatching,
    TrajectoryBalance,
    SubTrajectoryBalance,
    TrajectoryRL,
    TrajectoryRLEval,
    Trajectory_TRPO
)
from src.gfn.losses import (
    Parametrization,
    Evalmetrization,
    DBParametrization,
    FMParametrization,
    PFBasedParametrization,
    SubTBParametrization,
    TBParametrization,
    RLParametrization,
)
from src.gfn.envs import Env,HyperGrid,DAG_BN,BioSeqPendEnv
from src.gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler,BackwardDiscreteActionsSampler,LocalsearchSampler
import torch
from typing import Tuple
import numpy as np
def EnvConfig(args):
    if args.Env=='HyperGrid':
        env=HyperGrid(ndim=args.ndim,
                      height=args.height,
                      R0=args.R0,R1=0.5,R2=2.0,
                      reward_cos=False,
                      preprocessor_name="KHot",)
    elif args.Env=="BayesianNetwork":
        from data_dag.factories import get_scorer
        import networkx as nx
        #########################################
        scorer, data, graph = get_scorer(args)
        graph = torch.tensor(nx.to_numpy_array(graph, nodelist=sorted(graph.nodes), weight=None))
        all_graphs = torch.tensor(np.load('data_dag/DAG-5-list.npy'))
        env=DAG_BN(ndim=int(graph.shape[-1]),all_graphs=all_graphs,true_graph=graph,
                   score=scorer,max_edges=args.max_edges,alpha=args.alpha,reward_types=args.reward_type)
    elif args.Env == "SIX6":
        env = BioSeqPendEnv(ndim=8, nbase=4,
                        oracle_path='data_bio/tfbind8/tfbind8-exact-v0-all.pkl',
                        mode_path='data_bio/tfbind8/modes_tfbind8.pkl',alpha=args.alpha, R_max=10, R_min=1e-3,name="TFbind8")#alpha=3
    elif args.Env == "qm9str":
        env = BioSeqPendEnv(ndim=5, nbase=11,
                        oracle_path='data_bio/qm9str/block_qm9str_v1_s5.pkl',
                        mode_path='data_bio/qm9str/modes_qm9str.pkl',alpha=args.alpha,R_max=10,R_min=1e-3,name="qm9str")       #alpha=5
    elif args.Env == "sehstr":
        env = BioSeqPendEnv(ndim=6, nbase=18,
                        oracle_path='data_bio/sehstr/block_18_stop6.pkl', alpha=args.alpha, R_max=10, R_min=1e-3,name="sehstr") #alpha=6
    elif args.Env == "PHO4":
        env = BioSeqPendEnv(ndim=10, nbase=4,
                        oracle_path='data_bio/tfbind10/tfbind10-exact-v0-all.pkl', alpha=args.alpha, R_max=10, R_min=0,name="TFbind10") #alpha=3
    else:
        raise "no environment supported"
    return env

def SamplerConfig(
        env: Env,
        parametrization: Parametrization,k=0.25,i=7) -> [TrajectoriesSampler,TrajectoriesSampler]:
    if isinstance(parametrization, FMParametrization):
        estimator,B_estimator  = parametrization.logF,parametrization.logit_PB
    elif isinstance(parametrization, PFBasedParametrization):
        estimator,B_estimator  = parametrization.logit_PF,parametrization.logit_PB
    else:
        raise ValueError(f"Cannot parse sampler for parametrization {parametrization}")
    actions_sampler = DiscreteActionsSampler(estimator=estimator)
    B_actions_sampler=BackwardDiscreteActionsSampler(estimator=B_estimator)

    trajectories_sampler   = TrajectoriesSampler(env=env, actions_sampler=actions_sampler)
    B_trajectories_sampler = TrajectoriesSampler(env=env, actions_sampler=B_actions_sampler)
    local_search_sampler   = LocalsearchSampler(env=env, actions_sampler=actions_sampler,
                                                bctions_sampler=B_actions_sampler,backratio=k,iterations=i)
    return trajectories_sampler,B_trajectories_sampler,local_search_sampler

def OptimConfig(parametrization: Parametrization,lr=0.001,lr_Z=0.1):
    params = [{"params":param ,"lr": lr if estimator != "logZ" else lr_Z}
              for estimator,param in parametrization.parameters.items()]
    optimizer = torch.optim.Adam(params)
    return optimizer

def RLOptimConfig(parametrization:RLParametrization|SubTBParametrization,lr=0.001,lr_Z=0.1):
    params = [{"params":param ,"lr": lr if estimator != "logZ" else lr_Z}
              for estimator,param in parametrization.parameters.items() if estimator!="logit_PB"]
    optimizer = torch.optim.Adam(params)
    B_optimizer = torch.optim.Adam(parametrization.logit_PB.parameters(),lr) if len(parametrization.logit_PB.parameters()) else None
    return optimizer, B_optimizer

def EvalOptimConfig(evalmetrization: Evalmetrization,lr_V=0.005):
    V_optimizer = torch.optim.Adam(evalmetrization.logV.parameters(), lr_V)  # V_optimizer = torch.optim.LBFGS(logV.parameters(),history_size=10, max_iter=4)
    VB_optimizer = torch.optim.Adam(evalmetrization.logVB.parameters(), lr_V) if len(evalmetrization.logVB.parameters()) else None
    return V_optimizer, VB_optimizer

def get_estimators(env:Env,
                   PB_parameterized,
                   logZ_init,
                   **GFNModuleConfig)-> Tuple[LogitPFEstimator, LogitPBEstimator,
          LogStateFlowEstimator,LogEdgeFlowEstimator,LogZEstimator]:

    logit_PF=logit_PB =logF_state=logF_edge = GFNModuleConfig
    logit_PF = LogitPFEstimator(env=env, **logit_PF)
    logit_PB = LogitPBEstimator(env=env, **logit_PB)  if PB_parameterized \
        else LogitPBEstimator(env=env,module_name= 'Uniform')
    logF_state = LogStateFlowEstimator(env=env, **logF_state)
    logF_edge  =  LogEdgeFlowEstimator(env=env, **logF_edge)
    logZ = LogZEstimator(tensor=torch.tensor(logZ_init, dtype=torch.float))
    return (logit_PF, logit_PB, logF_state,logF_edge,logZ)

def get_evaluator(env:Env,PB_parameterized,**VModuleConfig)-> Tuple[LogStateFlowEstimator,LogStateFlowEstimator]:
    logV=logVB=VModuleConfig
    logV= LogStateFlowEstimator(env=env,**logV)# **logV)
    logVB= LogStateFlowEstimator(env=env, **logVB) if PB_parameterized \
        else LogStateFlowEstimator(env=env, module_name= 'Uniform')
    return (logV, logVB)
def FMLossConfig(env:Env,args):
    _,logit_PB,_,logF_edge,_ = get_estimators(env=env,PB_parameterized=False,logZ_init=0.0, **args.GFNModuleConfig)
    parametrization = FMParametrization(logF_edge,logit_PB)
    optimizer=OptimConfig(parametrization,**args.optim)
    loss = FlowMatching(parametrization,optimizer)
    return parametrization, loss

def DBLossConfig(env:Env,args,all_sf=False):
    logit_PF,logit_PB,logF_state,_,_ = get_estimators(env=env,PB_parameterized=args.PB_parameterized,logZ_init=0.0,**args.GFNModuleConfig)
    parametrization = DBParametrization(logit_PF, logit_PB, logF_state)
    optimizer = OptimConfig(parametrization, **args.optim)
    loss = DetailedBalance(parametrization,optimizer) if not all_sf \
        else DetailedBalance(parametrization,optimizer,all_sf=True)
    return (parametrization, loss)

def SubTBLossConfig(env:Env,args):
    logit_PF, logit_PB, logF_state,_,_, = get_estimators(env=env,PB_parameterized=args.PB_parameterized,logZ_init=0.0,**args.GFNModuleConfig )
    parametrization = SubTBParametrization(logit_PF, logit_PB, logF_state)
    optimizer = RLOptimConfig(parametrization, **args.optim)
    loss = SubTrajectoryBalance(parametrization,optimizer,weighing=args.weighing,
                                log_reward_clip_min=args.log_reward_clip_min,lamb=args.lamb)
    return (parametrization, loss)


def TBLossConfig(env:Env,args,logZ_init: float = 0.0):
    logit_PF, logit_PB,_,_,logZ= get_estimators(env=env,PB_parameterized=args.PB_parameterized,logZ_init=logZ_init,**args.GFNModuleConfig)
    parametrization = TBParametrization(logit_PF, logit_PB, logZ)
    optimizer = OptimConfig(parametrization, **args.optim)
    loss = TrajectoryBalance(parametrization,optimizer,log_reward_clip_min=args.log_reward_clip_min)
    return (parametrization, loss)

def RLLossConfig(env:Env,args,logZ_init: float = 0.0,is_eval=False):
    logit_PF, logit_PB,_,_,logZ= get_estimators(env=env,PB_parameterized=args.PB_parameterized,logZ_init=logZ_init,**args.GFNModuleConfig)
    logV,logVB= get_evaluator(env=env,PB_parameterized=args.PB_parameterized,**args.VModuleConfig)
    parametrization = RLParametrization(logit_PF, logit_PB, logZ)
    evalmetrization = Evalmetrization(logV,logVB)
    optimizer   = RLOptimConfig(parametrization, **args.optim)
    evaloptimizer = EvalOptimConfig(evalmetrization, **args.evaloptim)
    loss = TrajectoryRL(parametrization,optimizer,evalmetrization,evaloptimizer,lamb=args.lamb,lamda=args.lamda,
                        log_reward_clip_min=args.log_reward_clip_min,env=env) if not is_eval \
        else TrajectoryRLEval(parametrization,optimizer,evalmetrization,evaloptimizer,lamb=args.lamb,lamda=args.lamda,
                        log_reward_clip_min=args.log_reward_clip_min,env=env,weighing=args.weighing)
    return (parametrization, loss)

def TRPOLossConfig(env:Env,args,logZ_init: float = 0.0):
    logit_PF, logit_PB,_,_,logZ= get_estimators(env=env,PB_parameterized=args.PB_parameterized,logZ_init=logZ_init,**args.GFNModuleConfig)
    logV,logVB= get_evaluator(env=env,PB_parameterized=args.PB_parameterized,**args.VModuleConfig)
    parametrization = RLParametrization(logit_PF, logit_PB, logZ)
    evalmetrization = Evalmetrization(logV,logVB)
    optimizer = RLOptimConfig(parametrization, **args.optim)
    evaloptimizer = EvalOptimConfig(evalmetrization, **args.evaloptim)
    loss = Trajectory_TRPO(parametrization,optimizer,evalmetrization,evaloptimizer,lamb=args.lamb,lamda=args.lamda,
                        log_reward_clip_min=args.log_reward_clip_min,env=env,weighing=args.weighing,delta=args.trpo_delta)
    return (parametrization, loss)

def Config(args):
    env = EnvConfig(args)
    if args.Loss=='FM':
        parametrization, loss =FMLossConfig(env,args)
    elif args.Loss=="DB":
        parametrization, loss =DBLossConfig(env,args)
    elif args.Loss == "TB":
        parametrization, loss = TBLossConfig(env,args)
    elif args.Loss == "Sub_TB":
        parametrization, loss = SubTBLossConfig(env,args)
    elif args.Loss == "RL":
        parametrization, loss = RLLossConfig(env,args)
    elif args.Loss == "RLEval":
        parametrization, loss = RLLossConfig(env,args,is_eval=True)
    elif args.Loss == "TRPO":
        parametrization, loss = TRPOLossConfig(env,args)
    else:
        raise 'loss function not implemented'
    return env,parametrization,loss

def args_process(args):
    args.device_str = "cpu" if not torch.cuda.is_available() else args.device_str
    # 'DB, TB, TRPO only support forward phase,  #RL support two phase and forward phase'
    if args.Loss in ['DB','FM','TB','TRPO']: args.train_mode = 'forward_phrase'
    if args.Loss in ['RL']: args.train_mode = 'two_phrase' if args.PB_parameterized else 'forward_phrase'
    if args.train_mode in ['forward_phrase']: args.replay_type ='None'
    return args