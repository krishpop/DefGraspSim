import hydra, os, yaml
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from grasp_evaluator.grasp_sim import GraspSim 

@hydra.main(config_path="cfg", config_name="config.yaml")
def run(cfg: DictConfig):
    cfg_full = OmegaConf.to_container(cfg, resolve=True)
    # print(cfg_full)
    # grasp_sim = instantiate(cfg.simulator)
    grasp_sim = GraspSim(**cfg_full['simulator'])
    grasp_sim.run_simulation()


if __name__ == "__main__":
    run()
