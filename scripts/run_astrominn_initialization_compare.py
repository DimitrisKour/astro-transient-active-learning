#!/usr/bin/env python
"""Run one prepared, checkpoint-audited initialization fit."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from al_pipeline.astronomy.initialization_compare import run_task
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan',type=Path,required=True)
p.add_argument('--task-index',type=int,required=True)
p.add_argument('--device',default='cuda')
if __name__ == '__main__':
    a=p.parse_args();print(run_task(a.plan,a.task_index,device=a.device))
