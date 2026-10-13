#!/usr/bin/env python
"""Report existing random restarts and two matched-data initialization variants."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from al_pipeline.astronomy.initialization_reporting import report_comparison
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output-dir',type=Path)
    p.add_argument('--no-plots',action='store_true');a=p.parse_args()
    print(json.dumps(report_comparison(a.plan,a.output_dir or a.plan.parent/'report',plots=not a.no_plots),indent=2))
