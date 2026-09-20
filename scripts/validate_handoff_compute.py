"""Validate a checkpoint/evaluation and render audited existing gameplay examples on compute."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image, ImageDraw
from scripts.diagnose_failure_branches import make_env
from scripts.eval_controller_diagnostic import reset_with_provenance, x_position
from src.logs.video import save_video

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--smoke-only',action='store_true'); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 start=time.monotonic(); exp=ROOT/'outputs/controller_policy/stall-sampling-confirmation-v1-20260911'
 manifest=json.loads((exp/'manifest.json').read_text()); movies=a.out/'clips'; movies.mkdir(exist_ok=True)
 results=[]
 def rec(level,arm,i):
  f=exp/'confirmation'/level/arm/f'episode_{i:04d}.json'; return f,json.loads(f.read_text())
 rescue=next(i for i in range(100) if rec('Level1-1','greedy',i)[1]['timed_out'] and rec('Level1-1','stall_sampled',i)[1]['completed'])
 loss=next(i for i in range(100) if rec('Level1-1','greedy',i)[1]['completed'] and not rec('Level1-1','stall_sampled',i)[1]['completed'])
 death=next(i for i in range(100) if rec('Level1-1','greedy',i)[1]['died'])
 selected=[('stall_rescue','Level1-1',arm,rescue) for arm in ['greedy','stall_sampled']]+[('lost_success','Level1-1',arm,loss) for arm in ['greedy','stall_sampled']]+[('early_death','Level1-1','greedy',death),('reliable_control','Level1-3','greedy',0)]
 configs={}
 for category,level,arm,i in ([] if a.smoke_only else selected):
  if level not in configs:
   cp=Path(manifest['levels'][level]['checkpoint']); assert sha(cp)==manifest['levels'][level]['checkpoint_sha256']
   configs[level]=torch.load(cp,map_location='cpu',weights_only=False)['cfg_snapshot']
  path,r=rec(level,arm,i); trace_path=path.with_name(path.stem+'_trace.npz'); assert sha(trace_path)==r['artifacts'][trace_path.name]
  with np.load(trace_path,allow_pickle=False) as f: trace={k:f[k] for k in f.files}
  env=make_env(configs[level],level,r); obs,reset=reset_with_provenance(env)
  assert reset['initial_observation_sha256']==r['initial_observation_sha256']
  if category in ['stall_rescue','lost_success']:
   gated=rec(level,'stall_sampled',i)[1]; centre=gated['rescue_trigger_steps'][0]; lo=max(0,centre-60); hi=centre+240
  elif category=='early_death': lo=max(0,r['steps']-240); hi=r['steps']
  else: lo=0; hi=240
  frames=[]
  try:
   for t,action in enumerate(trace['actions']):
    obs,reward,done,info=env.step(int(action))
    assert x_position(info)==int(trace['x_after'][t]),(level,i,t,'x')
    assert bool(done)==bool(trace['done'][t]),(level,i,t,'done')
    assert np.isclose(reward,trace['rewards'][t],atol=1e-7,rtol=0),(level,i,t,'reward')
    if lo<=t<hi:
     raw=Image.fromarray(env.latest_raw_rgb()).resize((480,448),Image.Resampling.NEAREST)
     canvas=Image.new('RGB',(480,500),'#111c2c');canvas.paste(raw,(0,52));d=ImageDraw.Draw(canvas)
     d.text((8,5),f'{level} | {arm} | episode {i} | decision {t}',fill='white')
     outcome='COMPLETED' if r['completed'] else 'DEATH' if r['died'] else 'TIMEOUT'
     d.text((8,22),f'Recorded full episode: {outcome} | shown at game speed',fill='white')
     frames.append(np.asarray(canvas))
  finally: env.close()
  filename=f'{category}_{arm}.mp4';save_video(movies/filename,frames,fps=15)
  results.append({'category':category,'level':level,'arm':arm,'episode_index':i,'source':str(path),'source_sha256':sha(path),'trace_sha256':sha(trace_path),'checkpoint_sha256':manifest['levels'][level]['checkpoint_sha256'],'full_steps_verified':r['steps'],'shown_decisions':[lo,min(hi,r['steps'])],'completed':r['completed'],'died':r['died'],'timed_out':r['timed_out'],'video':filename,'video_sha256':sha(movies/filename)})
  print('Verified and rendered',filename,flush=True)
 if not a.smoke_only: (movies/'manifest.json').write_text(json.dumps({'selection':'Smallest episode index satisfying each labelled category in the 100-trial confirmation; control is episode 0. Clips illustrate selected outcomes and do not estimate rates.','records':results},indent=2)+'\n')
 # Run one short inference-backed evaluation using the frozen experiment runner.
 runner=exp/'source/scripts/eval_controller_diagnostic.py'; smoke=a.out/'model_smoke'; dev=Path(manifest['development_experiment'])
 subprocess.run([sys.executable,str(runner),'--manifest',str(dev/'manifest.json'),'--out',str(smoke),'--level','Level6-1','--smoke','--episodes','1','--conditions','greedy'],check=True)
 summary=json.loads((smoke/'summary.json').read_text()); assert summary['complete']; assert summary['conditions'][0]['infrastructure_errors']==0
 (a.out/'result.json').write_text(json.dumps({'passed':True,'elapsed_seconds':time.monotonic()-start,'clips':len(json.loads((movies/'manifest.json').read_text())['records']),'evaluation':'one frozen-checkpoint greedy Level6-1 smoke trial; not new performance evidence','smoke_summary_sha256':sha(smoke/'summary.json'),'checkpoint_sha256':manifest['levels']['Level6-1']['checkpoint_sha256'],'python':sys.version,'torch':torch.__version__,'cuda_available':torch.cuda.is_available()},indent=2)+'\n')
if __name__=='__main__': main()
