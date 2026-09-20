"""Build a local research talk (HTML, PDF, notes) and September 20 figures."""
import base64,html,json,os,tempfile,textwrap,hashlib,shutil
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'muzero-handoff-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/handoff/presentation'; FIG=ROOT/'images/handoff'

def export(fig,name):
 for ext in ['png','pdf','svg']:
  path=FIG/f'{name}.{ext}';fig.savefig(path,dpi=160,facecolor='white')
  if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
 plt.close(fig)

def figures(d):
 fig,(ax,depth)=plt.subplots(1,2,figsize=(15,7),gridspec_kw={'width_ratios':[1.4,1]});fig.subplots_adjust(left=.24,right=.97,bottom=.24,top=.80,wspace=.35)
 fig.suptitle('Replacing searched leaf values helps some failures and harms some successes',x=.025,ha='left',fontsize=18)
 groups=[g for g in d['leaf_groups'] if g['condition']=='greedy' and (g['signature'] in ['pipe_898','successful_reference'] or g.get('timing') in ['early','late'])]
 # Aggregate early/late cases by signature without double-counting the all-timing groups.
 wanted=[('Level6-1','pit_2807'),('Level6-1','obstacle_1394'),('Level6-1','successful_reference'),('Level1-1','pipe_898'),('Level1-1','successful_reference')]
 matrix=[]; names=[]; cells=[]
 for lv,sig in wanted:
  gs=[g for g in d['leaf_roots'] if g['level']==lv and g['signature']==sig and g['condition']=='greedy']
  assert gs,(lv,sig)
  g={'n_episodes':len(set(r['episode'] for r in gs)),'n_roots':len(gs)}
  counts=[(sum(r['outcomes'][a]['completed'] for r in gs),sum(r['outcomes'][a]['n'] for r in gs)) for a in ['baseline','1','8']]
  matrix.append([k/n for k,n in counts]);cells.append(counts)
  desc={'pit_2807':'pit approaches','obstacle_1394':'obstacle failures','successful_reference':'successful references','pipe_898':'pipe failures'}[sig]
  names.append(f"{lv.replace('Level','')} {desc}\n{g['n_episodes']} episodes / {g['n_roots']} roots")
 ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto')
 for y,cs in enumerate(cells):
  for x,(k,n) in enumerate(cs):ax.text(x,y,f'{k}/{n}',ha='center',va='center',color='white' if k/n>.55 else '#18344a',fontsize=12)
 ax.set_yticks(range(5),names,fontsize=10);ax.set_xticks(range(3),['Normal','Replace 1\ndecision','Replace 8\ndecisions']);ax.set_title('Conditional completions; same frozen checkpoints',fontsize=11)
 xs=sorted(map(int,d['baseline_depths']));ys=[d['baseline_depths'][str(x)] for x in xs]
 depth.bar(xs,np.array(ys)/sum(ys)*100,color='#2878a8');depth.set_xticks(xs);depth.set_xlabel('Depth of an evaluated leaf (decisions)');depth.set_ylabel('Baseline backups (%)');depth.set_title('80.5% of baseline backups reach depth 2–3',fontsize=11)
 fig.text(.035,.04,'1,120 new branches; 56 selected roots from 28 source episodes. Repeated branch seeds are conditional, not independent starts.\nPrivileged emulator access; replacement values are predictions from the same network, not ground truth.\nNo terminal paths were reached in any arm. Intervention affects subsequent search allocation as well as value backups.',fontsize=10)
 export(fig,'06_actual_search_values')
 fig,ax=plt.subplots(figsize=(13,6));fig.subplots_adjust(left=.09,right=.97,top=.77,bottom=.28)
 fig.suptitle('A short value target is not the only possible bottleneck',x=.04,ha='left',fontsize=19)
 specs=[(40,'right_run_jump','6-1 obstacle: successful jump'),(35,'right_jump','6-1 pit: successful jump'),(37,'continue','6-1 successful reference')]
 for i,(rid,arm,label) in enumerate(specs):
  r=next(r for r in d['target_audit']['rows'] if r['root_id']==rid and r['arm']==arm)
  assert r['terminal_tails']==r['n']==r['completed']==10
  vals=[r['prediction_median'],r['target_medians']['10'],r['target_medians']['200'],r['realized_tail_median']]
  for j,(v,c) in enumerate(zip(vals,['#2878a8','#d77c22','#8d65a8','#138a72'])):
   x=i+(j-1.5)*.18;ax.bar(x,v,.17,color=c,label=['Value prediction','10-step proxy target','200-step proxy target','Realized terminal return'][j] if i==0 else None);ax.text(x,v+3,f'{v:.0f}',ha='center',fontsize=10)
 ax.set_xticks(range(3),[x[2] for x in specs],fontsize=10);ax.set_ylabel('Discounted reward units at post-prefix endpoint');ax.set_ylim(0,190)
 ax.legend(ncol=2,loc='upper right',frameon=False,fontsize=9)
 fig.text(.035,.035,'Illustrative pre-existing roots 40, 35 and 37; position 8; median of 10 conditional continuations per root/arm.\nThe current online network substitutes for the unavailable historical lagged target network. These are not original learner targets.\nAt the obstacle root, the 10-step proxy already demands a correction (18 → 131). This does not identify why training missed it.',fontsize=10)
 export(fig,'07_value_target_examples')


def human_slide():
 data=json.loads((ROOT/'docs/handoff/evidence.json').read_text());rows=data['benchmark'];x=np.arange(len(rows))
 fig,(rate,speed)=plt.subplots(2,1,figsize=(15,7.2),sharex=True,gridspec_kw={'height_ratios':[1.2,1]})
 fig.subplots_adjust(left=.075,right=.98,top=.87,bottom=.19,hspace=.22)
 fig.suptitle('Reliability varies sharply across the human-covered levels',x=.035,ha='left',fontsize=20)
 rate.scatter(x,[r['human']['no_death_rate']*100 for r in rows],marker='D',s=28,color='#343a40',label='Human first-life corpus reference',zorder=4)
 for arm,offset,color,label in [('greedy',-.14,'#2878a8','Greedy MCTS'),('sampled',.14,'#d77c22','Always sampled')]:
  for i,r in enumerate(rows):
   if arm not in r['arms']:continue
   a=r['arms'][arm];p=a['rate']*100;lo,hi=np.array(a['wilson95'])*100
   rate.errorbar(i+offset,p,yerr=[[max(0,p-lo)],[max(0,hi-p)]],fmt='o',ms=4,color=color,alpha=.9,label=label if i==0 else None,lw=1)
   if a['successful_median'] is not None:speed.scatter(i+offset,a['successful_median']/r['human']['length_median'],s=28,color=color)
 rate.set_ylabel('Completion (%)');rate.set_ylim(-4,110);rate.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.26),fontsize=10,frameon=False)
 speed.axhline(1,color='#343a40',ls='--');speed.set_ylabel('Successful duration /\nhuman successful median');speed.set_ylim(0,1.9)
 speed.set_xticks(x,[r['level'].replace('Level','') for r in rows]);speed.set_xlabel('Level • 5-3 has human data but no model benchmark')
 missing=next(i for i,r in enumerate(rows) if not r['arms']);speed.text(missing,.12,'Missing',rotation=90,ha='center',fontsize=10,color='#8e4555')
 for a in [rate,speed]:a.grid(axis='y',alpha=.18);a.set_axisbelow(True)
 fig.text(.035,.025,'30 development attempts per model/controller; model error bars are fixed-checkpoint Wilson 95% intervals. Human rates pool repeated attempts from five people.\nSpeed dots condition on success; human successful lives include respawns. Starts, practice and budgets differ. Faster successful runs do not establish human-level reliability.',fontsize=10)
 export(fig,'human_context_presentation')

SLIDES=[
 {'title':'MuZero–Mario: what works, what fails, and how to resume','bullets':['A research handoff • 20 September 2026','Partial specialist competence; broad human-level performance is not established.','A replicated controller improvement and controlled tests now narrow the failure mechanisms.'],'notes':'0:00–1:00. State the purpose: preserve the project and agree on the next experiment. The human reference is the recorded CNeuroMod cohort, not expert speedrunners. This talk reports existing experiments, not newly trained models.'},
 {'title':'What would human-level performance mean?','bullets':['Reliability: completing a level on the first life, counting deaths and timeouts.','Speed: time among successful attempts, reported separately.','Breadth: all 22 human-covered levels; 21 currently evaluated model levels, with 5-3 missing.','Robustness: starts and training seeds; a specialist fleet is not a single generalist.'],'notes':'1:00–2:30. Five human participants, repeated practice and scanner conditions. Human/model starts and time caps differ. Human successful duration bands include respawn lives. Avoid one human-normalized headline ratio. Explain uncertainty conditions on frozen checkpoints.'},
 {'title':'Inside MuZero: observe, imagine, search, act','image':'05_inside_muzero.png','notes':'2:30–4:00. Explain representation, dynamics, policy prior, value, and MCTS. The model predicts task-relevant hidden states, not screenshots. One executed action lasts four emulator frames; the system replans afterward. Values are expected shaped rewards, not winning probabilities. Benchmark search uses 50 simulations, leaf batch four.'},
 {'title':'What the project has built','bullets':['Specialists across the human-covered corpus; archived model bundles include training provenance.','Human demonstrations rescue some self-play failures; multi-level curriculum arms and a distillation pipeline exist.','Frozen checkpoint evaluations, exact replay and paired interventions separate several failure types.','A portable report, reproducible figures, checkpoint examples and an experiment/data registry.'],'notes':'4:00–5:00. Coverage of representations is not proof of competent gameplay. Flag imitation-trained models in downstream brain comparisons. T7 terminal noisy self-play metrics are not matched evaluation scores; T8 has no established completed student result.'},
 {'title':'Where we stand against recorded humans','image':'human_context_presentation.png','notes':'5:00–7:00. This is the all-level development comparison, 30 attempts/controller/level. The lower panel divides successful model median duration by the human successful median; human rate diamonds pool repetitions. Point to strong levels and unresolved failures; 5-3 is missing, not zero. Use the separate PDF to inspect fine detail. Do not infer general human-level play from isolated point estimates.'},
 {'title':'Removing stalls is not enough','image':'02_failure_balance.png','notes':'7:00–8:00. On the active 21-model-level set, 203 versus 202 completions out of 630. Always-on sampling eliminates 90 timeouts but adds 91 deaths. The older 22-model-level total included 2-2; it is not this cohort.'},
 {'title':'A practical improvement that replicated','image':'03_confirmed_controller.png','notes':'8:00–9:30. Independent 100-pair confirmation, 900 episodes. Stall-triggered sampling uses extra RAM x/player-state input, with unchanged weights. It rescues 76/78 greedy timeouts on 1-1 but loses one success. 86 versus 83 is not established superiority. No all-level gated result yet.'},
 {'title':'Watch the same recorded start under two controllers','clips':['stall_rescue_greedy.mp4','stall_rescue_stall_sampled.mp4'],'bullets':['Matched confirmation episode; first qualifying timeout-to-success pair by episode index.','Clips show a fixed window around intervention, at emulated game speed. Full trajectories were replay-verified.'],'notes':'9:30–10:30. Play the baseline and gated clips. Explain that clips illustrate the mechanism; aggregate tables estimate rates. Additional early-death, successful-control and lost-success clips are in clips/. If video is unavailable, show slide 7 and describe the exact endpoints.'},
 {'title':'Action timing can help—and harm','image':'04_failure_mechanisms.png','notes':'10:30–12:00. Holding the initial action rescues the late 6-1 obstacle roots, but harms successful references. The imagined/observed value discrepancy is a clue, not ground truth and not automatically the discrepancy encountered by search. Ten seeds at a selected root are conditional repeats.'},
 {'title':'Values on actual searched paths: a selective causal effect','image':'06_actual_search_values.png','notes':'12:00–13:30. Eight-decision substitution rescues 6-1 pit roots, not obstacle roots. Some 1-1 successes are lost. No terminal paths were encountered, so terminal-value replacement cannot explain these effects. Changed backups also alter future search allocation; the intervention does not isolate dynamics versus representation/value-head effects.'},
 {'title':'What the completed value-target audit adds','image':'07_value_target_examples.png','notes':'13:30–15:00. Audit complete: 56 roots and 2800 reused records. These are illustrative endpoints, not a population test. At root 40 a 10-step proxy target is already 131 against prediction 18; at another pit root it remains low. Historical target network and training replay are unavailable, so this cannot prove the original training cause.'},
 {'title':'The next work should answer three decisions','bullets':['1. Breadth: does a fixed gated controller preserve successes and improve completion over all eligible levels?','2. Mechanism: do useful states need better replay coverage/fitting, or different value targets?','3. Generality: can a student retain specialist skills under the same evaluation protocol?'],'notes':'15:00–17:00. The experiment cards in NEXT_EXPERIMENTS.md specify inputs, boundaries, outputs and decision rules. Finish the protocol for broader confirmation before consuming fresh seeds. Do not launch all experiments at once. Prefer targeted pilots and intact controls.'},
 {'title':'How someone can pick this up','bullets':['Read START_HERE.md, then REPORT.md and the current experiment registry.','Recreate figures in the pinned lightweight environment; run the bounded compute check for models.','Use the frozen source bundle and manifests, not a historical default configuration.','Confirm recipient data/compute access and name a project owner before departure.'],'notes':'17:00–18:00. Present verification results and distinguish clean plotting setup from tested existing cluster inference environment. The access checklist is user-specific; a test under the current account cannot prove a colleague has access. Agree ownership, first experiment and a review date.'}
]

def build():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
 plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
 d=json.loads((ROOT/'docs/handoff/diagnostic_evidence_20260920.json').read_text());figures(d);human_slide()
 (OUT/'slides.json').write_text(json.dumps(SLIDES,indent=2)+'\n')
 sections=[];notes=['# Speaker notes — approximately 18 minutes\n']
 with PdfPages(OUT/'MuZero_Mario_handoff.pdf') as pdf:
  for i,s in enumerate(SLIDES,1):
   fig=plt.figure(figsize=(16,9),facecolor='#f6f8fb');fig.text(.045,.94,s['title'],fontsize=24,weight='bold',color='#17334d',va='top')
   body=''
   if 'image' in s:
    im=plt.imread(FIG/s['image']);ax=fig.add_axes([.025,.085,.95,.79]);ax.imshow(im);ax.axis('off')
    b64=base64.b64encode((FIG/s['image']).read_bytes()).decode();body=f'<img alt="{html.escape(s["title"])}" src="data:image/png;base64,{b64}">'
   else:
    bullets=s.get('bullets',[])
    for j,b in enumerate(bullets):
     fig.text(.065,(.82-j*.08) if 'clips' in s else (.75-j*.15),textwrap.fill('• '+b,115 if 'clips' in s else 90),fontsize=16 if 'clips' in s else 23,color='#263d50',va='top',linespacing=1.4)
    body='<ul>'+''.join('<li>'+html.escape(b)+'</li>' for b in bullets)+'</ul>'
    if 'clips' in s:
     body='<div class="videos">'+''.join(f'<figure><video controls preload="metadata" src="clips/{c}"></video><figcaption>{html.escape(c.replace(".mp4","").replace("_"," "))}</figcaption></figure>' for c in s['clips'])+'</div>'+body

     for col,c in enumerate(s['clips']):
      poster=OUT/'clips'/c.replace('.mp4','.png')
      if poster.exists():
       ax=fig.add_axes([.12+col*.40,.14,.30,.43]);ax.imshow(plt.imread(poster));ax.axis('off')
     fig.text(.065,.07,'Static preview: play the paired videos in the HTML deck for the time course.',fontsize=13,color='#2878a8')
   fig.text(.045,.035,f'MuZero–Mario | Evidence snapshot 20 September 2026',fontsize=11,color='#536879');fig.text(.95,.035,f'{i}/{len(SLIDES)}',ha='right',fontsize=12,color='#536879');pdf.savefig(fig);plt.close(fig)
   sections.append(f'<section class="slide" id="s{i}"><h1>{html.escape(s["title"])}</h1>{body}<aside>{html.escape(s["notes"])}</aside><footer>MuZero–Mario • 20 September 2026 <span>{i}/{len(SLIDES)}</span></footer></section>')
   notes.append(f'## {i}. {s["title"]}\n\n{s["notes"]}\n')
 (OUT/'SPEAKER_NOTES.md').write_text('\n'.join(notes))
 css='''*{box-sizing:border-box}body{margin:0;background:#172a3b;font-family:Arial,sans-serif;color:#17334d}.slide{display:none;position:relative;background:#f6f8fb;width:100vw;height:100vh;padding:3vh 4vw 6vh;overflow:auto}.slide.active{display:flex;flex-direction:column}h1{font-size:clamp(24px,3.1vw,48px);margin:0 0 2vh}img{object-fit:contain;min-height:0;max-width:100%;flex:1}ul{font-size:clamp(20px,2.3vw,36px);line-height:1.5;margin:auto 2vw}li{margin:1.8vh 0}.videos{display:flex;justify-content:center;gap:2vw;min-height:0;height:55vh}figure{margin:0;text-align:center}video{max-height:48vh;max-width:43vw}figcaption{font-size:16px}footer{position:absolute;bottom:1.5vh;left:4vw;right:4vw;font-size:14px;color:#52677a}footer span{float:right}aside{display:none;background:#fff2ce;padding:1em;font-size:18px}.notes aside{display:block;position:absolute;bottom:7vh;left:5vw;right:5vw;z-index:2}.help{position:fixed;bottom:8px;left:40%;z-index:3;font-size:12px;color:#52677a} @media print{@page{size:16in 9in;margin:0}.slide,.slide.active{display:flex;width:16in;height:9in;break-after:page}.help,aside{display:none!important}}'''
 js='''let index=0;const slides=[...document.querySelectorAll('.slide')];function show(n){index=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle('active',i===index));location.hash='s'+(index+1)}document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(index+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(index-1)}if(e.key.toLowerCase()==='n')document.body.classList.toggle('notes');if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1)});show(parseInt(location.hash.slice(2)||'1')-1);'''
 (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>MuZero–Mario handoff</title><style>'+css+'</style><body>'+''.join(sections)+'<div class="help">← → slides · N notes · browser print for PDF</div><script>'+js+'</script></body></html>')
 (OUT/'provenance.json').write_text(json.dumps({'diagnostic_evidence_sha256':hashlib.sha256((ROOT/'docs/handoff/diagnostic_evidence_20260920.json').read_bytes()).hexdigest(),'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'slides':len(SLIDES)},indent=2)+'\n')
 print('Built 13-slide HTML/PDF talk, speaker notes and two new diagnostic figures.')
if __name__=='__main__':build()
