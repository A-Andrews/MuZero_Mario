"""Extract completed September 20 diagnostics into a portable, source-hashed summary."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 leaf_path=ROOT/'outputs/controller_policy/search-leaf-values-v1b-20260919/evidence_summary.json'
 target_path=ROOT/'diagnostic_outputs/value-target-audit-v1b-20260920/summary.json'
 leaf=json.loads(leaf_path.read_text()); target=json.loads(target_path.read_text())
 assert leaf['search_recovery_complete'] and leaf['recovered_roots']==56
 assert target['complete'] and target['branch_records']==2800 and len(target['roots'])==56
 rows=[]
 signatures={r['root_id']:r['signature'] for r in leaf['roots']}
 for root in target['roots']:
  meta=root['root']; case=meta['case']
  for arm in ['continue','logged_prefix','release','right_jump','right_run_jump']:
   endpoints=[e for e in root['endpoints'] if e['position']==8 and e['arm']==arm]
   if not endpoints: continue
   assert len(endpoints)==10 and all(e['remaining_forced_steps']==0 for e in endpoints)
   row={'root_id':meta['root_id'],'level':case['level'],'source_episode':case['episode_index'],'signature':signatures[meta['root_id']],'source_step':meta['root']['step'],'source_controller':case['source_condition'],'arm':arm,'n':len(endpoints),'completed':sum(e['completed'] for e in endpoints),'terminal_tails':sum(e['tail_reaches_true_terminal'] for e in endpoints),'timeouts':sum(e['timed_out'] for e in endpoints),'prediction_median':statistics.median(e['value'] for e in endpoints),'realized_tail_median':statistics.median(e['realized_discounted_tail'] for e in endpoints),'target_medians':{str(h):statistics.median(e['targets'][str(h)]['current_online_bootstrap_target'] for e in endpoints) for h in [1,10,50,100,200]}}
   rows.append(row)
 depths={}
 for root in leaf['roots']:
  for depth,count in root['search']['baseline']['backup_depths'].items(): depths[depth]=depths.get(depth,0)+count
 terminal=sum(r['search'][a]['terminal_backups'] for r in leaf['roots'] for a in ['baseline','1','8'])
 assert sum(depths.values())==224000 and terminal==0
 data={'as_of':'2026-09-20','sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [leaf_path,target_path]],'leaf_groups':leaf['groups'],'leaf_roots':[{k:r[k] for k in ['root_id','level','episode','condition','signature','step','outcomes']} for r in leaf['roots']],'baseline_depths':depths,'terminal_backups_all_arms':terminal,'target_audit':{'complete':True,'branch_records':2800,'roots':56,'position':8,'rows':rows,'interpretation':'Online-network bootstrap proxy for unavailable historical target network; repeated conditional seeds are not independent starts. Nonterminal timeout tails are finite sums, not Monte Carlo truth.'}}
 (ROOT/'docs/handoff/diagnostic_evidence_20260920.json').write_text(json.dumps(data,indent=2)+'\n')
 print('Validated 56 leaf roots, 224000 baseline backups and 2800 value-target records; saved',len(rows),'root/arm summaries.')
if __name__=='__main__':main()
