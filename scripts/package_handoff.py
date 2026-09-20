"""Freeze the readable working tree and presentation without modifying unrelated Git work."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'handoff_release');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 files=[]
 for dirname in ['src','conf','scripts','tests','docs','images','analysis']:
  files += [f for f in (ROOT/dirname).rglob('*') if f.is_file() and not f.is_symlink() and '__pycache__' not in f.parts and f.suffix not in ['.pyc','.mp4']]
 files += [ROOT/f for f in ['README.md','CLAUDE.md','BACKLOG.md','requirements.txt','setup.py','.gitignore']]
 records=[];target=a.out/'muzero_mario_source_20260920.zip'
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(set(files)):
   raw=f.read_bytes();name=str(f.relative_to(ROOT));z.writestr(name,raw);records.append({'path':name,'bytes':len(raw),'sha256':sha(raw)})
  meta={'snapshot':'muzero-mario-handoff-20260920','git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'description':'Exact working-tree source/document snapshot, including uncommitted relevant files. Base commit alone does not identify this code; file hashes do. Excludes ROM, raw human data, checkpoints, environment and videos.','files':records}
  z.writestr('HANDOFF_SOURCE_MANIFEST.json',json.dumps(meta,indent=2)+'\n')
 (a.out/'source_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
 target2=a.out/'muzero_mario_presentation_20260920.zip'
 with zipfile.ZipFile(target2,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted((ROOT/'docs/handoff/presentation').rglob('*')):
   if f.is_file():z.write(f,str(f.relative_to(ROOT/'docs/handoff/presentation')))
  z.writestr('README.txt','Open index.html locally. Arrow keys navigate; N toggles notes. Keep clips/ beside it. PDF is the static fallback. Selected clips illustrate outcomes, not population rates.\n')
 manifest={'snapshot':meta['snapshot'],'archives':[{'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in [target,target2]],'source_file_count':len(records),'external_assets':'docs/handoff/verification/assets.json','notes':'Source archive contains a dirty-tree manifest. Presentation archive includes six MP4s. Nothing has been uploaded or shared.'}
 (a.out/'release.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (a.out/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['file']}\n" for r in manifest['archives']))
 for r in manifest['archives']:print(r['file'],r['bytes'],r['sha256'])
if __name__=='__main__':main()
