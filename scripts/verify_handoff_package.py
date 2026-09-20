"""Verify and regenerate a source package in an isolated extraction with a chosen Python."""
import argparse,hashlib,json,subprocess,tempfile,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--python',required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();start=time.monotonic()
 with tempfile.TemporaryDirectory(prefix='muzero-handoff-verify-') as directory:
  out=Path(directory)
  with zipfile.ZipFile(a.archive) as z:
   assert z.testzip() is None
   assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
   z.extractall(out)
  m=json.loads((out/'HANDOFF_SOURCE_MANIFEST.json').read_text())
  for f in m['files']:assert digest(out/f['path'])==f['sha256'],f['path']
  scripts=['plot_project_handoff.py','build_handoff_presentation.py']
  for s in scripts:subprocess.run([a.python,str(out/'scripts'/s)],cwd=tempfile.gettempdir(),check=True)
  assert not (out/'outputs').exists()
  assert (out/'images/handoff/benchmark.csv').read_bytes()==(ROOT/'images/handoff/benchmark.csv').read_bytes()
  html=(out/'docs/handoff/presentation/index.html').read_text();assert html.count('class="slide"')==13
  r={'passed':True,'source_archive_sha256':digest(a.archive),'source_files_verified':len(m['files']),'elapsed_seconds':time.monotonic()-start,'python':a.python,'checks':['ZIP integrity and every source file hash','both renderers run from isolated extraction with no outputs, models or ROM','benchmark CSV byte-identical','HTML has 13 slides'],'renderer_sha256':{s:digest(out/'scripts'/s) for s in scripts},'limits':'Final package may add this report and documentation commit metadata; renderer hashes identify verified executable code.'}
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(r,indent=2)+'\n');print('PASS:',json.dumps(r))
if __name__=='__main__':main()
