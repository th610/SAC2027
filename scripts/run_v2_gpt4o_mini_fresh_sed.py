#!/usr/bin/env python3
"""Fresh, cache-isolated 65-task GPT-4o-mini V2/SED reproduction."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hiddenbench.benchmark import find_task, load_benchmark
from hiddenbench.prompts import load_prompts, render_prompt
from hiddenbench.simulator import Profile, format_discussion_transcript, instantiate_agents, run_scenario

MODEL='gpt-4o-mini'; TEMP=0.0; SEED=42; ROUNDS=3; MAX_ATTEMPTS=3
PROMPT=ROOT/'src/hiddenbench/prompts_v2.json'
V1_PROMPT=ROOT/'src/hiddenbench/prompts.json'
REFERENCE=ROOT/'runs/prompt_v2_gpt4o_mini_hidden_tasks_1_20_fresh_20261015'
SMOKE_DEFAULT=ROOT/'runs/prompt_v2_gpt4o_mini_hidden_fresh_sed_20261017_smoke'
FULL_DEFAULT=ROOT/'runs/prompt_v2_gpt4o_mini_hidden_fresh_sed_20261017'
DOTENV=Path('.env')

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); os.replace(tmp,p)
def ref_run(tid): return json.loads((REFERENCE/f'task_{tid}.json').read_text())['runs'][0]
def vote_summary(votes):
 counts=Counter(v['vote'] for v in votes); return counts.most_common(1)[0][0]

def static_validate():
 benchmark=load_benchmark(path=str(ROOT/'data/benchmark.json')); prompts=load_prompts(version='v2')
 assert len(benchmark)==65
 checks=Counter(); reference_hashes={}
 for tid in range(1,66):
  task=find_task(benchmark,task_id=str(tid)); ref=ref_run(tid); p=REFERENCE/f'task_{tid}.json'
  reference_hashes[str(p.relative_to(ROOT))]=digest(p)
  assert ref['task_id']==task.id and ref['correct_answer']==task.correct_answer
  assert ref['possible_answers']==task.possible_answers
  agents,assignments=instantiate_agents(task,None,Profile.HIDDEN,prompts,random.Random(SEED))
  assert len(agents)==ref['num_agents'] and assignments==ref['fact_assignments']
  assert [a.history[0]['content'] for a in agents]==[x['prompt'] for x in ref['prompt_diagnostics']['system_prompts']]
  checks['task/gold/options/agents/fact_assignment/system_prompt']+=1
  init=render_prompt(prompts.first_vote_prompt,{'possible_answers':', '.join(task.possible_answers),'group_discussion':'No discussion has occurred yet. This is your initial vote based solely on the information available to you.'})
  assert init==ref['prompt_diagnostics']['initial_vote_prompt']; checks['initial_prompt']+=1
  names=[a.name for a in agents]; previous_round={}; messages=ref['rounds']
  assert len(messages)==ROUNDS
  for rid,rd in enumerate(messages):
   assert len(rd['messages'])==len(agents)
   if rid==0:
    prior=[]
    for idx,m in enumerate(rd['messages']):
     iv=ref['initial_votes'][idx]
     prefix=f"Your initial vote was: {iv['vote']}\nYour initial rationale was: {iv['rationale']}\n\n"
     exp=prefix+prompts.first_user_prompt if not prior else prefix+render_prompt(prompts.user_prompt,{'messages':'\n'.join(prior),'extra':''})
     assert m['prompt']==exp,(tid,'R1',idx)
     prior.append(f"{names[idx]}: {m['response']}")
   else:
    latest=dict(previous_round)
    for idx,m in enumerate(rd['messages']):
     incoming=[f"{names[(idx+off)%len(names)]}: {latest[names[(idx+off)%len(names)]]}" for off in range(1,len(names))]
     exp=render_prompt(prompts.user_prompt,{'messages':'\n'.join(incoming),'extra':''})
     assert m['prompt']==exp,(tid,f'R{rid+1}',idx)
     latest[m['agent']]=m['response']
   previous_round={m['agent']:m['response'] for m in rd['messages']}
   checks[f'round_{rid+1}_prompts']+=1
  final=render_prompt(prompts.vote_prompt,{'group_discussion':format_discussion_transcript(ref['rounds'][-1]['messages']),'possible_answers':', '.join(task.possible_answers)})
  assert final==ref['prompt_diagnostics']['final_vote_prompt']; checks['final_vote_prompt']+=1
 expected=sum(len(t.hidden_information) for t in benchmark)*5
 assert expected==1265
 return {'status':'PASS','tasks':65,'model':MODEL,'temperature':TEMP,'seed':SEED,'rounds':ROUNDS,'cache_policy':'full output directory is distinct from smoke and all previous runs; no old task cache reused','prompt_sha256':digest(PROMPT),'v1_backbone_sha256':digest(V1_PROMPT),'benchmark_sha256':digest(ROOT/'data/benchmark.json'),'reference_task_files_sha256':reference_hashes,'reference_checks':dict(checks),'agents_total':sum(len(t.hidden_information) for t in benchmark),'expected_logical_calls':expected,'api_called':False}

class TracingClient:
 model_name=MODEL
 def __init__(self,api_key):
  self.client=OpenAI(api_key=api_key,timeout=90,max_retries=0)
  self.attempts=[]; self.logical_requests=0
 def _request(self,messages,response_format=None):
  self.logical_requests+=1; last=None
  for attempt in range(1,MAX_ATTEMPTS+1):
   rec={'attempt':attempt,'model_requested':MODEL,'messages':messages,'temperature':TEMP,'response_format':response_format,'started_at':datetime.now(timezone.utc).isoformat()}
   try:
    kw={'model':MODEL,'messages':messages,'temperature':TEMP}
    if response_format is not None: kw['response_format']=response_format
    response=self.client.chat.completions.create(**kw); choice=response.choices[0]; text=choice.message.content or ''; usage=response.usage
    rec.update(request_id=getattr(response,'id',None),response_model=getattr(response,'model',None),finish_reason=choice.finish_reason,raw_response=text,usage={k:getattr(usage,k,None) for k in ('prompt_tokens','completion_tokens','total_tokens')} if usage else None)
    if not text.strip(): raise ValueError('Empty response')
    if choice.finish_reason=='length': raise ValueError('Response truncated at max length')
    rec['status']='ok'; rec['completed_at']=datetime.now(timezone.utc).isoformat(); self.attempts.append(rec)
    return text
   except Exception as exc:
    last=exc
    rec.update(status='error',error_type=type(exc).__name__,error=str(exc),completed_at=datetime.now(timezone.utc).isoformat()); self.attempts.append(rec)
    if attempt==MAX_ATTEMPTS or isinstance(exc,ValueError): raise
    time.sleep(min(2**(attempt-1),8))
  raise last
 def chat(self,messages): return self._request(messages)
 def chat_json(self,messages,schema):
  req=messages+[{'role':'user','content':'Return only one valid JSON object. Do not include markdown or commentary.'}]
  fmt={'type':'json_schema','json_schema':{'name':'hiddenbench_response','strict':True,'schema':schema}}
  return json.loads(self._request(req,fmt))
 def close(self): self.client.close()

def validate_cached(path,tid,static,runner_hash):
 if not path.exists(): return None
 d=json.loads(path.read_text()); m=d['metadata']; r=d['run']
 expected={'task_id':tid,'model':MODEL,'temperature':TEMP,'seed':SEED,'rounds':ROUNDS,'prompt_sha256':static['prompt_sha256'],'runner_sha256':runner_hash,'dataset_sha256':static['benchmark_sha256']}
 if any(m.get(k)!=v for k,v in expected.items()): raise ValueError(f'Cached task config mismatch: {path}')
 ref=ref_run(tid)
 assert r['correct_answer']==ref['correct_answer'] and r['possible_answers']==ref['possible_answers'] and r['fact_assignments']==ref['fact_assignments']
 assert len(r['initial_votes'])==len(r['final_votes'])==r['num_agents']
 assert len(r['rounds'])==3 and all(len(x['messages'])==r['num_agents'] for x in r['rounds'])
 assert r['majority_vote']==vote_summary(r['final_votes'])
 assert d['generation_call_count']==r['num_agents']*5
 return d

def exact_mcnemar(b,c):
 n=b+c
 if n==0:return 1.0
 return min(1.0,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/(2**n))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['static','smoke','full'],required=True); ap.add_argument('--output-dir',type=Path); ap.add_argument('--smoke-dir',type=Path,default=SMOKE_DEFAULT); ap.add_argument('--dotenv',type=Path,default=DOTENV); args=ap.parse_args()
 static=static_validate()
 if args.phase=='static': print(json.dumps({k:v for k,v in static.items() if k!='reference_task_files_sha256'},ensure_ascii=False,indent=2)); return 0
 load_dotenv(args.dotenv)
 key=os.getenv('OPENAI_API_KEY')
 if not key: raise RuntimeError('OPENAI_API_KEY missing; expected in configured dotenv/environment')
 out=(args.smoke_dir if args.phase=='smoke' else (args.output_dir or FULL_DEFAULT)).resolve()
 if args.phase=='full':
  gate=args.smoke_dir.resolve()/'smoke_pass.json'
  if not gate.exists(): raise RuntimeError(f'Smoke gate missing: {gate}')
  g=json.loads(gate.read_text()); assert g['prompt_sha256']==static['prompt_sha256'] and g['model']==MODEL
 out.mkdir(parents=True,exist_ok=True); runner_hash=digest(Path(__file__).resolve())
 frozen=out/'static_validation.json'
 if frozen.exists(): assert json.loads(frozen.read_text())==static
 else: atomic_json(frozen,static)
 ids=[1,2] if args.phase=='smoke' else list(range(1,66))
 baseline_rows=[]; task_rows=[]; total_attempts=0; failed=[]
 for tid in ids:
  dest=out/f'task_{tid}.json'
  cached=validate_cached(dest,tid,static,runner_hash) if args.phase=='full' else None
  if cached:
   saved=cached; print(f'Task {tid}: cache verified',flush=True)
  else:
   task=find_task(load_benchmark(path=str(ROOT/'data/benchmark.json')),task_id=str(tid)); client=TracingClient(key)
   try:
    print(f'Task {tid}/{len(ids)} started',flush=True); t0=time.monotonic()
    run=run_scenario(task,client,Profile.HIDDEN,load_prompts(version='v2'),rounds=ROUNDS,seed=SEED)
    n=run['num_agents']; expected=n*5
    assert client.logical_requests==expected,(tid,client.logical_requests,expected)
    assert len(run['initial_votes'])==len(run['final_votes'])==n
    usage={k:sum((a.get('usage') or {}).get(k) or 0 for a in client.attempts) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    metadata={'task_id':tid,'condition':'SED_V2_FULL_FRESH_REPRODUCTION','model':MODEL,'profile':'hidden','seed':SEED,'rounds':ROUNDS,'temperature':TEMP,'prompt_file':str(PROMPT.relative_to(ROOT)),'prompt_sha256':static['prompt_sha256'],'v1_backbone_sha256':static['v1_backbone_sha256'],'dataset_sha256':static['benchmark_sha256'],'runner_sha256':runner_hash,'fresh_no_cache':True,'created_at':datetime.now(timezone.utc).isoformat()}
    saved={'metadata':metadata,'run':run,'generation_call_count':client.logical_requests,'expected_generation_calls':expected,'api_attempt_count':len(client.attempts),'retry_count':len(client.attempts)-client.logical_requests,'token_usage':usage,'elapsed_seconds':time.monotonic()-t0,'api_attempts':client.attempts}
    if dest.exists(): raise FileExistsError(f'Refusing overwrite: {dest}')
    atomic_json(dest,saved); client.close()
    print(f"Task {tid}: {run['majority_vote']} correct={run['majority_vote']==run['correct_answer']} logical={saved['generation_call_count']} attempts={saved['api_attempt_count']}",flush=True)
   except Exception as exc:
    failed.append({'task_id':tid,'type':type(exc).__name__,'error':str(exc)})
    atomic_json(out/f'task_{tid}.error.json',{'task_id':tid,'error_type':type(exc).__name__,'error':str(exc),'api_attempts':client.attempts,'created_at':datetime.now(timezone.utc).isoformat()})
    client.close(); print(f'Task {tid}: FAILED {type(exc).__name__}: {exc}',flush=True); continue
   saved=saved
  run=saved['run']; old=ref_run(tid); gold=run['correct_answer']; ini=vote_summary(run['initial_votes']); fin=vote_summary(run['final_votes'])
  task_rows.append({'task_id':tid,'correct':fin==gold,'initial_correct':ini==gold,'majority':fin,'gold':gold,'calls':saved['generation_call_count']})
  total_attempts+=saved['api_attempt_count']
  baseline_rows.append({'task_id':tid,'fresh_correct':fin==gold,'reference_correct':old['majority_vote']==old['correct_answer'],'fresh_majority':fin,'reference_majority':old['majority_vote']})
 summary={'model':MODEL,'condition':'SED_V2_FULL_FRESH_REPRODUCTION','completed':len(task_rows),'failed_task_ids':[x['task_id'] for x in failed],'errors':failed,'correct':sum(x['correct'] for x in task_rows),'accuracy':sum(x['correct'] for x in task_rows)/len(task_rows) if task_rows else None,'initial_correct':sum(x['initial_correct'] for x in task_rows),'expected_logical_calls':sum(x['calls'] for x in task_rows),'api_attempt_count':total_attempts,'retry_count':total_attempts-sum(x['calls'] for x in task_rows),'paired':{},'mcnemar_exact_two_sided_p':None,'task_rows':task_rows,'created_at':datetime.now(timezone.utc).isoformat(),'static_validation':{k:v for k,v in static.items() if k!='reference_task_files_sha256'}}
 if len(task_rows)==65:
  pairs=Counter((x['reference_correct'],x['fresh_correct']) for x in baseline_rows)
  b=pairs[(True,False)]; c=pairs[(False,True)]
  summary['paired']={'both_correct':pairs[(True,True)],'old_only_correct':b,'fresh_only_correct':c,'both_wrong':pairs[(False,False)],'task_majority_identical':sum(x['fresh_majority']==x['reference_majority'] for x in baseline_rows)}
  summary['mcnemar_exact_two_sided_p']=exact_mcnemar(b,c)
 atomic_json(out/'summary.json',summary)
 if args.phase=='smoke':
  assert len(task_rows)==2 and not failed
  atomic_json(out/'smoke_pass.json',{'status':'PASS','task_ids':[1,2],'model':MODEL,'prompt_sha256':static['prompt_sha256'],'fresh_output_excludes_smoke':True,'api_calls':total_attempts})
 print(json.dumps({k:v for k,v in summary.items() if k not in ('task_rows','static_validation')},ensure_ascii=False,indent=2),flush=True)
 return int(bool(failed))
if __name__=='__main__': raise SystemExit(main())
