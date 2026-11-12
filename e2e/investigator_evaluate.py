"""Small, reproducible synthetic functional evaluation; not a real-world benchmark."""
import argparse,copy,json,random,re,statistics,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ledger_recon.investigator.demo import make_bundle
from ledger_recon.investigator.engine import Engine


def varied(seed):
    bundle,expected=make_bundle(seed)
    # Vary money while preserving the same nine disclosed case templates.
    scale=random.Random(seed).randint(2,9)
    for kind in ('invoices','transactions'):
        for record in bundle[kind]:record['amount_cents']*=scale
    for doc in bundle['documents']:
        text=doc['text']
        text=re.sub(r'(amount_cents=|Credit amount \(cents\): )(\d+)',lambda m:m[1]+str(int(m[2])*scale),text)
        text=re.sub(r'(INV-[A-Z-]+:)(\d+)',lambda m:m[1]+str(int(m[2])*scale),text)
        doc['text']=text
    return bundle,expected


def baseline(bundle,tx):
    candidates=[i for i in bundle['invoices'] if i['customer']==tx['customer'] and i['currency']==tx['currency'] and i['amount_cents']==tx['amount_cents']]
    return [candidates[0]['id']] if len(candidates)==1 else []


def main():
    p=argparse.ArgumentParser();p.add_argument('--ollama-smoke',action='store_true');args=p.parse_args()
    report={'dataset':'Nine hand-authored templates; six unseen amount seeds. Synthetic functional coverage, not generalization or production accuracy.','seeds':list(range(21001,21007)),'results':[]}
    with tempfile.TemporaryDirectory(prefix='investigator-eval-') as tmp:
        for seed in report['seeds']:
            bundle,expected=varied(seed);engine=Engine(Path(tmp)/str(seed),mode='rules',embeddings=False);engine.import_bundle(bundle)
            for tx in bundle['transactions']:
                key=tx['id'];t=time.perf_counter();r=engine.investigate(key);elapsed=time.perf_counter()-t
                expected_ids=expected[key]['invoice_ids'];ids=sorted(a['invoice_id'] for a in r['allocations'])
                correct=r['status']==expected[key]['status'] and (r['status']=='blocked' or ids==expected_ids)
                base=baseline(bundle,tx)
                report['results'].append({'seed':seed,'case':key,'kind':expected[key]['kind'],'expected_status':expected[key]['status'],'status':r['status'],'correct':correct,'false_proposal':r['status']=='review' and (expected[key]['status']!='review' or ids!=expected_ids),'baseline_proposal':base,'baseline_correct_proposal':bool(base) and expected[key]['status']=='review' and base==expected_ids,'seconds':round(elapsed,4),'retrieval_mode':r['retrieval_mode']})
            engine.shutdown()
        rows=report['results'];report['summary']={'cases':len(rows),'correct_case_outcomes':sum(r['correct'] for r in rows),'false_proposals':sum(r['false_proposal'] for r in rows),'reviewable_cases':sum(r['expected_status']=='review' for r in rows),'investigator_correct_proposals':sum(r['correct'] and r['status']=='review' for r in rows),'baseline_correct_proposals':sum(r['baseline_correct_proposal'] for r in rows),'median_seconds':round(statistics.median(r['seconds'] for r in rows),4),'extraction_mode':'rules','embeddings':False}
        if args.ollama_smoke:
            # Single integration case, explicitly not an LLM accuracy benchmark.
            bundle,_=make_bundle();bundle={k:[r for r in v if (r['id']=='INV-CREDIT' if k=='invoices' else r['id']=='PAY-CREDIT' if k=='transactions' else r['id'] in ('credit-note','credit-remittance'))] for k,v in bundle.items()}
            engine=Engine(Path(tmp)/'local-model',mode='ollama',model='qwen2.5:7b',embeddings=True);engine.import_bundle(bundle);t=time.perf_counter();r=engine.investigate('PAY-CREDIT');elapsed=time.perf_counter()-t
            report['ollama_smoke']={'cases':1,'status':r['status'],'seconds':round(elapsed,3),'model':r.get('extraction_metadata',{}).get('model'),'provider':r.get('extraction_metadata',{}).get('provider'),'retrieval_mode':r['retrieval_mode'],'retrieval_note':r['retrieval_note'],'allocations':r['allocations'],'credits':r['credits'],'blockers':r['blockers'],'citation_count':len(r['citations']),'model_raw_output':r.get('extraction_metadata',{}).get('raw_output'),'limitation':'One post-prompt-development integration case, not held-out LLM accuracy. Earlier development attempts were blocked for amount/credit errors.'}
            engine.shutdown()
    Path('reports').mkdir(exist_ok=True)
    Path('reports/payment-investigator.json').write_text(json.dumps(report,indent=2)+'\n')
    s=report['summary'];text=f"# Payment investigator verification\n\n{report['dataset']}\n\nRules mode with real LangGraph, Qdrant sparse retrieval and SQLite: {s['correct_case_outcomes']}/{s['cases']} expected case outcomes; {s['false_proposals']} false actionable proposals. {s['investigator_correct_proposals']}/{s['reviewable_cases']} evidence-supported proposals versus {s['baseline_correct_proposals']} from unique exact-amount matching. Median investigation time {s['median_seconds']} s. This baseline is deliberately simple and is not the original RL benchmark's stronger heuristic.\n\nAll six seeds repeat the same nine templates with varied amounts. These results do not estimate accuracy on unseen document formats, scanned invoices, real bank records or a different company. Embeddings and LLM extraction are disabled in this deterministic matrix.\n"
    if report.get('ollama_smoke'):
        r=report['ollama_smoke'];text+=f"\nLocal model integration: {r['model']} with {r['retrieval_mode']}, one credit case, status {r['status']}, {r['seconds']} seconds, {r['citation_count']} validated citations. Earlier development model calls made cent/credit errors and were blocked; this successful-or-blocked smoke result is not an accuracy claim. Raw observed output is preserved in the JSON report. No paid model API used.\n"
    text+='\nPost-implementation functional checks cover stale and superseded proposals, conflicting evidence, credit replay, concurrent approval, balance contention, persisted graph interruption/resume, invalid imports and untrusted document instructions. Browser/HTTP checks cover investigation, citations, approval, reload, PDF text extraction and blocked controls.\n'
    Path('reports/payment-investigator.md').write_text(text)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    assert all(r['correct'] for r in rows), 'A rules case failed; inspect report before claiming success'


if __name__=='__main__':main()
