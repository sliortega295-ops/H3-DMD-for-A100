#!/usr/bin/env python3
import csv, json, pathlib, sqlite3, hashlib, re
from collections import defaultdict

RUN = pathlib.Path(__file__).resolve().parents[1]
RAW = RUN / 'raw'

def union(intervals):
    iv=sorted((int(a),int(b)) for a,b in intervals if b>a)
    out=[]
    for a,b in iv:
        if out and a <= out[-1][1]:
            if b>out[-1][1]: out[-1]=(out[-1][0],b)
        else: out.append((a,b))
    return sum(b-a for a,b in out), out

def subtract(a,b):
    # interval-list a minus union/list b
    _, bb=union(b)
    out=[]
    for x,y in a:
        cur=x
        for p,q in bb:
            if q<=cur: continue
            if p>=y: break
            if p>cur: out.append((cur,min(p,y)))
            cur=max(cur,q)
            if cur>=y: break
        if cur<y: out.append((cur,y))
    return out

def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def load(p):
    c=sqlite3.connect(p)
    strings=dict(c.execute('select id,value from StringIds'))
    nv=[(a,b,t) for a,b,t in c.execute("select start,end,text from NVTX_EVENTS where text in ('h3/student_step','h3/critic_update_F') and end is not null")]
    phases={t:(a,b) for a,b,t in nv}
    # preserve duplicates by stage, but phase windows are disjoint here
    if nv:
        wstart=min(a for a,b,_ in nv); wend=max(b for a,b,_ in nv)
    else:
        wstart=wend=None
    k=[]
    for a,b,d,s,m,dev,stream,gpid,corr in c.execute('select start,end,demangledName,shortName,mangledName,deviceId,streamId,globalPid,correlationId from CUPTI_ACTIVITY_KIND_KERNEL'):
        name=strings.get(d) if d else None
        if not name and s: name=strings.get(s)
        if not name and m: name=strings.get(m)
        name = name or 'UNRESOLVED_KERNEL'
        k.append(dict(start=a,end=b,name=name,device=dev,stream=stream,gpid=gpid,corr=corr))
    m=[]
    for a,b,bytes_,ck,src,dst,dev,ctx,stream,gpid,corr in c.execute('select start,end,bytes,copyKind,srcKind,dstKind,deviceId,contextId,streamId,globalPid,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY'):
        m.append(dict(start=a,end=b,bytes=bytes_,copy_kind=ck,src=src,dst=dst,device=dev,stream=stream,gpid=gpid,corr=corr))
    c.close()
    return strings,nv,phases,wstart,wend,k,m

def overlap_records(recs, ranges):
    return [r for r in recs if any(r['start'] < b and r['end'] > a for a,b in ranges)]

rows=[]; result={'run_id':RUN.name,'profile_kind':'diagnostic_only','formal_timing_used':False,'nodes':{},'world16_usable':False}
for p in sorted(RAW.glob('node*/*.sqlite')):
    node=p.parent.name
    strings,nv,phases,ws,we,k,m=load(p)
    ranges=[(a,b) for a,b,_ in nv]
    k2=overlap_records(k,ranges); m2=overlap_records(m,ranges)
    def iv_of(xs): return [(x['start'],x['end']) for x in xs]
    all_u,_=union(iv_of(k2))
    n_u,_=union(iv_of([x for x in k2 if 'nccl' in x['name'].lower()]))
    c_u,_=union(iv_of([x for x in k2 if x['name']!='UNRESOLVED_KERNEL' and 'nccl' not in x['name'].lower()]))
    unk=[x for x in k2 if x['name']=='UNRESOLVED_KERNEL']
    unk_u,_=union(iv_of(unk))
    exposed_n=union(subtract(iv_of([x for x in k2 if 'nccl' in x['name'].lower()]), iv_of([x for x in k2 if x['name']!='UNRESOLVED_KERNEL' and 'nccl' not in x['name'].lower()])))[0]
    # all named intervals as compute proxy is deliberately not used as true utilization
    h2=[x for x in m2 if x['copy_kind']==1]
    d2=[x for x in m2 if x['copy_kind']==2]
    d2u,_=union(iv_of(d2)); h2u,_=union(iv_of(h2))
    full_h2=sum(x['bytes'] for x in m if x['copy_kind']==1); full_d2=sum(x['bytes'] for x in m if x['copy_kind']==2)
    phase_rows={}
    for stage,(a,b) in phases.items():
        kk=[x for x in k if x['start']<b and x['end']>a]
        mm=[x for x in m if x['start']<b and x['end']>a]
        stage_all=union(iv_of(kk))[0]; stage_n=union(iv_of([x for x in kk if 'nccl' in x['name'].lower()]))[0]
        stage_d2=[x for x in mm if x['copy_kind']==2]
        phase_rows[stage]={'wall_s':(b-a)/1e9,'kernel_union_s':stage_all/1e9,'known_nccl_union_s':stage_n/1e9,'d2h_bytes':sum(x['bytes'] for x in stage_d2),'d2h_events':len(stage_d2)}
    device_ids=sorted({x['device'] for x in k2})
    proc_errors=[]
    log=RUN/'logs'/f'{node}_launcher.log'
    if log.exists() and 'Errors occurred while processing the raw events' in log.read_text(errors='ignore'):
        proc_errors.append('nsys_report_processing_error')
    info={
      'sqlite':str(p),'sqlite_sha256':sha(p),'nvtx_ranges':[(a,b,t,(b-a)/1e9) for a,b,t in nv],
      'workload_window_ns':[ws,we] if ws is not None else None,'workload_window_s':((we-ws)/1e9 if ws is not None else None),
      'phase_complete':set(phases)=={'h3/student_step','h3/critic_update_F'},
      'capture_processing_errors':proc_errors,
      'kernel_events_total':len(k),'kernel_events_in_nvtx':len(k2),'kernel_devices_in_nvtx':device_ids,
      'kernel_device_coverage':len(device_ids)/8,
      'kernel_union_s_observed':all_u/1e9,'known_compute_union_s':c_u/1e9,'known_nccl_union_s':n_u/1e9,
      'known_nccl_exposed_vs_known_compute_s':exposed_n/1e9,'unresolved_kernel_union_s':unk_u/1e9,
      'kernel_sum_duration_s':sum(x['end']-x['start'] for x in k2)/1e9,
      'kernel_coverage_vs_window_observed':(all_u/(we-ws) if ws and we else None),
      'h2d_events_nvtx':len(h2),'h2d_bytes_nvtx':sum(x['bytes'] for x in h2),'h2d_union_s_nvtx':h2u/1e9,
      'd2h_events_nvtx':len(d2),'d2h_bytes_nvtx':sum(x['bytes'] for x in d2),'d2h_union_s_nvtx':d2u/1e9,
      'h2d_events_full':sum(1 for x in m if x['copy_kind']==1),'h2d_bytes_full':full_h2,
      'd2h_events_full':sum(1 for x in m if x['copy_kind']==2),'d2h_bytes_full':full_d2,
      'copy_kind8_events_nvtx':sum(1 for x in m2 if x['copy_kind']==8),'copy_kind8_bytes_nvtx':sum(x['bytes'] for x in m2 if x['copy_kind']==8),
      'phases':phase_rows,
    }
    result['nodes'][node]=info
    for metric,val in [('kernel_union_s_observed',all_u/1e9),('known_compute_union_s',c_u/1e9),('known_nccl_union_s',n_u/1e9),('known_nccl_exposed_vs_known_compute_s',exposed_n/1e9),('unresolved_kernel_union_s',unk_u/1e9),('h2d_bytes_nvtx',sum(x['bytes'] for x in h2)),('d2h_bytes_nvtx',sum(x['bytes'] for x in d2)),('d2h_events_nvtx',len(d2))]:
        rows.append({'node':node,'scope':'all_nvtx','metric':metric,'value':val})
    for stage,v in phase_rows.items():
        for metric,val in v.items(): rows.append({'node':node,'scope':stage,'metric':metric,'value':val})
# validity: profile must have both phases and all 8 device kernel coverage with low unresolved fraction and no processing errors
valid=True; reasons=[]
for node,x in result['nodes'].items():
    if not x['phase_complete']: valid=False; reasons.append(f'{node}:missing_phase')
    if x['kernel_device_coverage']<1: valid=False; reasons.append(f'{node}:kernel_devices={x["kernel_devices_in_nvtx"]}')
    if x['capture_processing_errors']: valid=False; reasons.append(f'{node}:processing_error')
    if x['kernel_events_in_nvtx']==0: valid=False; reasons.append(f'{node}:no_kernel_events')
result['world16_usable']=valid
result['validity_reasons']=reasons
json.dump(result,open(RUN/'analysis/nsys_no_offload_summary.json','w'),indent=2)
with open(RUN/'analysis/nsys_no_offload_summary.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['node','scope','metric','value']); w.writeheader(); w.writerows(rows)
print(json.dumps(result,indent=2))
