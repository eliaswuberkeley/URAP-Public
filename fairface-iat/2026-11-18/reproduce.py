"""Recompute published statistics and rebuild three charts from saved responses."""
import csv
import hashlib
import json
import os
import sys
from collections import Counter,defaultdict
from pathlib import Path

D=Path(__file__).resolve().parent
OUT=D/"runtime/rebuilt"
os.environ.setdefault("MPLCONFIGDIR",str(D/"runtime/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME",str(D/"runtime/cache"))
sys.path.insert(0,str(D.parent))
from iat_protocol import POSITIVE,NEGATIVE,annotate
import numpy as np

SIGNS=("positive","neutral","negative");RACES=("Black","White")

def csvrows(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def read_responses():
    roster={r["image_id"]:r for r in csvrows(D.parent/"data/curated_images.csv")}
    rows=csvrows(D/"data/responses.csv")
    for r in rows:
        meta=roster[r["image_id"]]
        assert meta["split"]=="test" and all(r[k]==meta[k] for k in ("source_race","source_gender","age_band","pair_id","image_sha256"))
        for k,v in annotate(r["sampled_raw"],r["fallback_raw"]).items():
            assert r[k]==str(v)
            r[k]=v
        r["iteration"]=int(r["iteration"])
        key=f'10062026|iat|{r["image_id"]}|{r["iteration"]}'
        assert int(r["seed"])==int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%(2**31-1)
    ids={r["image_id"] for r in rows}
    assert len(rows)==720 and len(ids)==144 and {(r["image_id"],r["iteration"]) for r in rows}=={(i,j) for i in ids for j in range(1,6)}
    for sign in SIGNS:
        pairs=defaultdict(set)
        for r in rows:
            if r["expression"]==sign:pairs[r["pair_id"]].add((r["image_id"],r["source_race"]))
        assert len(pairs)==24 and all(len(v)==2 and {race for _,race in v}==set(RACES) for v in pairs.values())
    return rows

def exact_p(ds):
    # Each nonzero integer difference can take either sign under paired exchangeability.
    dist=Counter({0:1})
    for d in ds:
        if not d:continue
        nxt=Counter()
        for k,n in dist.items():nxt[k+d]+=n;nxt[k-d]+=n
        dist=nxt
    return sum(n for k,n in dist.items() if abs(k)>=abs(sum(ds)))/sum(dist.values())

def rates(sub,strict):
    field,metric=("sampled_strict_valid","sampled_negative_count") if strict else ("valid_three","negative_count")
    by=defaultdict(lambda:defaultdict(list))
    for r in sub:
        if r[field]:by[r["pair_id"]][r["source_race"]].append(r[metric])
    prs=[];ds=[]
    for k,rr in sorted(by.items()):
        if any(not rr[g] for g in RACES):continue
        # 60 is divisible by each possible number (1..5) of valid responses.
        b,w=[sum(rr[g])*(60//len(rr[g])) for g in RACES]
        ds.append(b-w)
        prs.append(dict(pair_id=k,black_valid=len(rr["Black"]),white_valid=len(rr["White"]),
                       black_negative=sum(rr["Black"]),white_negative=sum(rr["White"]),
                       black_pct=b*100/180,white_pct=w*100/180,gap_pp=(b-w)*100/180))
    assert prs
    rng=np.random.default_rng(20261118);arr=np.array(ds)*100/180
    ci=np.percentile(rng.choice(arr,(100000,len(ds))).mean(axis=1),[2.5,97.5]).tolist()
    s=dict(pairs=len(ds),black_pct=float(np.mean([p["black_pct"] for p in prs])),
           white_pct=float(np.mean([p["white_pct"] for p in prs])),gap_pp=float(arr.mean()),ci95_pp=ci,
           exact_p=exact_p(ds),black_higher=sum(d>0 for d in ds),white_higher=sum(d<0 for d in ds),
           ties=ds.count(0),both_zero=sum(p["black_negative"]==p["white_negative"]==0 for p in prs),
           valid_by_race={g:sum(r[field] for r in sub if r["source_race"]==g) for g in RACES})
    return s,prs

def main():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    rows=read_responses()
    OUT.mkdir(parents=True,exist_ok=True)
    result={};per_pair=[];word_rows=[]
    all_counts=Counter((r["expression"],r["source_race"],w) for r in rows if r["valid_three"] for w in r["words"].split("|"))
    chart_max=10*((max(all_counts.values())+9)//10)+5
    for sign in SIGNS:
        sub=[r for r in rows if r["expression"]==sign];assert len(sub)==240
        item=result[sign]=dict(responses=len(sub),generation_calls=sum(r["generation_calls"] for r in sub),
                                fallbacks=sum(not r["sampled_valid"] for r in sub))
        for strict in (False,True):
            name="sampled_strict" if strict else "primary"
            s,prs=rates(sub,strict);item[name]=s
            per_pair.extend(dict(expression=sign,analysis=name,**p) for p in prs)
        counts={g:Counter(w for r in sub if r["source_race"]==g and r["valid_three"] for w in r["words"].split("|")) for g in RACES}
        ww=[dict(expression=sign,word=w,valence=v,Black=counts["Black"][w],White=counts["White"][w])
            for v,vocab in (("positive",POSITIVE),("negative",NEGATIVE))
            for w in sorted(vocab,key=lambda w:(-sum(counts[g][w] for g in RACES),w))]
        word_rows+=ww;s=item["primary"]
        item["raw_negative"]={g:sum(counts[g][w] for w in NEGATIVE) for g in RACES}
        for g in RACES:assert sum(counts[g].values())==3*s["valid_by_race"][g]
        fig,ax=plt.subplots(figsize=(17,6.2));x=np.arange(len(ww));cut=len(POSITIVE)
        ax.axvspan(-.5,cut-.5,color="#eef6f0");ax.axvspan(cut-.5,len(x)-.5,color="#fbf2f0")
        ax.axvline(cut-.5,color="#b9b8b3",linewidth=.8)
        for g,dx,c in (("Black",-.20,"#17667b"),("White",.20,"#c7752a")):
            ax.bar(x+dx,[r[g] for r in ww],.40,color=c,label=g+" portraits")
        ax.set(xlim=(-.55,len(x)-.45),ylim=(0,chart_max),ylabel="Times word was selected")
        ax.set_xticks(x,[r["word"] for r in ww],rotation=63,ha="right",fontsize=8);ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[["right","top"]].set_visible(False);ax.legend(frameon=False)
        fig.suptitle(f"Base Gemma: {sign}-expression FairFace pairs",fontsize=19,y=.98)
        lo,hi=s["ci95_pp"]
        fig.text(.5,.89,f'Negative word rate: Black {s["black_pct"]:.2f}% vs White {s["white_pct"]:.2f}%.  Matched gap {s["gap_pp"]:+.2f} pp (95% pair bootstrap interval {lo:+.2f} to {hi:+.2f}).',ha="center",fontsize=12)
        ax.set_title("Positive words (left) and negative words (right), sorted by frequency within each group",fontsize=10,pad=9)
        fig.text(.5,.025,"24 manually approved pairs; five three-word responses per portrait. Counts are word selections, not numbers of people.",ha="center",fontsize=10)
        fig.subplots_adjust(top=.79,bottom=.235,left=.055,right=.99);fig.savefig(OUT/f"iat61_{sign}.png",dpi=160);plt.close(fig)
    adj=0.
    for j,sign in enumerate(sorted(SIGNS,key=lambda k:result[k]["primary"]["exact_p"])):
        s=result[sign]["primary"];adj=max(adj,min(1,(3-j)*s["exact_p"]));s["holm_p"]=adj
    assert json.loads((D/"results/statistics.json").read_text())==result
    for name,records in (("pair_rates",per_pair),("word_counts",word_rows)):
        assert csvrows(D/"results"/(name+".csv"))==[{k:str(v) for k,v in r.items()} for r in records]
    print(f"Verified all 720 responses, pair rates, word counts and statistics. Rebuilt charts: {OUT}")
if __name__=="__main__":main()
