"""Small local rendering of saved interpretation summaries; no inference."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'results/interpretability_deep_20261017'
OUT = ROOT / 'results/interpretability_explained_20261017'
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, INK = '#3977c4', '#d87735', '#16816a', '#203149'
def font(n, bold=False):
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf', n)
def canvas(title, subtitle):
    im=Image.new('RGB',(1500,900),'#fafbfd'); d=ImageDraw.Draw(im)
    d.text((65,35),title,font=font(36,True),fill=INK)
    d.text((65,88),subtitle,font=font(23),fill=INK)
    return im,d
def text(d,xy,s,size=22,color=INK,bold=False): d.text(xy,s,font=font(size,bold),fill=color)
def save(im,name): im.save(OUT/(name+'.png'))
linear=json.loads((SRC/'linear_summary.json').read_text())
ens=json.loads((SRC/'ensemble_summary.json').read_text())
im,d=canvas('1. What moves the linear forecast?', 'Mean absolute net contribution to the quarter-hour correction (EUR/MWh)')
for i,g in enumerate(linear['groups']):
    y=155+i*62; v=g['mean_abs_net_contribution']
    text(d,(65,y+5),g['group'],23)
    d.rectangle((430,y,430+v*245,y+35),fill=BLUE if i<2 else '#a9bed8')
    text(d,(445+v*245,y+3),f'{v:.2f}',23,bold=True)
text(d,(65,745),'Similar past days supply the largest shape corrections; current ramps come next.',26,bold=True)
text(d,(65,790),'One saved linear checkpoint, September. Contributions can cancel.',23)
text(d,(65,825),'These are forecast movements, not feature accuracy gains or causal effects.',23)
save(im,'01_drivers')

examples=json.loads((SRC/'linear_examples.json').read_text())
im,d=canvas('2. The same correction can help or overshoot', 'Observed prices versus interpolation and the explained linear model')
for panel,e in enumerate(examples[1:3]):
    x0=90+panel*735; y0=205; w=590; h=390
    vals=e['baseline']+e['predicted']+e['observed']; lo=min(vals)-20; hi=max(vals)+20
    def point(q,v): return (x0+q*w/3,y0+h-(v-lo)/(hi-lo)*h)
    text(d,(x0,145),f"{e['date']} | {e['zone']} | hour {e['hour']}",24,bold=True)
    for j in range(5):
        v=lo+(hi-lo)*j/4; yy=point(0,v)[1]
        d.line((x0,yy,x0+w,yy),fill='#dce3ec',width=2); text(d,(x0-55,yy-10),f'{v:.0f}',18)
    for key,c in [('baseline',ORANGE),('predicted',BLUE),('observed',GREEN)]:
        pts=[point(q,v) for q,v in enumerate(e[key])]; d.line(pts,fill=c,width=5)
        for x,y in pts: d.ellipse((x-6,y-6,x+6,y+6),fill=c)
    for q in range(4): text(d,(x0+q*w/3-12,610),f'Q{q+1}',22)
    mae=lambda k:sum(abs(a-b) for a,b in zip(e[k],e['observed']))/4
    text(d,(x0,660),f"Hour MAE: {mae('baseline'):.2f} → {mae('predicted'):.2f} EUR/MWh",25,bold=True)
    text(d,(x0,705),'History sharpens the rise.' if panel==0 else 'History makes the rise too steep.',24)
for j,(s,c) in enumerate([('Observed',GREEN),('Interpolation',ORANGE),('Linear model',BLUE)]):
    text(d,(100+j*440,770),s,25,c,True)
text(d,(65,835),'Hindsight-selected best/worst improvements: illustrations, not typical-case estimates.',22)
save(im,'02_success_failure')

im,d=canvas('3. Where does the champion help most?', 'MAE reduction versus mean-corrected interpolation; equal weight to each of eight months')
for i,g in enumerate(ens['curve']):
    y=175+i*90; v=100*(1-g['equal_month_ratio'])
    text(d,(65,y+7),g['value'],25); d.rectangle((230,y,230+v*16,y+42),fill=BLUE)
    text(d,(245+v*16,y+5),f'{v:.1f}%',25,bold=True)
text(d,(900,160),'By predicted uncertainty',26,bold=True)
for i,g in enumerate(ens['uncertainty']):
    y=220+i*125; v=100*(1-g['equal_month_ratio'])
    text(d,(900,y),g['label'].title(),24)
    text(d,(1100,y),f'{v:.1f}% gain',24,GREEN,True)
    text(d,(900,y+40),f"Model MAE: {g['equal_month_mae']:.2f} EUR/MWh",22)
text(d,(65,690),'Ramps offer more room to improve than flat hours.',27,bold=True)
text(d,(65,745),'Wide-uncertainty cases gain more in relative terms, but remain less accurate in absolute terms.',23)
text(d,(65,805),'All uncertainty groups beat their matching interpolation baseline in all eight months.',23)
text(d,(65,845),'Developmental associations; uncertainty width does not cause the improvement.',22)
save(im,'03_where_it_helps')

ex=json.loads((SRC/'ensemble_examples.json').read_text())
e=next(e for e in ex if e['date']=='2026-06-01' and e['label']=='Largest cancellation')
im,d=canvas('4. How averaging cancels errors', 'Signed quarter-hour errors: above zero = overprediction; below zero = underprediction')
colors=[BLUE,ORANGE,'#9467ad']; labels=['MLP','Temporal CNN','Analog CNN']
series=[(labels[i],m['error'],colors[i]) for i,m in enumerate(e['members'].values())]
series.append(('Blend',[a-b for a,b in zip(e['blend'],e['observed'])],GREEN))
for q in range(4):
    x=180+q*330; text(d,(x,160),f'Q{q+1}',27,bold=True)
    for j,(label,errs,c) in enumerate(series):
        xx=x-65+j*40; yy=440-errs[q]*6
        d.rectangle((xx,min(440,yy),xx+28,max(440,yy)),fill=c)
        text(d,(xx-4,yy-25 if errs[q]>=0 else yy+5),f'{errs[q]:.1f}',17,c)
d.line((80,440,1420,440),fill=INK,width=2)
for j,(label,_,c) in enumerate(series): text(d,(90+j*350,690),label,24,c,True)
text(d,(65,750),f"This hour: weighted member MAE {e['weighted_member_mae']:.2f} → blend MAE {e['blend_mae']:.2f} EUR/MWh",25,bold=True)
text(d,(65,800),'Opposite errors cancel at Q1, Q2 and Q4. At Q3, all experts overpredict.',23)
text(d,(65,843),'Selected extreme example, June 1, ZONE_0, hour 21. Overall cancellation gain: 2.85% of member loss.',21)
save(im,'04_cancellation')
