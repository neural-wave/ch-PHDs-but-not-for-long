"""Render the problem overview from saved, unchanged illustrative predictions."""
import csv, json
from pathlib import Path
R=Path(__file__).resolve().parents[1]; O=R/'results/problem_main_figure_20261017'; O.mkdir(exist_ok=True)
d=list(csv.DictReader((R/'results/illustrative_quantile_bands_20261017/example_predictions.csv').open()))
y=[float(r['observed']) for r in d]; p=[float(r['q0.5']) for r in d]; lo=[float(r['lower90']) for r in d]; hi=[float(r['upper90']) for r in d]; h=[sum(y[i:i+4])/4 for i in range(0,96,4)]
g='#25ff00'; b='#28a9e2'; pink='#ff3854'; gold='#ffd43b'; s=['<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1120" viewBox="0 0 1800 1120"><rect width="1800" height="1120" fill="black"/><g font-family="Arial, sans-serif">']
def text(x,y,t,size=23,color=g,weight='normal'):
 s.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size * 1.35:.1f}" font-weight="{weight}">{t}</text>')
def line(x1,y1,x2,y2,c=g,w=2,opacity=1):s.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{c}" stroke-width="{w}" opacity="{opacity}"/>')
def panel(x,Y,w,H,xmin,xmax,ymin,ymax,xticks,yticks):
 X=lambda v:x+(v-xmin)/(xmax-xmin)*w; F=lambda v:Y+H-(v-ymin)/(ymax-ymin)*H
 for v in xticks:line(X(v),Y,X(v),Y+H,g,1,.13);text(X(v)-9,Y+H+29,str(v),19)
 for v in yticks:line(x,F(v),x+w,F(v),g,1,.13);text(x-47,F(v)+6,str(v),19)
 line(x,Y,x,Y+H);line(x,Y+H,x+w,Y+H)
 return X,F
def curve(X,F,xx,yy,c,w=3):s.append('<polyline fill="none" stroke="'+c+'" stroke-width="'+str(w)+'" points="'+' '.join(f'{X(a):.2f},{F(v):.2f}' for a,v in zip(xx,yy))+'"/>')
def band(X,F,xx,l,u):s.append('<polygon fill="'+g+'" fill-opacity=".18" stroke="'+g+'" stroke-opacity=".65" points="'+' '.join(f'{X(a):.2f},{F(v):.2f}' for a,v in list(zip(xx,u))+list(zip(xx,l))[::-1])+'"/>')
# Simplified single plot with an inset; keep the earlier overview exports intact.
s[0]=s[0].replace('height="1120"','height="860"').replace('1800 1120','1800 860')
text(90,65,'FROM HOURLY MEANS TO QUARTER-HOUR PRICES',38,weight='bold')
text(90,108,'One hourly mean. Four quarter-hour prices to recover.',25)
text(90,156,'11 June 2026 · ZONE_3',20)
line(500,150,540,150,gold,4);text(553,157,'Known hourly mean',20,gold)
line(830,150,870,150,b,3);text(883,157,'Quarter-hour target',20,b)
X,F=panel(90,195,1620,525,0,24,-10,80,range(0,25,2),[0,20,40,60,80])
xx=[i/4+.125 for i in range(96)]
curve(X,F,[a for i in range(24) for a in (i,i+1)],[v for v in h for _ in (0,1)],gold,4)
curve(X,F,xx,y,b,3)
text(760,780,'Hour of delivery day',22)
text(90,188,'EUR/MWh',18)
# Highlight the source hour and connect it to the inset.
s.append(f'<rect x="{X(21)}" y="{F(32)}" width="{X(22)-X(21)}" height="{F(-3)-F(32)}" fill="{gold}" fill-opacity=".05" stroke="{gold}" stroke-dasharray="7 6"/>')
line(X(21.5),F(32),1290,455,gold,1.5,.65)
s.append(f'<rect x="780" y="275" width="510" height="260" rx="6" fill="black" stroke="{g}" stroke-opacity=".6"/>')
text(810,307,'Zoom · 21:00–22:00',21,weight='bold')
A,B=panel(840,333,420,155,21,22,-5,30,[21,21.5,22],[0,15,30])
i=range(84,88); z=[xx[k] for k in i]; obs=[y[k] for k in i]
curve(A,B,[21,22],[h[21],h[21]],gold,3);curve(A,B,z,obs,b,3)
for a,v in zip(z,obs):s.append(f'<circle cx="{A(a)}" cy="{B(v)}" r="4" fill="{b}"/>')
text(90,830,'Reconstructed quarters must preserve each supplied hourly mean.',22,gold)
s.append('</g></svg>')
(O/'main_figure_inset.svg').write_text('\n'.join(s))
print(O/'main_figure_inset.svg')
