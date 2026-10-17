"""Promote the current research champion and rank saved evaluation recipes.

Metadata consolidation only: no fitting, inference or new test-driven tuning.
Every primary score is recomputed from the eight saved monthly MAEs.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'results/model_ranking_20261017'
MONTHS = [f'2026-{m:02d}' for m in range(2, 10)]
DAYS = [28, 30, 30, 31, 30, 31, 31, 30]
SOURCES = [
    'normalized_validation_20261017', 'monthly_ratio_20261017',
    'monthly_quantiles_20261017', 'zone_weights_20261017',
    'ssm_monthly_20261017', 'normalization_20261017',
    'linear_s4d_20261017', 'ensemble_normalization_20261017',
    'minimize_expanding_20261017', 'riccardo_comparison_20261017',
]
TITLES = {
    'normalized_validation_20261017': 'Original expanding search',
    'monthly_ratio_20261017': 'Point-model monthly-ratio search',
    'monthly_quantiles_20261017': 'Original quantile ensemble',
    'zone_weights_20261017': 'Quantile zone-weight study',
    'ssm_monthly_20261017': 'Matched SSM comparison',
    'normalization_20261017': 'CNN/SSM normalization study',
    'linear_s4d_20261017': 'One-layer affine S4D',
    'ensemble_normalization_20261017': 'Controlled six-expert normalization',
    'minimize_expanding_20261017': 'Fixed-scale ensemble with seed averaging',
    'riccardo_comparison_20261017': 'Riccardo chronological adaptation',
}


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(''); return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family(campaign, recipe):
    text = recipe.lower()
    if 'selector' in text:
        return 'Selection policy'
    if text in ['interpolation', 'interpolation_fixed', 'original_interpolation']:
        return 'Corrected interpolation'
    if any(x in text for x in ['blend', 'global_', 'zone_', 'common_']):
        return 'Ensemble'
    if campaign == 'linear_s4d_20261017':
        return 'Affine S4D'
    if 's4d' in text:
        return 'S4D'
    if text.startswith('s5'):
        return 'S5'
    if 'gru' in text:
        return 'GRU'
    if 'cnn2d' in text:
        return 'Analog 2D CNN'
    if 'cnn' in text or text == 'tcn' or (campaign in ['normalization_20261017', 'ssm_monthly_20261017'] and text.startswith('compact')):
        return 'Temporal CNN'
    if text.startswith('riccardo_'):
        return 'Riccardo zonal CNN'
    if 'linear_quantile' in text:
        return 'Linear quantile model'
    if 'additive' in text:
        return 'Additive model'
    if 'ridge' in text:
        return 'Ridge regression'
    if 'linear' in text:
        return 'Linear model'
    if 'mlp' in text:
        return 'MLP'
    if text == 'hierarchical':
        return 'Hierarchical model'
    if text == 'direct':
        return 'Direct neural model'
    if 'analog' in text:
        return 'Causal analog rule'
    return recipe


def alias(campaign, recipe):
    canonical = 'monthly_quantiles_20261017/'
    if recipe in ['interpolation', 'interpolation_fixed', 'original_interpolation']:
        target = 'normalized_validation_20261017/interpolation'
        return '' if campaign+'/'+recipe == target else target
    if campaign == 'minimize_expanding_20261017' and recipe.startswith('standard_'):
        return canonical+recipe[len('standard_'):]
    if campaign == 'zone_weights_20261017' and recipe.startswith('original_'):
        return canonical+recipe[len('original_'):]
    if campaign == 'ensemble_normalization_20261017':
        if recipe == 'blend__standard':
            return canonical+'blend_uniform'
        if recipe.endswith('__standard'):
            return canonical+recipe[:-len('__standard')]+'_uniform'
    return ''


def collect_primary():
    base_rows = read_csv(PROJECT/'results/normalized_validation_20261017/monthly_metrics.csv')
    baseline = {r['test_month']: float(r['mae']) for r in base_rows if r['method'] == 'interpolation'}
    assert set(baseline) == set(MONTHS) and min(baseline.values()) > 0
    entries = []; monthly_out = []; receipts = []
    for campaign in SOURCES:
        root = PROJECT/'results'/campaign
        summary_path = root/'summary.csv'
        monthly_path = root/('monthly_metrics.csv' if campaign in ['normalized_validation_20261017', 'riccardo_comparison_20261017'] else 'monthly_scores.csv')
        audit_path = root/('independent_audit.json' if campaign == 'riccardo_comparison_20261017' else 'audit.json')
        audit = read_json(audit_path)
        assert audit.get('passed') is True and not audit.get('smoke', False), campaign
        summary = read_csv(summary_path); monthly = read_csv(monthly_path)
        receipts.append(dict(campaign=campaign, summary=str(summary_path.relative_to(PROJECT)),
                             summary_sha256=digest(summary_path), monthly_sha256=digest(monthly_path),
                             audit=str(audit_path.relative_to(PROJECT)), audit_sha256=digest(audit_path),
                             rows=len(summary), audit_passed=True))
        for score in summary:
            recipe = score.get('recipe', score.get('method')); identifier = campaign+'/'+recipe
            group = [r for r in monthly if r.get('recipe', r.get('method')) == recipe]
            group.sort(key=lambda r: r.get('month', r.get('test_month')))
            assert [r.get('month', r.get('test_month')) for r in group] == MONTHS, identifier
            assert [int(r['test_days']) for r in group] == DAYS, identifier
            maes = [float(r['mae']) for r in group]
            ratios = [mae/baseline[m] for mae, m in zip(maes, MONTHS)]
            assert all(math.isfinite(v) for v in maes+ratios)
            for r, m, value in zip(group, MONTHS, ratios):
                if 'interpolation_mae' in r:
                    assert abs(float(r['interpolation_mae'])-baseline[m]) < 1e-8, identifier
                assert abs(value-float(r.get('ratio', r.get('mae_over_interpolation_mae')))) < 1e-8, identifier
                monthly_out.append(dict(model_id=identifier, month=m, test_days=int(r['test_days']),
                                        mae=float(r['mae']), interpolation_mae=baseline[m], ratio=value))
            mean = sum(ratios)/8; pooled = sum(v*n for v, n in zip(maes, DAYS))/241
            assert int(score['folds']) == 8
            assert abs(mean-float(score['mean_monthly_ratio'])) < 1e-8, identifier
            assert abs(pooled-float(score['pooled_mae'])) < 1e-8, identifier
            quantiles = campaign not in ['monthly_ratio_20261017', 'riccardo_comparison_20261017', 'normalized_validation_20261017'] or recipe == 'linear_quantile_median'
            note = 'Developmental; candidate budgets differ across campaigns.'
            if campaign == 'zone_weights_20261017':
                note += ' Expert early stopping already used all selection days; internal tuning is not independently nested.'
            if recipe == 'linear_quantile_median':
                note += ' Inner selection used joint MAE/pinball objective.'
            entries.append(dict(model_id=identifier, campaign=campaign, campaign_title=TITLES[campaign],
                                recipe=recipe, family=family(campaign, recipe),
                                recipe_kind='pre_test_policy' if 'selector' in recipe else 'fixed_recipe',
                                output_type='ordered_quantiles' if quantiles else 'point_only_historical',
                                mean_monthly_ratio=mean, improvement_percent=100*(1-mean), pooled_mae=pooled,
                                pooled_mae_ratio=pooled/(sum(baseline[m]*n for m, n in zip(MONTHS, DAYS))/241),
                                worst_month_ratio=max(ratios), months_beating_interpolation=sum(v < 1-1e-12 for v in ratios),
                                coverage90=float(score['coverage90']) if score.get('coverage90') else None,
                                width90=float(score['width90']) if score.get('width90') else None,
                                folds=8, test_days=241, targets=92544, official_only=True, audit_passed=True,
                                alias_of=alias(campaign, recipe), source=str(summary_path.relative_to(PROJECT)),
                                audit_source=str(audit_path.relative_to(PROJECT)), notes=note))
    # No fitting is needed for the original hourly-repeat baseline.
    baseline_path=PROJECT/'results/monthly_baseline_mae.csv'
    baseline_audit=read_json(PROJECT/'results/normalized_validation_20261017/audit.json')
    assert digest(baseline_path)==baseline_audit['sources']['results/monthly_baseline_mae.csv']
    raw = {r['month']: r for r in read_csv(baseline_path) if r['month'] in MONTHS}
    assert [int(raw[m]['days']) for m in MONTHS]==DAYS
    flat = [float(raw[m]['hourly_mean']) for m in MONTHS]
    ratio = [v/baseline[m] for v, m in zip(flat, MONTHS)]
    entries.append(dict(model_id='deterministic/hourly_repeat', campaign='deterministic', campaign_title='Deterministic control',
                        recipe='hourly_repeat', family='Hourly-repeat baseline', recipe_kind='fixed_recipe', output_type='point_only_historical',
                        mean_monthly_ratio=sum(ratio)/8, improvement_percent=100*(1-sum(ratio)/8),
                        pooled_mae=sum(v*n for v,n in zip(flat,DAYS))/241,
                        pooled_mae_ratio=(sum(v*n for v,n in zip(flat,DAYS))/sum(baseline[m]*n for m,n in zip(MONTHS,DAYS))),
                        worst_month_ratio=max(ratio), months_beating_interpolation=0, coverage90=None, width90=None,
                        folds=8,test_days=241,targets=92544,official_only=True,audit_passed=True,alias_of='',
                        source='results/monthly_baseline_mae.csv',audit_source='results/normalized_validation_20261017/audit.json',
                        notes='Historical deterministic control; preserves each supplied hourly mean.'))
    for month, n, mae, value in zip(MONTHS,DAYS,flat,ratio):
        monthly_out.append(dict(model_id='deterministic/hourly_repeat',month=month,test_days=n,mae=mae,interpolation_mae=baseline[month],ratio=value))
    lookup = {r['model_id']: r for r in entries}; assert len(lookup) == len(entries)
    for r in entries:
        if r['alias_of']:
            target = lookup[r['alias_of']]
            assert abs(r['mean_monthly_ratio']-target['mean_monthly_ratio']) < 1e-10
            assert abs(r['pooled_mae']-target['pooled_mae']) < 1e-10
    ranked = sorted([r for r in entries if not r['alias_of']],key=lambda r:(r['mean_monthly_ratio'],r['model_id']))
    previous=None; rank=0
    for index, r in enumerate(ranked, 1):
        if previous is None or abs(r['mean_monthly_ratio']-previous)>1e-12:
            rank=index;previous=r['mean_monthly_ratio']
        r['rank']=rank
    for r in entries:
        if r['alias_of']:r['rank']=lookup[r['alias_of']]['rank']
    entries.sort(key=lambda r:(r['rank'],bool(r['alias_of']),r['model_id']))
    return ranked, entries, monthly_out, receipts


def historical_and_catalog():
    history=[]; catalog=[]; inventory=[]; diagnostic=[]
    def add(campaign, recipe, mae, baseline, scope, source, selection='', external=False):
        assert math.isfinite(float(mae))
        history.append(dict(model_id=campaign+'/'+recipe,campaign=campaign,recipe=recipe,scope=scope,
                            mae=float(mae),ratio=float(mae)/float(baseline) if baseline else None,
                            selection=selection,official_only=not external,primary_rank=None,source=source,
                            notes='Different evaluation scope; excluded from primary eight-fold promotion.'))
    september_base=7.090321234809027
    for root in sorted((PROJECT/'results').iterdir()):
        if not root.is_dir() or root == OUT:continue
        metrics=root/'metrics.json'; campaign=root.name
        inventory.append(dict(campaign=campaign,primary_table=campaign in SOURCES,metrics_available=metrics.exists(),
                              summary_available=(root/'summary.csv').exists(),
                              audit_available=(root/'audit.json').exists() or (root/'independent_audit.json').exists(),
                              status='matched_eight_fold' if campaign in SOURCES else 'historical_or_diagnostic_or_incomplete'))
        if not metrics.exists():continue
        data=read_json(metrics); source=str(metrics.relative_to(PROJECT))
        def walk(value, pointer=''):
            if isinstance(value,dict):
                cfg=value.get('config')
                name=value.get('name')
                scalar={k:v for k,v in value.items() if isinstance(v,(int,float)) and not isinstance(v,bool) and
                        k in ['validation_mae','selection_mae','selection_joint','selection_ratio','test_mae','september_mae','best_epoch','epochs','epochs_run','parameters']}
                if scalar and (cfg is not None or name is not None or pointer.endswith('/selected')):
                    catalog.append(dict(campaign=campaign,source=source,json_pointer=pointer or '/',
                                        configuration=json.dumps(cfg if cfg is not None else {'name':name},sort_keys=True),
                                        scores=json.dumps(scalar,sort_keys=True),primary_rank=None,
                                        note='Individual candidate; recorded scores only. No inferred eight-fold score.'))
                for k,v in value.items():
                    if k not in ['daily_mae','by_zone','train_dates','test_dates','fit_dates','calibration_dates','selection_dates']:
                        walk(v,pointer+'/'+str(k).replace('~','~0').replace('/','~1'))
            elif isinstance(value,list):
                if value and isinstance(value[0],(dict,list)):
                    for i,v in enumerate(value):walk(v,pointer+'/'+str(i))
        walk(data)
        if campaign in ['search_gpu1','rich_search_gpu1','zone_search_gpu1','catboost_search_gpu1','projection_search_gpu1','smoke_gpu1']:
            if data.get('holdout',{}).get('mae') is not None:
                add(campaign,'selected_ensemble',data['holdout']['mae'],data['holdout_baseline']['mae'],
                    'Historical 60-day holdout: 2026-08-02..2026-09-30',source,'2026-06-03..2026-08-01')
        elif campaign.startswith('monthly_') and campaign.endswith('_gpu1'):
            score=data.get('test',data.get('holdout',{}));base=data.get('test_baseline',data.get('holdout_baseline',{}))
            if score.get('mae') is not None:
                add(campaign,'selected_pipeline',score['mae'],base.get('mae',september_base),'September 2026 only',source,'Full August')
        elif campaign=='simple_5090':
            for name,row in data['by_family'].items():
                add(campaign,name,row['september_mae'],september_base,'September 2026 only',source,'August1–15')
        elif campaign=='interpretable_vast':
            row=data['selected'];add(campaign,row['name'],row['september_mae'],september_base,'September 2026 only',source,'August1–15')
        elif campaign in ['hierarchical_vast','hierarchical_hpo_vast','hierarchical_hpo_lags_vast']:
            add(campaign,data['selected'],data['test_mae'],september_base,'September 2026 only',source,'August1–15')
        elif campaign in ['simple_joint_zone_cnn_20261017','parallel_2d_zone_cnn_20261017','joint_zone_cnn_20261017']:
            for name,row in data['by_history'].items():
                add(campaign,'history_'+name,row['september_mae'],september_base,'September 2026 only',source,'Full August')
        elif campaign=='parallel_2d_tuning_20261017':
            for name,row in data['by_arm'].items():add(campaign,name,row['september_mae'],september_base,'September 2026 only',source,'Full August; differing budget')
        elif campaign=='quantile_2d_cnn_20261017':
            for name,row in data['by_group'].items():add(campaign,name,row['september_mae'],september_base,'September 2026 only',source,'August1–15; cohort winners')
        elif campaign in ['four_frequency_20261017','segmented_frequency_20261017']:
            for name,row in data.items():
                if 'september' in row:add(campaign,name,row['september']['mae'],september_base,'September 2026 only',source,'Full August')
                if 'oracle_september' in row:
                    diagnostic.append(dict(campaign=campaign,recipe=name,mae=row['oracle_september']['mae'],source=source,
                                           reason='Target-derived oracle weights; representation bound, not a deployable predictor.'))
        elif campaign=='flowstate_20261017':
            for name,row in data['methods'].items():
                add(campaign,name,row['september_mae'],september_base,'Exploratory pretrained comparison: September only',source,
                    'Full August; pretrained overlap unaudited',external='flowstate' in name)
        elif campaign=='ssm_pilot_20261017':
            for name,row in data['by_family'].items():
                add(campaign,name,row['september_mae'],september_base,'Superseded SSM pilot: September only',source,'Full August; later eight-fold campaign supersedes pilot')
            for name,row in data['rolling'].items():
                add(campaign,'rolling_'+name,row['mae'],None,'Superseded SSM pilot: four earlier rolling folds',source,
                    'Fixed width32/lr.001/seed17; different folds, no eight-month ratio')
        # Explicitly catalogue fixed-September training-size diagnostics separately.
        if campaign in ['expanding_vast','expanding_simple_5090','linear_quantiles_vast']:
            for fold in data.get('folds',[]):
                meta=fold.get('manifest',fold)
                if meta.get('plan')=='expanding':continue
                selected=fold.get('selected',{})
                if campaign=='linear_quantiles_vast':
                    add(campaign,f"fixed_history_{meta['training_months']}_linear_quantile",fold['test_metrics']['mae'],september_base,
                        'Fixed-September training-history diagnostic',source,'Joint MAE/pinball selection; repeated September target set')
                for name,row in selected.items():
                    if isinstance(row,dict) and row.get('test_mae') is not None:
                        add(campaign,f"fixed_history_{meta.get('months',meta.get('training_months'))}_{name}",row['test_mae'],september_base,
                            'Fixed-September training-history diagnostic',source,'Eight histories reuse September; not eight independent tests')
            if campaign=='expanding_vast':
                for row in data.get('summary',[]):
                    if row.get('plan')=='fixed_september':
                        add(campaign,'history_average_'+row['family'],row['weighted_test_mae'],september_base,
                            'Fixed-September history diagnostic: average over training histories',source,
                            'Diagnostic average, not one fitted model or independent folds')
        if campaign.endswith('_smoke') or data.get('smoke',False):
            inventory[-1]['status']='smoke_only_not_ranked'
    # Trial files contain scores not necessarily repeated in campaign metrics.
    known={(r['source'],r['json_pointer']) for r in catalog}
    for p in sorted((PROJECT/'results').glob('**/trials/*/metrics.json')):
        row=read_json(p);source=str(p.relative_to(PROJECT)); cfg=row.get('config',{})
        if (source,'/') not in known:
            scalar={k:v for k,v in row.items() if k in ['validation_mae','selection_mae','selection_ratio','test_mae','september_mae','best_epoch','parameters'] and isinstance(v,(int,float))}
            catalog.append(dict(campaign=p.relative_to(PROJECT/'results').parts[0],source=source,json_pointer='/',configuration=json.dumps(cfg,sort_keys=True),
                                scores=json.dumps(scalar,sort_keys=True),primary_rank=None,note='Individual stored trial; no inferred eight-fold score.'))
    # Ensure smoke results cannot be mistaken for full historical winners.
    history=[r for r in history if not r['campaign'].endswith('_smoke') and r['campaign']!='smoke_gpu1']
    history.sort(key=lambda r:(r['scope'],r['mae'],r['model_id']))
    for scope in sorted(set(r['scope'] for r in history)):
        for rank,row in enumerate([r for r in history if r['scope']==scope],1):row['rank_within_scope']=rank
    return history,catalog,inventory,diagnostic


def display_name(row):
    return TITLES.get(row['campaign'],row.get('campaign_title',row['campaign']))+' · '+row['recipe']


def promote(ranked):
    champion=ranked[0]
    assert champion['model_id']=='minimize_expanding_20261017/fixed_blend_ratio', 'A newer result won; inspect before promotion'
    policies=[r for r in ranked if r['recipe_kind']=='pre_test_policy']
    policy=min(policies,key=lambda r:r['mean_monthly_ratio'])
    source=PROJECT/'results/minimize_expanding_20261017'
    backup=read_json(source/'backup_verification.json');archive=Path(backup['archive'])
    assert backup['verified'] and archive.stat().st_size==backup['bytes']
    fold_path=OUT/'champion_per_fold.json'
    exact=read_json(fold_path)
    assert exact['model_id']==champion['model_id']
    assert [f['test_month'] for f in exact['folds']]==MONTHS
    assert [f['training_days'] for f in exact['folds']]==[91,122,150,180,210,241,271,302]
    assert len(exact['quantiles'])==11 and exact['quantiles'][5]==0.5
    for fold in exact['folds']:
        assert abs(sum(r['weight'] for r in fold['leaves'])-1)<1e-12
        assert min(r['weight'] for r in fold['leaves'])>0
        assert all(len(r['model_sha256'])==64 for r in fold['leaves'])
    result=dict(schema_version=1,promoted_on='2026-10-17',status='current_research_champion',
                promotion_authority='User explicitly requested promotion of the winning model and a complete ranking.',
                validation_protocol='VALIDATION_PROTOCOL.md',validation_required_for_future_models=True,
                validation_protocol_sha256=digest(PROJECT/'VALIDATION_PROTOCOL.md'),
                metric='Arithmetic mean of eight monthly MAE / mean-corrected linear-interpolation MAE ratios; lower is better',
                model_id=champion['model_id'],campaign=champion['campaign'],recipe=champion['recipe'],
                mean_monthly_ratio=champion['mean_monthly_ratio'],pooled_mae=champion['pooled_mae'],
                improvement_percent=champion['improvement_percent'],coverage90=champion['coverage90'],width90=champion['width90'],
                trained_from_scratch=True,training_months=[3,10],test_months=MONTHS,targets=92544,
                model='Quantile residual ensemble; fixed physical scaling, ordinary/ratio fitting objectives and validation-selected seed means.',
                blend_rule='Within each fold, preceding selection dates choose fixed rich-linear pair or equal top2/3/4/6 family blend; ratio-trained experts for this champion.',
                components='Often equal MLP/temporal-CNN/analog-2D-CNN; February rich-linear/MLP, March six experts. Exact fold selections are in the recovery archive.',
                selection_scope='Best observed developmental recipe after comparing aggregate test scores; not a blind test or a preselected across-campaign winner.',
                best_pre_test_policy=dict(model_id=policy['model_id'],mean_monthly_ratio=policy['mean_monthly_ratio'],
                                          pooled_mae=policy['pooled_mae'],note='Per-fold choices use prior data; selecting this policy across campaigns is still developmental.'),
                audit='results/minimize_expanding_20261017/audit.json',
                backup_verification='results/minimize_expanding_20261017/backup_verification.json',
                archive=str(archive.relative_to(PROJECT)),archive_bytes=backup['bytes'],archive_sha256=backup['sha256'],
                artifact_paths_in_archive='results/minimize_expanding_20261017/folds/<month>/selection.json and trials/*/model.pt',
                exact_fold_recipes='results/model_ranking_20261017/champion_per_fold.json',
                exact_fold_recipes_sha256=digest(fold_path),
                execution_host='root@93.91.156.85:55340',remote_project='/workspace/HACKATON',
                rank_source='results/model_ranking_20261017/primary_ranking.csv',
                inference_entrypoint_status='Research champion registry; an all-history inference refit of this recipe has not been made.',
                existing_inference_manifest='results/deployment_best.json')
    write_json(PROJECT/'best_model.json',result)
    return result


def docs(ranked,all_rows,historical,catalog,inventory,diagnostics,champion):
    lines=['# Model ranking — 17 October 2026','',
           'Standing rule: every future predictive comparison and research promotion uses [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md). '
           'Eight expanding February–September tests, preceding-month days 1–15 selection/16–end calibration, fresh fitting, and equal-month MAE/mean-corrected-interpolation MAE. Lower is better.', '',
           f"**Current research champion:** `{champion['model_id']}` — **{champion['mean_monthly_ratio']:.6f}**, "
           f"**{champion['improvement_percent']:.2f}% less MAE than interpolation**, pooled MAE **{champion['pooled_mae']:.6f} EUR/MWh**.", '',
           'This user-authorized promotion is the best observed developmental recipe. It does not claim blind-test generalization or statistical significance. '
           'The strongest per-fold prior-data selection policy is separately identified in [best_model.json](best_model.json).', '',
           f"Coverage: **{len(all_rows)}** reported eight-fold rows, **{len(ranked)}** rows after declared reused-control aliases, "
           f"**{len(historical)}** historical recipe/scope rows and **{len(catalog)}** locally available individual candidate records. "
           'The catalogue can repeat a trial across campaign and per-trial sources. '
           'Raw HPO trials without matched eight-fold evaluations receive no primary rank. Search budgets and inner objectives differ; this ranks complete saved recipes.', '',
           '## Leading eight-fold recipes','',
           '| Rank | Model recipe | Mean monthly ratio | Reduction | Pooled MAE | Output |',
           '|---:|---|---:|---:|---:|---|']
    for r in ranked[:25]:
        lines.append(f"| {r['rank']} | {display_name(r)} | {r['mean_monthly_ratio']:.6f} | {r['improvement_percent']:.2f}% | {r['pooled_mae']:.6f} | {'11 quantiles' if r['output_type']=='ordered_quantiles' else 'Historical point control'} |")
    lines += ['', '## Best result per model family','',
              'Descriptive winners within each family; normalization, features, loss and ensemble search spaces differ.', '',
              '| Family | Best observed recipe | Mean monthly ratio | Pooled MAE |', '|---|---|---:|---:|']
    families={}
    for r in ranked:families.setdefault(r['family'],r)
    for family_name,r in sorted(families.items(),key=lambda v:v[1]['mean_monthly_ratio']):
        lines.append(f"| {family_name} | `{r['model_id']}` | {r['mean_monthly_ratio']:.6f} | {r['pooled_mae']:.6f} |")
    lines += ['', '## Complete primary ranking','',
              'All rows below have the same eight months/241 days/92,544 targets and a passed owning audit. '
              'Declared reused controls are retained in all_reported_models.csv with alias targets, rather than presented as independent improvements.', '',
              '| Rank | Recipe ID | Mean monthly ratio | Pooled MAE | Worst month | Evidence |',
              '|---:|---|---:|---:|---:|---|']
    for r in ranked:
        lines.append(f"| {r['rank']} | `{r['model_id']}` | {r['mean_monthly_ratio']:.6f} | {r['pooled_mae']:.6f} | {r['worst_month_ratio']:.6f} | [scores]({r['source']}) |")
    for scope in sorted(set(r['scope'] for r in historical)):
        lines += ['', '## '+scope,'',
                  'Ranks apply only within this scope. These rows do not enter the primary eight-fold promotion.', '',
                  '| Rank within scope | Recipe | MAE | Ratio to same-scope interpolation | Selection |',
                  '|---:|---|---:|---:|---|']
        for r in [v for v in historical if v['scope']==scope]:
            ratio=f"{r['ratio']:.6f}" if r['ratio'] is not None else 'Unavailable'
            lines.append(f"| {r['rank_within_scope']} | `{r['model_id']}` | {r['mae']:.6f} | {ratio} | {r['selection']} |")
    lines += ['', '## Unranked diagnostics and coverage','',
              f"{len(diagnostics)} target-derived frequency/PCA oracle results are listed in oracle_diagnostics.csv. They measure representation bounds and cannot become champions. "
              'Exploratory pretrained FlowState results have a separate scope and are excluded from eligible challenge promotion. '
              'Fixed-September history sizes reuse one test month. Smokes, superseded pilots, incomplete results and all-data deployment refits have no held-out primary rank.', '',
              'Inventory and full data: [primary_ranking.csv](results/model_ranking_20261017/primary_ranking.csv), '
              '[all reported rows and aliases](results/model_ranking_20261017/all_reported_models.csv), '
              '[monthly scores](results/model_ranking_20261017/monthly_scores.csv), '
              '[historical scopes](results/model_ranking_20261017/historical_ranking.csv), '
              '[candidate catalogue](results/model_ranking_20261017/candidate_catalog.csv), '
              '[campaign inventory](results/model_ranking_20261017/campaign_inventory.csv), '
              '[provenance/audit](results/model_ranking_20261017/audit.json).', '',
              'Reproduce with `python3 run.py rank_models`. This consolidates saved metrics and recomputes their aggregate scores; it performs no new fitting, model prediction or recipe tuning. '
              'All results remain developmental. Original evaluation and deployment artifacts retain their provenance.']
    (PROJECT/'MODEL_RANKING.md').write_text('\n'.join(lines)+'\n')
    policy=champion['best_pre_test_policy']
    best=['# Current best research model','',
          f"Promoted at the user's request on 17 October 2026: **`{champion['model_id']}`**.", '',
          f"Mean monthly MAE/interpolation MAE: **{champion['mean_monthly_ratio']:.6f}** "
          f"(**{champion['improvement_percent']:.2f}% reduction**). Pooled MAE: **{champion['pooled_mae']:.6f} EUR/MWh**.", '',
          'The quantile ensemble learns zero-mean within-hour corrections around mean-corrected linear interpolation. '
          'It uses known four-zone hourly curves and strictly prior quarter history, fixed physical feature scales and fitting-only additive knots. '
          'Models fit from scratch as training expands from three to ten months. '
          'Preceding-month days 1–15 select models, optional two-seed averages and global convex blend weights; those stay fixed throughout the following test month. '
          'Days 16–end separately calibrate all eleven quantiles and 50/80/90/95% intervals.', '',
          'April–September uses equal MLP/temporal-CNN/analog-2D-CNN weights. '
          'February uses 25% rich linear and 75% MLP; March averages all six experts. '
          'The ratio objective weights fitting months by their own interpolation errors and counts; no test denominator is an input.', '',
          f"Calibrated 90% empirical coverage: **{100*champion['coverage90']:.2f}%**; mean width **{champion['width90']:.4f} EUR/MWh**.", '',
          'The best observed recipe was identified after reporting the developmental test aggregates. '
          f"The strongest separate prior-data selection policy is `{policy['model_id']}` with ratio **{policy['mean_monthly_ratio']:.6f}**. "
          'These are distinct claims; cross-campaign rankings remain developmental.', '',
          'Canonical registry: [best_model.json](best_model.json). Full ranking: [MODEL_RANKING.md](MODEL_RANKING.md). '
          'Exact per-fold configurations, weights and checkpoint hashes: [champion_per_fold.json](results/model_ranking_20261017/champion_per_fold.json). '
          'Audit: [campaign audit](results/minimize_expanding_20261017/audit.json). '
          'Artifacts: [verified recovery archive](results/minimize_expanding_20261017_completed.tar.gz), '
          '[checksum receipt](results/minimize_expanding_20261017/backup_verification.json).', '',
          'This registry promotes the research recipe. The existing 363-day inference bundle still belongs to the earlier frozen pipeline. '
          'The new recipe has no all-history deployment refit; the registry does not label an evaluation checkpoint as such a refit.', '',
          'Every future comparison and promotion must follow [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md).']
    (PROJECT/'BEST_MODEL.md').write_text('\n'.join(best)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ranked,all_rows,monthly,receipts=collect_primary()
    historical,catalog,inventory,diagnostics=historical_and_catalog()
    champion=promote(ranked)
    write_csv(OUT/'primary_ranking.csv',ranked)
    write_csv(OUT/'all_reported_models.csv',all_rows)
    write_csv(OUT/'monthly_scores.csv',monthly)
    write_csv(OUT/'historical_ranking.csv',historical)
    write_csv(OUT/'candidate_catalog.csv',catalog)
    write_csv(OUT/'campaign_inventory.csv',inventory)
    write_csv(OUT/'oracle_diagnostics.csv',diagnostics)
    write_json(OUT/'ranking.json',dict(primary=ranked,all_reported=all_rows,historical=historical,champion=champion))
    docs(ranked,all_rows,historical,catalog,inventory,diagnostics,champion)
    audit=dict(passed=True,source_campaigns=len(SOURCES),reported_primary_rows=len(all_rows),ranked_primary_rows=len(ranked),
               alias_rows=sum(bool(r['alias_of']) for r in all_rows),monthly_rows=len(monthly),
               historical_rows=len(historical),candidate_records=len(catalog),campaigns_in_inventory=len(inventory),
               oracle_rows=len(diagnostics),expected_test_months=MONTHS,expected_test_days=DAYS,targets_per_recipe=92544,
               champion=champion['model_id'],best_pre_test_policy=champion['best_pre_test_policy'],source_receipts=receipts,
               exact_fold_recipes_sha256=champion['exact_fold_recipes_sha256'],
               checks=['passed owning audits','eight exact months and day counts','same corrected interpolation denominators',
                       'recomputed mean monthly ratios and pooled MAE','declared control alias score identity',
                       'historical/pretrained/oracle/smoke scope separation','source hashes and verified champion archive',
                       'eight exact champion fold recipes and 51 checkpoint leaf hashes'],
               limitation='Metadata consolidation over prior audits, not a fresh full-model replay; budgets and inner objectives differ.')
    write_json(OUT/'audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['source_receipts','checks']},indent=2))


if __name__=='__main__':
    main()
