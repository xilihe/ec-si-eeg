#!/usr/bin/env python3
"""Build manuscript inserts only from independently validated real summaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', required=True, type=Path)
    p.add_argument('--submission', required=True, type=Path)
    a = p.parse_args()
    assert json.loads((a.analysis / 'independent_validation.json').read_text())['status'] == 'PASS'
    ledger = []
    def read(file):
        with (a.analysis / file).open() as f:
            return list(csv.DictReader(f))
    def number(row, key='value'):
        value = f'{100 * float(row[key]):.3f}'
        ledger.append(dict(value=value, field=key, task=row['task'], condition=row.get('condition', row.get('left_condition')),
            measure=row.get('measure', row.get('left_measure')), source_value=row[key]))
        return value
    def ci(row, key='value'):
        return number(row, key) + ' (' + number(row, key + '_ci_low') + ', ' + number(row, key + '_ci_high') + ')'
    summary, contrasts, behavior, gaps = {}, {}, {}, {}
    for task in ('faced', 'seed_binary'):
        summary[task] = {(r['condition'], r['measure']): r for r in read(f'projection_{task}_summary.csv')}
        contrasts[task] = {r['left_measure']: r for r in read(f'projection_{task}_contrasts.csv')}
        behavior[task] = {(r['condition'], r['measure']): r for r in read(f'behavior_{task}_summary.csv')}
        gaps[task] = {r['left_condition']: r for r in read(f'behavior_{task}_contrasts.csv')}
    text = r'''\subsection{Post-hoc Projection and Stimulus-Role Results}
The fixed global linear projection reduced the conditional identity Probe in frozen CE features on both tasks (Table~\ref{tab:projection}). On FACED, the matched Probe difference was ''' + ci(contrasts['faced']['probe']) + r''' percentage points, while the matched ridge-classification difference was ''' + ci(contrasts['faced']['accuracy']) + r'''. SEED's corresponding differences were ''' + ci(contrasts['seed_binary']['probe']) + ' and ' + ci(contrasts['seed_binary']['accuracy']) + r''' percentage points. Parentheses give nonsimultaneous 95\% participant-bootstrap intervals. These are before/after comparisons with identical readers, not direct comparisons with the native neural emotion head. FACED's projected score remains close to the nine-class uniform reference; the identity reduction does not validate reliable non-neutral classification.

\begin{table}[t]
\caption{Training-only one-pass global linear projection on frozen CE features. Values are percentages; $n=98$ FACED and $n=12$ SEED participants. The same fixed ridge emotion reader is fitted before and after projection. Identity Probes are fitted independently in each feature space. SEED uses equal clips within category here, unlike the original run-equal Probe summary. Full intervals and class recalls appear in Tables S33 and S36.}
\label{tab:projection}
\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrr}
\toprule
Task & CE feature space & Class acc. & Identity Probe \\
\midrule
'''
    for task, name in [('faced', 'FACED'), ('seed_binary', 'SEED')]:
        for condition, label in [('unprojected_ridge', 'Original'), ('projected_ridge', 'Projected')]:
            text += name + ' & ' + label + ' & ' + number(summary[task][(condition, 'accuracy')]) + ' & ' + number(summary[task][(condition, 'probe')]) + r' \\' + '\n'
    text += r'''\bottomrule
\end{tabular}
\end{table}

On the same unseen participants, standard FACED CE classified training-visible clips at ''' + number(behavior['faced'][('ce', 'seen')]) + r'''\% and held-out clips at ''' + number(behavior['faced'][('ce', 'unseen')]) + r'''\%. The seen-minus-unseen gap was ''' + ci(gaps['faced']['ce']) + r''' percentage points, compared with ''' + ci(gaps['faced']['ecsi']) + ' for EC-SI and ' + ci(gaps['faced']['global']) + r''' for Global. The complete 11-configuration results are in Table S34. A smaller gap at high strength accompanies concentrated category decisions and does not independently demonstrate reduced shortcut use.

On SEED, the standard EC-SI gap was ''' + ci(gaps['seed_binary']['ecsi']) + r''' percentage points (Table S35). Its interval includes either sign. Seen evaluations cover ten clips and held-out evaluations eight in the retained SEED partitions; even on FACED, the two roles involve different training fits. These gaps describe behavior on the fixed collections and do not identify a causal identity-dependence effect.
'''
    (a.submission / 'projection_behavior_results.tex').write_text(text)
    si = r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=19mm]{geometry}
\usepackage{booktabs,array,amsmath}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\begin{document}
\section*{Post-hoc projection and stimulus-role analyses}
\textbf{Supplementary Table S33. Matched linear-projection control.}
All results use real frozen CE representations: 108 FACED and 48 SEED fits. Projection fitting uses training features and global stimulus identities without emotion labels. Original and projected spaces receive the same $\alpha=1$ emotion ridge reader and separately training-fitted conditional centroids. Values are percentages; parentheses give nonsimultaneous 95\% whole-participant bootstrap intervals ($n=98$ FACED; $n=12$ SEED). These exploratory analyses introduce no new confirmatory tests.

\begin{center}\small
\begin{tabular}{llrr}
\toprule Task & Space & Class accuracy (95\% CI) & Identity Probe (95\% CI)\\
\midrule
'''
    for task, name in [('faced', 'FACED'), ('seed_binary', 'SEED')]:
        for condition, label in [('unprojected_ridge', 'Original'), ('projected_ridge', 'Projected')]:
            si += name + ' & ' + label + ' & ' + ci(summary[task][(condition, 'accuracy')]) + ' & ' + ci(summary[task][(condition, 'probe')]) + r' \\' + '\n'
    si += r'''\bottomrule\end{tabular}\end{center}
\begin{center}\small\begin{tabular}{lrr}
\toprule Task & Accuracy difference (95\% CI) & Probe difference (95\% CI)\\\midrule
'''
    for task, name in [('faced', 'FACED'), ('seed_binary', 'SEED')]:
        si += name + ' & ' + ci(contrasts[task]['accuracy']) + ' & ' + ci(contrasts[task]['probe']) + r' \\' + '\n'
    si += r'''\bottomrule\end{tabular}\end{center}
Differences are projected minus original, in percentage points; negative Probe differences mean lower decoding. SVD removes 18 dimensions on every FACED fit and seven on every SEED fit. Centering, scaling, projection, ridge and centroids use no evaluation rows. The original neural emotion head is not the matched baseline.

The new SEED Probe is clip-equal within category (unprojected mean 31.582\%); the original complete-curve Probe is run-equal (31.858\%). Unequal stimulus exposure across folds 2--5 makes these summaries different estimands. The before/after projection contrast always uses the same clip-equal rule. No source outcome was excluded or reweighted after observing its effect.
'''
    order = ['ce', 'ecsi', 'global', 'rho0.1_ecsi', 'rho0.1_global', 'rho0.3_ecsi', 'rho0.3_global', 'rho1_ecsi', 'rho1_global', 'rho3_ecsi', 'rho3_global']
    def label(c):
        names = {'ce': 'CE', 'ecsi': 'EC-SI', 'global': 'Global'}
        if '_' not in c:
            return 'Standard ' + names[c]
        strength, method = c.split('_')
        return 'Norm ' + strength[3:] + ' ' + names[method]
    for task, name, ntable in [('faced', 'FACED', 'S34'), ('seed_binary', 'SEED', 'S35')]:
        si += '\n\\clearpage\n\\textbf{Supplementary Table ' + ntable + '. ' + name + r''' stimulus-role behavior.}
The saved native neural head classifies seen (training-visible) and unseen (held-out) clips viewed by the same unseen participants. Accuracy averages repeats within clip/session, sessions within clip, clips within category and categories within participant. Values are percentages; the last column is seen minus unseen in percentage points. Parentheses are nonsimultaneous participant-bootstrap 95\% intervals. All 11 complete-curve configurations are retained.

\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrr}
\toprule Policy/objective & Seen accuracy (95\% CI) & Unseen accuracy (95\% CI) & Gap (95\% CI)\\\midrule
'''
        for c in order:
            si += label(c) + ' & ' + ci(behavior[task][(c, 'seen')]) + ' & ' + ci(behavior[task][(c, 'unseen')]) + ' & ' + ci(gaps[task][c]) + r' \\' + '\n'
        si += r'''\bottomrule\end{tabular}\end{center}
'''
        if task == 'faced':
            si += r'''There are 98 participants and 108 fits per configuration. Each role covers all 28 fixed clips across the retained partitions, but the classification rule is learned from different training-clip assignments. A smaller gap can accompany concentration in neutral decisions and is not a causal measure of shortcut dependence. The full class recalls are retained in the source CSVs.
'''
        else:
            si += r'''There are 12 participants and 48 fits per configuration, with three repeated sessions. Across folds 2--5, seen evaluations contain ten non-neutral clips whereas unseen evaluations contain eight (four per class). The development-fold pair is not held out in these fits. Stimulus composition, content/difficulty, the fitted classifier and cohort reuse constrain interpretation; intervals spanning zero allow either direction. No new stimulus source or independent cohort is introduced.
'''
    si += r'''
\clearpage
\textbf{Supplementary Table S36. Matched projection class recalls.}
Class recalls from the ridge readers in Table S33. Original/projected columns use the same training-only normalization and fixed reader penalty. Values are percentages with nonsimultaneous participant-bootstrap 95\% intervals. They describe assigned-label decisions, not self-reported emotions.
\begin{center}\small\begin{tabular}{llrr}
\toprule Task & Category & Original recall (95\% CI) & Projected recall (95\% CI)\\\midrule
'''
    for task, name, classes in [('faced', 'FACED', 9), ('seed_binary', 'SEED', 2)]:
        for c in range(classes):
            category = ('Neutral (4)' if c == 4 else str(c)) if task == 'faced' else ('Negative' if c == 0 else 'Positive')
            si += name + ' & ' + category + ' & ' + ci(summary[task][('unprojected_ridge', 'accuracy')], f'recall_{c}') + ' & ' + ci(summary[task][('projected_ridge', 'accuracy')], f'recall_{c}') + r' \\' + '\n'
    si += r'''\bottomrule\end{tabular}\end{center}
\textbf{Provenance and uncertainty.} The full matrix passed per-fit role, input-hash and checkpoint-head checks. An independent audit reproduced 11,010 source-score cells and 507 descriptive summary intervals, and checked all 156 saved projection bases for orthonormality. Failed engineering checks are excluded from scientific sample counts and retained in the private run records. These new analyses do not change the original 20-per-task or joint 40-test families. Whole-participant bootstrapping conditions on the realized training fits and fixed collections; it does not propagate new-cohort retraining or independent-stimulus uncertainty.
\end{document}
'''
    (a.submission / 'supplementary_projection_behavior.tex').write_text(si)
    with (a.submission / 'projection_behavior_numbers.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ledger[0]))
        w.writeheader(); w.writerows(ledger)
    validation = dict(status='GENERATED_FROM_VERIFIED_REAL_DATA', ledger_cells=len(ledger),
        independent_validation_sha256=hashlib.sha256((a.analysis / 'independent_validation.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.submission / n for n in
            ['projection_behavior_results.tex', 'supplementary_projection_behavior.tex', 'projection_behavior_numbers.csv']]})
    (a.submission / 'projection_behavior_generation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
