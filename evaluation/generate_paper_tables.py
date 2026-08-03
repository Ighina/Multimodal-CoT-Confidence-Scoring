#!/usr/bin/env python3
"""
Generate the LaTeX table bodies for the paper's main tables (Tables 1, 2 and
9) and the LCO encoder appendix (Appendix H, Tables 12-13); with --setting mw
it generates the geometry-weighted voting tables of Appendices J-K (Tables
15-16 and 18-19). Values are 100-run means from statistics/no_majority
(self-consistency row from statistics/majority_weighted); significance stars
come from the item-level paired bootstrap (bootstrap-results/<ds>-nomaj/
pairwise_p.csv): best per column vs runner-up, * p<0.05, ** p<0.01,
*** p<0.001. Output: generated_tables/*.tex fragments (tabular rows only,
to be pasted into the paper source).

--holm: additionally apply a Holm-Bonferroni step-down correction to the
significance markers of the three MAIN tables (tab:sf-gemini, tab:sf-minicpm,
tab:sf-aurac), one family per table per marker type: the 10 best-vs-runner-up
column tests of a table form one asterisk family, and the 10 within-
sampling-free column tests form one dagger family (for tab:sf-aurac the
families pool the 5 Gemini + 5 MiniCPM AURAC columns). The LCO encoder
appendix tables (tab:lco-auroc-ece, tab:lco-aurac) use the same marker
convention and get the same treatment: 10-test families per marker type for
the AUROC+ECE table, 5-test families for the AURAC table. Adjusted p-values
are re-thresholded with the unchanged star/dagger cutoffs. Corrected bodies
go to bootstrap-results/holm_tables/ and a per-column marker-change summary
is printed as CSV on stdout; the default (no --holm) output is untouched.

--setting mw: generate the majority-weighted (geometry-weighted voting)
tables instead: values are the 100-run means from statistics/
majority_weighted (method keys weighted_majority_*), markers come from the
item-level paired bootstrap of the majority_weighted setting
(bootstrap-results/{gemini,minicpm,lco-gemini}/pairwise_p.csv). Asterisks:
best per column vs runner-up, as in the main tables. Daggers: best of OUR
methods (the five weighted scores + aggregated_optimal) per column vs the
best NON-ours method (SC / UMPIRE / verbalised), same thresholds; the dagger
is placed on the winner of that comparison. Output fragments go to
bootstrap-results/mw_tables/ (with --holm: *_holm.tex plus the summary CSV).
"""

import argparse
import csv
import os

# Working root holding the released data artifacts and stored run outputs.
# Override with the OMNIEVAL_ROOT environment variable; defaults to the
# current working directory (see evaluation/README.md).
ROOT = os.environ.get("OMNIEVAL_ROOT", os.getcwd())
SUBSETS = ["Overall", "UNOBench-Audio", "UNOBench-MC", "UNOBench-MO",
           "UNOBench-Visual"]

DS = {
    "gemini": dict(stats="gemini-majority-multirun-logistic-reg/statistics",
                   boot="bootstrap-results/gemini-nomaj",
                   boot_mw="bootstrap-results/gemini",
                   gavg="cross_modal_coherence",
                   gmax="cross_modal_grounding_max"),
    "minicpm": dict(stats="minicpm-majority-multirun-logistic-reg/statistics",
                    boot="bootstrap-results/minicpm-nomaj",
                    boot_mw="bootstrap-results/minicpm",
                    gavg="cross_modal_overall",
                    gmax="cross_modal_max_step_coherence"),
    "lco": dict(stats="lco-gemini-majority-multirun-logistic-reg/statistics",
                boot="bootstrap-results/lco-gemini-nomaj",
                boot_mw="bootstrap-results/lco-gemini",
                gavg="cross_modal_overall",
                gmax="cross_modal_max_step_coherence"),
}

MW = "weighted_majority_"  # method-key prefix in the majority_weighted CSVs


def rows_for(ds, setting="nomaj"):
    c = DS[ds]
    if setting == "mw":
        return [
            ("Self-Consistency", "majority_vote_selfconsistency",
             "majority_weighted"),
            ("UMPIRE", MW + "umpire_normal", "majority_weighted"),
            ("MID1", None, None),
            ("Level-Based Conf.", MW + "confidence_score_level_based",
             "majority_weighted"),
            ("Self-Probing Conf.", MW + "confidence_score_selfprobing",
             "majority_weighted"),
            ("Two-Step CoT Conf.", MW + "confidence_score_verb_2s_cot",
             "majority_weighted"),
            ("MID2", None, None),
            (r"$S_{\text{smooth}}$", MW + "internal_smoothness",
             "majority_weighted"),
            (r"$S_{\text{dens}}$", MW + "internal_semantic_density",
             "majority_weighted"),
            (r"$S_{\text{goal}}$", MW + "internal_goal_directedness",
             "majority_weighted"),
            (r"$G_{\text{avg}}$", MW + c["gavg"], "majority_weighted"),
            (r"$G_{\max}$", MW + c["gmax"], "majority_weighted"),
            ("MID3", None, None),
            (r"$\mathcal{C}_{\text{chain}}$", MW + "aggregated_optimal",
             "majority_weighted"),
        ]
    return [
        ("Self-Consistency", "majority_vote_selfconsistency", "majority_weighted"),
        ("UMPIRE", "umpire_normal", "no_majority"),
        ("MID1", None, None),
        ("Level-Based Conf.", "confidence_score_level_based", "no_majority"),
        ("Self-Probing Conf.", "confidence_score_selfprobing", "no_majority"),
        ("Two-Step CoT Conf.", "confidence_score_verb_2s_cot", "no_majority"),
        ("MID2", None, None),
        (r"$S_{\text{smooth}}$", "internal_smoothness", "no_majority"),
        (r"$S_{\text{dens}}$", "internal_semantic_density", "no_majority"),
        (r"$S_{\text{goal}}$", "internal_goal_directedness", "no_majority"),
        (r"$G_{\text{avg}}$", c["gavg"], "no_majority"),
        (r"$G_{\max}$", c["gmax"], "no_majority"),
        ("MID3", None, None),
        (r"$\mathcal{C}_{\text{chain}}$", "aggregated_optimal", "no_majority"),
    ]


def load_means(ds):
    """means[subset][(setting, method)][metric] = float"""
    out = {}
    for subset in SUBSETS:
        out[subset] = {}
        for setting in ["no_majority", "majority_weighted"]:
            path = os.path.join(ROOT, DS[ds]["stats"], setting, f"{subset}.csv")
            with open(path) as f:
                for r in csv.DictReader(f):
                    vals = {}
                    for m in ["AUROC", "ECE", "AURAC"]:
                        cell = r[m].split("±")[0].strip()
                        vals[m] = float(cell)
                    out[subset][(setting, r["Method"])] = vals
    return out


def load_pairwise(ds, setting="nomaj"):
    """p[(subset, metric)][(a, b)] = (delta_mean, p)"""
    out = {}
    boot_key = "boot_mw" if setting == "mw" else "boot"
    path = os.path.join(ROOT, DS[ds][boot_key], "pairwise_p.csv")
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["Subset"], r["Metric"])
            out.setdefault(key, {})[(r["MethodA"], r["MethodB"])] = (
                float(r["Delta_mean"]), float(r["p_boot"]))
    return out


def stars(p):
    if p < 0.001:
        return r"\textsuperscript{***}"
    if p < 0.01:
        return r"\textsuperscript{**}"
    if p < 0.05:
        return r"\textsuperscript{*}"
    return ""


def daggers(p):
    if p < 0.0001:
        return r"\textsuperscript{\dag\dag\dag}"
    if p < 0.001:
        return r"\textsuperscript{\dag\dag}"
    if p < 0.01:
        return r"\textsuperscript{\dag}"
    return ""


VERBALISED = {"confidence_score_level_based", "confidence_score_selfprobing",
              "confidence_score_verb_2s_cot"}


def ours_set(ds, setting="nomaj"):
    c = DS[ds]
    base = {"internal_smoothness", "internal_semantic_density",
            "internal_goal_directedness", c["gavg"], c["gmax"],
            "aggregated_optimal"}
    if setting == "mw":
        return {MW + m for m in base}
    return base


def nonours_set_mw():
    """The non-ours methods of the majority-weighted tables: SC, UMPIRE and
    the three verbalised baselines (all under weighted voting)."""
    return ({"majority_vote_selfconsistency", MW + "umpire_normal"}
            | {MW + m for m in VERBALISED})


def boot_name(method, setting):
    # bootstrap ran on the no_majority setting; SC kept its own name there
    return method


def col_tests(ds, means, pw, subset_metric_list, setting="nomaj"):
    """Compute the per-column tests. Returns (col_best, col_sf_best), each a
    {(subset, metric): (method, raw_p)} map."""
    rlist = rows_for(ds, setting)
    # best / runner-up per column
    col_best = {}
    for subset, metric in subset_metric_list:
        vals = []
        for label, method, row_setting in rlist:
            if method is None:
                continue
            v = means[subset][(row_setting, method)][metric]
            vals.append((v, method))
        rev = metric != "ECE"
        vals.sort(key=lambda t: t[0], reverse=rev)
        best_m, run_m = vals[0][1], vals[1][1]
        pmap = pw[(subset, metric)]
        pr = pmap.get((best_m, run_m)) or pmap.get((run_m, best_m))
        col_best[(subset, metric)] = (best_m, pr[1] if pr else 1.0)

    # dagger: two method groups compared through their per-column bests; the
    # dagger goes on the winner. nomaj: our scores vs the verbalised
    # baselines (both sampling-free, looked up under no_majority). mw: our
    # weighted scores vs the non-ours methods (SC/UMPIRE/verbalised, all
    # under majority_weighted).
    if setting == "mw":
        ours = ours_set(ds, "mw")
        other = nonours_set_mw()
        lookup = "majority_weighted"
    else:
        ours = ours_set(ds)
        other = VERBALISED
        lookup = "no_majority"
    universe = ours | other
    col_sf_best = {}
    for subset, metric in subset_metric_list:
        rev = metric != "ECE"
        ranked = sorted(
            ((means[subset][(lookup, m)][metric], m) for m in universe),
            key=lambda t: t[0], reverse=rev)
        best_m = ranked[0][1]
        other_family = other if best_m in ours else ours
        comp_m = sorted(
            ((means[subset][(lookup, m)][metric], m)
             for m in other_family),
            key=lambda t: t[0], reverse=rev)[0][1]
        pmap = pw[(subset, metric)]
        pr = pmap.get((best_m, comp_m)) or pmap.get((comp_m, best_m))
        col_sf_best[(subset, metric)] = (best_m, pr[1] if pr else 1.0)
    return col_best, col_sf_best


def holm_adjust(pvals):
    """Holm step-down adjusted p-values. pvals: {key: p} for one family.
    Returns {key: p_adj} with p_adj monotone and capped at 1."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        out[k] = min(1.0, running)
    return out


def make_cells(ds, means, pw, subset_metric_list,
               col_best=None, col_sf_best=None, table_setting="nomaj"):
    """Format the table body; the test maps may be passed in precomputed
    (e.g. with Holm-adjusted p-values), otherwise raw ones are computed."""
    rlist = rows_for(ds, table_setting)
    if col_best is None or col_sf_best is None:
        col_best, col_sf_best = col_tests(ds, means, pw, subset_metric_list,
                                          table_setting)
    dagger_row_setting = ("majority_weighted" if table_setting == "mw"
                          else "no_majority")

    lines = []
    for label, method, setting in rlist:
        if method is None:
            lines.append(r"\midrule")
            continue
        cells = [label]
        for subset, metric in subset_metric_list:
            v = means[subset][(setting, method)][metric]
            s = f"{v:.4f}"
            best_m, p = col_best[(subset, metric)]
            if method == best_m:
                s = r"\textbf{" + s + "}" + stars(p)
            sf_best_m, sf_p = col_sf_best[(subset, metric)]
            if setting == dagger_row_setting and method == sf_best_m:
                s += daggers(sf_p)
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


SHORT_SUBSET = {"Overall": "Overall", "UNOBench-Audio": "Audio",
                "UNOBench-MC": "MC", "UNOBench-MO": "MO",
                "UNOBench-Visual": "Visual"}

SHORT_METHOD = {
    "majority_vote_selfconsistency": "Self-Consistency",
    "umpire_normal": "UMPIRE",
    "confidence_score_level_based": "Level-Based Conf.",
    "confidence_score_selfprobing": "Self-Probing Conf.",
    "confidence_score_verb_2s_cot": "Two-Step CoT Conf.",
    "internal_smoothness": "S_smooth",
    "internal_semantic_density": "S_dens",
    "internal_goal_directedness": "S_goal",
    "cross_modal_coherence": "G_avg",
    "cross_modal_overall": "G_avg",
    "cross_modal_grounding_max": "G_max",
    "cross_modal_max_step_coherence": "G_max",
    "aggregated_optimal": "C_chain",
}


def short_method(m):
    """Readable method name for the summary; strips the mw prefix."""
    if m.startswith(MW):
        m = m[len(MW):]
    return SHORT_METHOD.get(m, m)


def star_marker(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else \
        "*" if p < 0.05 else ""


def dagger_marker(p):
    return "†††" if p < 0.0001 else "††" if p < 0.001 else \
        "†" if p < 0.01 else ""


def join_aurac(body_a, body_b):
    """Row-wise join of two AURAC fragments into the combined-table body."""
    out = []
    for a, b in zip(body_a.split("\n"), body_b.split("\n")):
        if a == r"\midrule":
            out.append(a)
            continue
        ca = a[:-3].rstrip().split(" & ")   # strip trailing " \\"
        cb = b[:-3].rstrip().split(" & ")
        assert ca[0] == cb[0]
        out.append(" & ".join(ca + cb[1:]) + r" \\")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holm", action="store_true",
                    help="apply Holm-Bonferroni per marker family per main "
                         "table; write corrected bodies to "
                         "bootstrap-results/holm_tables/ (or mw_tables/ "
                         "with --setting mw)")
    ap.add_argument("--setting", choices=["nomaj", "mw"], default="nomaj",
                    help="which evaluation setting to tabulate: 'nomaj' "
                         "(sampling-free, current main tables; default) or "
                         "'mw' (geometry-weighted majority voting)")
    args = ap.parse_args()
    st = args.setting

    ae_cols = [(s, m) for s in SUBSETS for m in ["AUROC", "ECE"]]
    aurac_cols = [(s, "AURAC") for s in SUBSETS]

    if st == "mw" and not args.holm:
        # raw-marker majority-weighted fragments
        outdir = os.path.join(ROOT, "bootstrap-results", "mw_tables")
        os.makedirs(outdir, exist_ok=True)
        bodies = {}
        for ds in ["gemini", "minicpm", "lco"]:
            means = load_means(ds)
            pw = load_pairwise(ds, "mw")
            body = make_cells(ds, means, pw, ae_cols, table_setting="mw")
            fname = f"mw_{ds}_auroc_ece.tex"
            with open(os.path.join(outdir, fname), "w") as f:
                f.write(body + "\n")
            print("wrote", fname)
            bodies[ds] = make_cells(ds, means, pw, aurac_cols,
                                    table_setting="mw")
            fname = f"mw_{ds}_aurac.tex"
            with open(os.path.join(outdir, fname), "w") as f:
                f.write(bodies[ds] + "\n")
            print("wrote", fname)
        with open(os.path.join(outdir, "mw_aurac_combined.tex"), "w") as f:
            f.write(join_aurac(bodies["gemini"], bodies["minicpm"]) + "\n")
        print("wrote mw_aurac_combined.tex")
        return

    if not args.holm:
        outdir = os.path.join(ROOT, "generated_tables")
        os.makedirs(outdir, exist_ok=True)

        # main AUROC+ECE tables (per generator): Overall..Visual x (AUROC, ECE)
        for ds, fname in [("gemini", "main_gemini_auroc_ece.tex"),
                          ("minicpm", "main_minicpm_auroc_ece.tex"),
                          ("lco", "lco_auroc_ece.tex")]:
            means = load_means(ds)
            pw = load_pairwise(ds)
            body = make_cells(ds, means, pw, ae_cols)
            with open(os.path.join(outdir, fname), "w") as f:
                f.write(body + "\n")
            print("wrote", fname)

        # AURAC tables: gemini+minicpm side by side; lco alone
        for ds, fname in [("gemini", "main_gemini_aurac.tex"),
                          ("minicpm", "main_minicpm_aurac.tex"),
                          ("lco", "lco_aurac.tex")]:
            means = load_means(ds)
            pw = load_pairwise(ds)
            body = make_cells(ds, means, pw, aurac_cols)
            with open(os.path.join(outdir, fname), "w") as f:
                f.write(body + "\n")
            print("wrote", fname)
        return

    # ---- Holm mode ----
    # nomaj: main tables (tab:sf-gemini/-minicpm/-aurac) plus the LCO encoder
    # appendix tables (tab:lco-auroc-ece/-aurac). mw: the majority-weighted
    # counterparts, written to bootstrap-results/mw_tables/.
    if st == "mw":
        outdir = os.path.join(ROOT, "bootstrap-results", "mw_tables")
        ae_specs = [
            ("gemini", "mw-gemini-auroc-ece", "mw_gemini_auroc_ece_holm.tex"),
            ("minicpm", "mw-minicpm-auroc-ece",
             "mw_minicpm_auroc_ece_holm.tex"),
            ("lco", "mw-lco-auroc-ece", "mw_lco_auroc_ece_holm.tex")]
        aurac_table = "mw-aurac"
        aurac_half_fmt = "mw_{}_aurac_holm.tex"
        aurac_comb_fname = "mw_aurac_combined_holm.tex"
        lco_aurac_table, lco_aurac_fname = "mw-lco-aurac", "mw_lco_aurac_holm.tex"
    else:
        outdir = os.path.join(ROOT, "bootstrap-results", "holm_tables")
        ae_specs = [
            ("gemini", "tab:sf-gemini", "main_gemini_auroc_ece_holm.tex"),
            ("minicpm", "tab:sf-minicpm", "main_minicpm_auroc_ece_holm.tex"),
            ("lco", "tab:lco-auroc-ece", "lco_auroc_ece_holm.tex")]
        aurac_table = "tab:sf-aurac"
        aurac_half_fmt = "main_{}_aurac_holm.tex"
        aurac_comb_fname = "main_aurac_combined_holm.tex"
        lco_aurac_table, lco_aurac_fname = "tab:lco-aurac", "lco_aurac_holm.tex"
    os.makedirs(outdir, exist_ok=True)

    means = {ds: load_means(ds) for ds in ["gemini", "minicpm", "lco"]}
    pw = {ds: load_pairwise(ds, st) for ds in ["gemini", "minicpm", "lco"]}

    summary = []  # (table, column, family, method, raw_p, adj_p, old, new)

    def adjust(tests, table, marker_fn, family):
        """tests: {out_key: (ds, col_key, method, raw_p)} = one Holm family.
        Returns {out_key: (method, adj_p)} and appends to the summary."""
        adj = holm_adjust({k: v[3] for k, v in tests.items()})
        res = {}
        for k, (ds, col, method, raw_p) in tests.items():
            res[k] = (method, adj[k])
            summary.append((table, k, family, short_method(method),
                            raw_p, adj[k],
                            marker_fn(raw_p), marker_fn(adj[k])))
        return res

    def split_by_ds(adjusted):
        out = {}
        for (ds, col), v in adjusted.items():
            out.setdefault(ds, {})[col] = v
        return out

    # AUROC+ECE tables (10 columns): one family per table per marker type
    for ds, table, fname in ae_specs:
        cb, csf = col_tests(ds, means[ds], pw[ds], ae_cols, st)
        stars_t = {(SHORT_SUBSET[s] + " " + m): (ds, (s, m), cb[(s, m)][0],
                                                 cb[(s, m)][1])
                   for s, m in ae_cols}
        dag_t = {(SHORT_SUBSET[s] + " " + m): (ds, (s, m), csf[(s, m)][0],
                                               csf[(s, m)][1])
                 for s, m in ae_cols}
        cb_adj = {stars_t[k][1]: v
                  for k, v in adjust(stars_t, table, star_marker,
                                     "asterisk").items()}
        csf_adj = {dag_t[k][1]: v
                   for k, v in adjust(dag_t, table, dagger_marker,
                                      "dagger").items()}
        body = make_cells(ds, means[ds], pw[ds], ae_cols,
                          col_best=cb_adj, col_sf_best=csf_adj,
                          table_setting=st)
        with open(os.path.join(outdir, fname), "w") as f:
            f.write(body + "\n")
        print("wrote", fname)

    # combined AURAC table: the 5 Gemini + 5 MiniCPM columns pool into ONE
    # family per marker type
    stars_t, dag_t = {}, {}
    for ds, tag in [("gemini", "Gemini"), ("minicpm", "MiniCPM")]:
        cb, csf = col_tests(ds, means[ds], pw[ds], aurac_cols, st)
        for s, m in aurac_cols:
            key = tag + " " + SHORT_SUBSET[s]
            stars_t[key] = (ds, (s, m), cb[(s, m)][0], cb[(s, m)][1])
            dag_t[key] = (ds, (s, m), csf[(s, m)][0], csf[(s, m)][1])
    cb_adj = split_by_ds({(stars_t[k][0], stars_t[k][1]): v for k, v in
                          adjust(stars_t, aurac_table, star_marker,
                                 "asterisk").items()})
    csf_adj = split_by_ds({(dag_t[k][0], dag_t[k][1]): v for k, v in
                           adjust(dag_t, aurac_table, dagger_marker,
                                  "dagger").items()})
    bodies = {}
    for ds in ["gemini", "minicpm"]:
        bodies[ds] = make_cells(ds, means[ds], pw[ds], aurac_cols,
                                col_best=cb_adj[ds], col_sf_best=csf_adj[ds],
                                table_setting=st)
        fname = aurac_half_fmt.format(ds)
        with open(os.path.join(outdir, fname), "w") as f:
            f.write(bodies[ds] + "\n")
        print("wrote", fname)
    fname = aurac_comb_fname
    with open(os.path.join(outdir, fname), "w") as f:
        f.write(join_aurac(bodies["gemini"], bodies["minicpm"]) + "\n")
    print("wrote", fname)

    # LCO AURAC: standalone 5-column table -> one 5-test family per
    # marker type
    ds, table = "lco", lco_aurac_table
    cb, csf = col_tests(ds, means[ds], pw[ds], aurac_cols, st)
    stars_t = {SHORT_SUBSET[s]: (ds, (s, m), cb[(s, m)][0], cb[(s, m)][1])
               for s, m in aurac_cols}
    dag_t = {SHORT_SUBSET[s]: (ds, (s, m), csf[(s, m)][0], csf[(s, m)][1])
             for s, m in aurac_cols}
    cb_adj = {stars_t[k][1]: v
              for k, v in adjust(stars_t, table, star_marker,
                                 "asterisk").items()}
    csf_adj = {dag_t[k][1]: v
               for k, v in adjust(dag_t, table, dagger_marker,
                                  "dagger").items()}
    body = make_cells(ds, means[ds], pw[ds], aurac_cols,
                      col_best=cb_adj, col_sf_best=csf_adj,
                      table_setting=st)
    fname = lco_aurac_fname
    with open(os.path.join(outdir, fname), "w") as f:
        f.write(body + "\n")
    print("wrote", fname)

    # marker-change summary (CSV on stdout)
    print()
    print("table,column,family,method,raw_p,holm_p,old_marker,new_marker,"
          "changed")
    for t, col, fam, meth, rp, ap_, old, new in summary:
        print(f"{t},{col},{fam},{meth},{rp:.4f},{ap_:.4f},"
              f"{old or '(none)'},{new or '(none)'},"
              f"{'YES' if old != new else 'no'}")


if __name__ == "__main__":
    main()
