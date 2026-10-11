#!/usr/bin/env python3
"""
Benchmark Evaluation Pipeline: AlphaFold 2 Multimer v3 on PDB 9CLP
Target: Full-Length Mature Ecarin (420 aa, Kenyan Echis pyramidum leakeyi, UniProt Q90495)
Antibody: Synthetic Neutralizing Human Fab H11 (VH: 120 aa, VL: 110 aa)
Cryo-EM Ground Truth: PDB 9CLP / EMD-45728 (3.43 A nominal resolution)

Metrics Evaluated:
- Domain-by-Domain Backbone C-alpha RMSD (Metalloproteinase, Disintegrin-like, CRD, Fv)
- CAPRI Quality Classification (High, Medium, Acceptable, Incorrect)
- DockQ Score (Basu & Wallner, 2016)
- Native Contact Recovery (Fnat) against 42 Cryo-EM contacts (< 5.0 A)
- Interface RMSD (iRMSD) and Ligand RMSD (lRMSD)
- AlphaFold confidence scores (pLDDT, ipTM, pTM)
"""

import os
import glob
import json
import urllib.request
import csv
import zipfile
import re
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AlphaFold 2 Multimer predictions on PDB 9CLP")
    parser.add_argument("--ref", default="9CLP.cif", help="Path to reference 9CLP structure (CIF or PDB)")
    parser.add_argument("--models_dir", default=".", help="Directory containing prediction files or zip archives")
    parser.add_argument("--out_csv", default="results.csv", help="Output path for benchmark metrics CSV")
    parser.add_argument("--overlay", default="overlay_full_complex_rank001.pdb", help="Output path for 3D overlay PDB")
    return parser.parse_args()


def download_reference_if_missing(ref_path):
    if not os.path.exists(ref_path):
        url = "https://files.rcsb.org/download/9CLP.cif"
        print(f"Downloading reference structure from {url}...")
        urllib.request.urlretrieve(url, ref_path)
        print(f"Saved reference to {ref_path}")


def parse_atoms(fpath):
    """Parse backbone and heavy atoms from PDB or mmCIF coordinate files."""
    atoms, is_cif = [], fpath.endswith(".cif")
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        in_loop, headers = False, []
        for line in f:
            if is_cif:
                line_str = line.strip()
                if line_str == "loop_":
                    in_loop, headers = False, []
                    continue
                if line_str.startswith("_atom_site."):
                    in_loop = True
                    headers.append(line_str)
                    continue
                if in_loop:
                    if line_str.startswith(("#", "_")) or not line_str:
                        in_loop = False
                        continue
                    p = line_str.split()
                    if len(p) >= len(headers):
                        d = dict(zip(headers, p[:len(headers)]))
                        try:
                            atoms.append({
                                "c": d.get("_atom_site.auth_asym_id", d.get("_atom_site.label_asym_id", "")),
                                "n": int(d.get("_atom_site.auth_seq_id", d.get("_atom_site.label_seq_id", "0"))),
                                "res": d.get("_atom_site.auth_comp_id", d.get("_atom_site.label_comp_id", "")),
                                "a": d.get("_atom_site.auth_atom_id", d.get("_atom_site.label_atom_id", "")),
                                "el": d.get("_atom_site.type_symbol", ""),
                                "xyz": np.array([float(d["_atom_site.Cartn_x"]), float(d["_atom_site.Cartn_y"]), float(d["_atom_site.Cartn_z"])]),
                                "bf": float(d.get("_atom_site.B_iso_or_equiv", 20.0))
                            })
                        except (ValueError, KeyError):
                            pass
            else:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        atoms.append({
                            "c": line[21].strip(),
                            "n": int(line[22:26].strip()),
                            "res": line[17:20].strip(),
                            "a": line[12:16].strip(),
                            "el": line[76:78].strip() if len(line) >= 78 else line[12:16].strip()[0],
                            "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                            "bf": float(line[60:66]) if len(line) >= 66 else 0.0
                        })
                    except (ValueError, IndexError):
                        pass
    return atoms


def kabsch(P, Q):
    """Compute optimal rotation R and translation t such that R @ P + t minimizes RMSD to Q."""
    cP, cQ = np.mean(P, axis=0), np.mean(Q, axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cQ - (R @ cP)
    return R, t


def rmsd(P, Q):
    """Compute Root Mean Square Deviation between coordinate arrays P and Q."""
    return np.sqrt(np.mean(np.sum((P - Q)**2, axis=-1)))


def extract_archives(models_dir):
    """Unpack result zip archives if prediction PDBs are compressed."""
    archives = glob.glob(os.path.join(models_dir, "**/*.result.zip"), recursive=True) + \
               glob.glob(os.path.join(models_dir, "*.zip"))
    for z in archives:
        try:
            with zipfile.ZipFile(z, "r") as zr:
                zr.extractall(models_dir)
        except zipfile.BadZipFile:
            pass


def find_prediction_pdbs(models_dir):
    """Locate unique ranked AlphaFold prediction PDB files."""
    pattern = os.path.join(models_dir, "**/*rank*.pdb")
    pdbs = sorted([f for f in glob.glob(pattern, recursive=True)
                   if "overlay" not in f and "ref" not in f and not os.path.basename(f).startswith("9CLP.")])
    if not pdbs:
        fallback = os.path.join(models_dir, "**/*.pdb")
        pdbs = sorted([f for f in glob.glob(fallback, recursive=True)
                       if "overlay" not in f and "ref" not in f and not os.path.basename(f).startswith("9CLP.")])

    rank_map = {}
    for f in pdbs:
        m = re.search(r"rank_\d{3}", os.path.basename(f))
        rk = m.group(0) if m else f
        if rk not in rank_map:
            rank_map[rk] = f
    return [rank_map[k] for k in sorted(rank_map.keys())]


def evaluate():
    args = parse_args()
    download_reference_if_missing(args.ref)
    extract_archives(args.models_dir)

    # 1. Parse Reference Structure (9CLP Ground Truth)
    cif = parse_atoms(args.ref)
    nat_ca_a = {a["n"]: a["xyz"] for a in cif if a["c"] == "A" and a["a"] == "CA" and 195 <= a["n"] <= 614}
    nat_hvy_a = {}
    for a in cif:
        if a["c"] == "A" and 195 <= a["n"] <= 614 and a["el"] != "H":
            nat_hvy_a.setdefault(a["n"], []).append(a["xyz"])

    nat_seq_ca_h = [a["xyz"] for a in cif if a["c"] == "H" and a["a"] == "CA"]
    nat_seq_ca_l = [a["xyz"] for a in cif if a["c"] == "L" and a["a"] == "CA"]

    def group_antibody_heavy(chain_id):
        groups, cur_res, cur_atoms = [], None, []
        for a in cif:
            if a["el"] == "H" or a["c"] != chain_id:
                continue
            rid = a["n"]
            if rid != cur_res:
                if cur_atoms:
                    groups.append(cur_atoms)
                cur_res, cur_atoms = rid, []
            cur_atoms.append(a["xyz"])
        if cur_atoms:
            groups.append(cur_atoms)
        return groups

    nat_h_hvy_seq = group_antibody_heavy("H")
    nat_l_hvy_seq = group_antibody_heavy("L")

    # Native contact mapping (< 5.0 A for contacts, < 10.0 A for interface residue set)
    contacts, int_rec_res, int_lig_res = [], set(), set()
    for rn, r_atoms in nat_hvy_a.items():
        r_arr = np.array(r_atoms)
        for h_idx, h_atoms in enumerate(nat_h_hvy_seq):
            d = np.min(np.linalg.norm(r_arr[:, None, :] - np.array(h_atoms)[None, :, :], axis=-1))
            if d <= 5.0:
                contacts.append((rn, ("H", h_idx)))
                int_rec_res.add(rn)
                int_lig_res.add(("H", h_idx))
            elif d <= 10.0:
                int_rec_res.add(rn)
                int_lig_res.add(("H", h_idx))
        for l_idx, l_atoms in enumerate(nat_l_hvy_seq):
            d = np.min(np.linalg.norm(r_arr[:, None, :] - np.array(l_atoms)[None, :, :], axis=-1))
            if d <= 5.0:
                contacts.append((rn, ("L", l_idx)))
                int_rec_res.add(rn)
                int_lig_res.add(("L", l_idx))
            elif d <= 10.0:
                int_rec_res.add(rn)
                int_lig_res.add(("L", l_idx))

    print(f"Native 9CLP Cryo-EM Ground Truth: {len(contacts)} contacts (<5.0 A), {len(int_rec_res)} interface residues on Ecarin.")

    # 2. Locate Prediction Models
    models = find_prediction_pdbs(args.models_dir)
    if not models:
        print("\n[ERROR] No AlphaFold prediction PDB files found.")
        print("Please check that your prediction job has completed and outputs are uncompressed.")
        return

    print(f"Found {len(models)} AlphaFold prediction models to evaluate.\n")

    # 3. Model Evaluation Loop
    results = []
    top_entry = None

    for idx, mpath in enumerate(models):
        m_atoms = parse_atoms(mpath)
        m_ca_a, m_hvy_a, m_bf_a = {}, {}, []
        m_ca_h, m_hvy_h_seq, m_bf_h = [], [], []
        cur_h_n, cur_h_at = None, []
        m_ca_l, m_hvy_l_seq, m_bf_l = [], [], []
        cur_l_n, cur_l_at = None, []

        for a in m_atoms:
            if a["el"] == "H":
                continue
            c, n, aname, xyz, bf = a["c"], a["n"], a["a"], a["xyz"], a["bf"]
            if c == "A":
                # Model chain A residue index 1..426 corresponds to physiological 191..616 (offset = 190)
                nat_n = n + 190
                if 195 <= nat_n <= 614:
                    if aname == "CA":
                        m_ca_a[nat_n] = xyz
                        m_bf_a.append(bf)
                    m_hvy_a.setdefault(nat_n, []).append(xyz)
            elif c == "B":
                if aname == "CA":
                    m_ca_h.append(xyz)
                    m_bf_h.append(bf)
                if n != cur_h_n:
                    if cur_h_at:
                        m_hvy_h_seq.append(cur_h_at)
                    cur_h_n, cur_h_at = n, []
                cur_h_at.append(xyz)
            elif c == "C":
                if aname == "CA":
                    m_ca_l.append(xyz)
                    m_bf_l.append(bf)
                if n != cur_l_n:
                    if cur_l_at:
                        m_hvy_l_seq.append(cur_l_at)
                    cur_l_n, cur_l_at = n, []
                cur_l_at.append(xyz)
        if cur_h_at:
            m_hvy_h_seq.append(cur_h_at)
        if cur_l_at:
            m_hvy_l_seq.append(cur_l_at)

        # Domain CA RMSDs
        mp_common = sorted(set(range(195, 388)) & set(m_ca_a.keys()) & set(nat_ca_a.keys()))
        dis_common = sorted(set(range(388, 486)) & set(m_ca_a.keys()) & set(nat_ca_a.keys()))
        crd_common = sorted(set(range(486, 615)) & set(m_ca_a.keys()) & set(nat_ca_a.keys()))
        full_common = sorted(set(range(195, 615)) & set(m_ca_a.keys()) & set(nat_ca_a.keys()))

        R_mp, t_mp = kabsch(np.array([m_ca_a[k] for k in mp_common]), np.array([nat_ca_a[k] for k in mp_common]))
        rmsd_mp = rmsd((R_mp @ np.array([m_ca_a[k] for k in mp_common]).T).T + t_mp, np.array([nat_ca_a[k] for k in mp_common]))

        R_dis, t_dis = kabsch(np.array([m_ca_a[k] for k in dis_common]), np.array([nat_ca_a[k] for k in dis_common]))
        rmsd_dis = rmsd((R_dis @ np.array([m_ca_a[k] for k in dis_common]).T).T + t_dis, np.array([nat_ca_a[k] for k in dis_common]))

        p_crd = np.array([m_ca_a[k] for k in crd_common])
        q_crd = np.array([nat_ca_a[k] for k in crd_common])
        R_crd, t_crd = kabsch(p_crd, q_crd)
        rmsd_crd = rmsd((R_crd @ p_crd.T).T + t_crd, q_crd)

        R_full, t_full = kabsch(np.array([m_ca_a[k] for k in full_common]), np.array([nat_ca_a[k] for k in full_common]))
        rmsd_full_enzyme = rmsd((R_full @ np.array([m_ca_a[k] for k in full_common]).T).T + t_full, np.array([nat_ca_a[k] for k in full_common]))

        # Native contact recovery (Fnat)
        recov = 0
        for rn, lig_desc in contacts:
            if rn not in m_hvy_a:
                continue
            ch, idx_seq = lig_desc
            m_lig = m_hvy_h_seq if ch == "H" else m_hvy_l_seq
            if idx_seq >= len(m_lig):
                continue
            d = np.min(np.linalg.norm(np.array(m_hvy_a[rn])[:, None, :] - np.array(m_lig[idx_seq])[None, :, :], axis=-1))
            if d <= 5.0:
                recov += 1
        Fnat = recov / len(contacts) if contacts else 0.0

        # Interface RMSD (iRMSD)
        int_r_common = sorted(int_rec_res & set(m_ca_a.keys()) & set(nat_ca_a.keys()))
        min_h = min(len(m_ca_h), len(nat_seq_ca_h))
        min_l = min(len(m_ca_l), len(nat_seq_ca_l))
        int_h = [i for i in range(min_h) if ("H", i) in int_lig_res]
        int_l = [i for i in range(min_l) if ("L", i) in int_lig_res]

        p_int = np.vstack([np.array([m_ca_a[k] for k in int_r_common])] +
                          ([np.array([m_ca_h[i] for i in int_h])] if int_h else []) +
                          ([np.array([m_ca_l[i] for i in int_l])] if int_l else []))
        q_int = np.vstack([np.array([nat_ca_a[k] for k in int_r_common])] +
                          ([np.array([nat_seq_ca_h[i] for i in int_h])] if int_h else []) +
                          ([np.array([nat_seq_ca_l[i] for i in int_l])] if int_l else []))
        R_int, t_int = kabsch(p_int, q_int)
        iRMSD = rmsd((R_int @ p_int.T).T + t_int, q_int)

        # Ligand RMSD (lRMSD) evaluated with receptor superimposition on CRD
        p_lig = np.vstack([np.array(m_ca_h[:min_h]), np.array(m_ca_l[:min_l])])
        q_lig = np.vstack([np.array(nat_seq_ca_h[:min_h]), np.array(nat_seq_ca_l[:min_l])])
        p_lig_on_rec = (R_crd @ p_lig.T).T + t_crd
        lRMSD = rmsd(p_lig_on_rec, q_lig)

        # DockQ score (Basu & Wallner, 2016)
        def f_dockq(rmsd_val, d1):
            return 1.0 / (1.0 + (rmsd_val / d1)**2)
        DockQ = (Fnat + f_dockq(iRMSD, 1.5) + f_dockq(lRMSD, 8.5)) / 3.0

        # CAPRI Classification
        if Fnat >= 0.5 and (lRMSD <= 1.0 or iRMSD <= 1.0):
            capri = "High"
        elif (Fnat >= 0.3 and (lRMSD <= 5.0 or iRMSD <= 2.0)) or (Fnat >= 0.5 and lRMSD > 1.0 and iRMSD > 1.0):
            capri = "Medium"
        elif (Fnat >= 0.1 and (lRMSD <= 10.0 or iRMSD <= 4.0)) or (Fnat >= 0.3 and lRMSD > 5.0 and iRMSD > 2.0):
            capri = "Acceptable"
        else:
            capri = "Incorrect"

        # Antibody internal folding RMSDs
        R_vh, t_vh = kabsch(np.array(m_ca_h[:min_h]), np.array(nat_seq_ca_h[:min_h]))
        rmsd_vh = rmsd((R_vh @ np.array(m_ca_h[:min_h]).T).T + t_vh, np.array(nat_seq_ca_h[:min_h]))
        R_vl, t_vl = kabsch(np.array(m_ca_l[:min_l]), np.array(nat_seq_ca_l[:min_l]))
        rmsd_vl = rmsd((R_vl @ np.array(m_ca_l[:min_l]).T).T + t_vl, np.array(nat_seq_ca_l[:min_l]))
        R_fv, t_fv = kabsch(p_lig, q_lig)
        rmsd_fv = rmsd((R_fv @ p_lig.T).T + t_fv, q_lig)

        # Extract confidence metrics from ColabFold JSON
        folder = os.path.dirname(mpath) or "."
        m_tag = re.search(r"model_\d", os.path.basename(mpath))
        tag = m_tag.group(0) if m_tag else ""
        iptm, ptm = None, None
        for jf in glob.glob(os.path.join(folder, "*.json")):
            if tag and tag in jf and "scores" in jf:
                try:
                    with open(jf) as f:
                        d = json.load(f)
                    iptm, ptm = d.get("iptm", iptm), d.get("ptm", ptm)
                except Exception:
                    pass
        if iptm is None:
            try:
                with open(os.path.join(folder, "ranking_debug.json")) as f:
                    d = json.load(f)
                for k, v in d.get("iptm", {}).items():
                    if tag in k:
                        iptm = float(v)
                for k, v in d.get("ptm", {}).items():
                    if tag in k:
                        ptm = float(v)
            except Exception:
                pass

        res = {
            "rank": f"rank_{idx+1:03d}",
            "file": os.path.basename(mpath),
            "DockQ": DockQ,
            "CAPRI": capri,
            "Fnat": Fnat,
            "recov": recov,
            "iRMSD": iRMSD,
            "lRMSD": lRMSD,
            "iptm": iptm,
            "ptm": ptm,
            "rmsd_mp": rmsd_mp,
            "rmsd_dis": rmsd_dis,
            "rmsd_crd": rmsd_crd,
            "rmsd_full": rmsd_full_enzyme,
            "rmsd_vh": rmsd_vh,
            "rmsd_vl": rmsd_vl,
            "rmsd_fv": rmsd_fv,
            "plddt_a": np.mean(m_bf_a) if m_bf_a else 0.0,
            "plddt_h": np.mean(m_bf_h) if m_bf_h else 0.0,
            "plddt_l": np.mean(m_bf_l) if m_bf_l else 0.0,
            "R_crd": R_crd,
            "t_crd": t_crd,
            "m_atoms": m_atoms
        }
        results.append(res)
        if idx == 0:
            top_entry = res

    # 4. Print Summary Tables
    print("=" * 80)
    print("       PHYSIOLOGICAL DOMAIN-BY-DOMAIN SANITY CHECK (FULL-LENGTH ECARIN)")
    print("=" * 80)
    print(f"Top Model: {top_entry['file']}")
    print("-" * 80)
    print(f"{'Domain / Component':<30} | {'Residues':<10} | {'True CA RMSD':<14} | {'Mean pLDDT':<10}")
    print("-" * 80)
    print(f"{'Metalloproteinase Domain':<30} | {'193 aa':<10} | {top_entry['rmsd_mp']:<5.2f} A        | {top_entry['plddt_a']:<5.1f}")
    print(f"{'Disintegrin-like Domain':<30} | {'98 aa':<10}  | {top_entry['rmsd_dis']:<5.2f} A        | {top_entry['plddt_a']:<5.1f}")
    print(f"{'Cysteine-Rich Domain (CRD)':<30} | {'129 aa':<10} | {top_entry['rmsd_crd']:<5.2f} A        | {top_entry['plddt_a']:<5.1f}")
    print(f"{'Full Ecarin Enzyme (Mature)':<30} | {'420 aa':<10} | {top_entry['rmsd_full']:<5.2f} A        | {top_entry['plddt_a']:<5.1f}")
    print("-" * 80)
    print(f"{'H11 Antibody VH Domain':<30} | {'120 aa':<10} | {top_entry['rmsd_vh']:<5.2f} A        | {top_entry['plddt_h']:<5.1f}")
    print(f"{'H11 Antibody VL Domain':<30} | {'110 aa':<10} | {top_entry['rmsd_vl']:<5.2f} A        | {top_entry['plddt_l']:<5.1f}")
    print(f"{'H11 Antibody Fv Dimer (VH+VL)':<30} | {'230 aa':<10} | {top_entry['rmsd_fv']:<5.2f} A        | {(top_entry['plddt_h']+top_entry['plddt_l'])/2:<5.1f}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print(f"             ALL {len(results)} MODELS EVALUATED (CAPRI BENCHMARK)")
    print("=" * 80)
    print(f"{'Rank':<10} | {'DockQ':<8} | {'CAPRI':<12} | {'Fnat':<8} | {'iRMSD':<9} | {'lRMSD':<9} | {'ipTM':<8} | {'pTM':<8}")
    print("-" * 80)
    for r in results:
        ip_s = f"{r['iptm']:.3f}" if r['iptm'] is not None else "N/A"
        pt_s = f"{r['ptm']:.3f}" if r['ptm'] is not None else "N/A"
        print(f"{r['rank']:<10} | {r['DockQ']:<8.3f} | {r['CAPRI']:<12} | {r['Fnat']:<8.3f} | {r['iRMSD']:<5.2f} A   | {r['lRMSD']:<5.2f} A   | {ip_s:<8} | {pt_s:<8}")
    print("=" * 80)

    # 5. Write Superimposed 3D Overlay PDB
    def write_line(idx, a, res, c, n, xyz, bf=20.0, el="C"):
        aname = f" {a}" if len(a) < 4 else a
        return f"ATOM  {idx:5d} {aname:<4s} {res:>3s} {c:1s}{n:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00{bf:6.2f}          {el:>2s}\n"

    with open(args.overlay, "w") as f:
        f.write("REMARK   1 NATIVE 9CLP CRYO-EM: CHAIN A=Full Ecarin, H=VH, L=VL\n")
        f.write("REMARK   2 ALPHAFOLD 2 RANK 001 (Superimposed on CRD): CHAIN D=Ecarin, E=VH, F=VL\n")
        idx = 1
        for a in cif:
            if a["el"] != "H" and ((a["c"] == "A" and 195 <= a["n"] <= 614) or (a["c"] == "H") or (a["c"] == "L")):
                f.write(write_line(idx, a["a"], a["res"], a["c"], a["n"], a["xyz"], 20.0, a["el"]))
                idx = (idx % 99999) + 1
        f.write("TER\n")
        cmap = {"A": "D", "B": "E", "H": "E", "C": "F", "L": "F"}
        for a in top_entry["m_atoms"]:
            if a["el"] != "H" and a["c"] in cmap:
                t_xyz = (top_entry["R_crd"] @ a["xyz"]) + top_entry["t_crd"]
                disp_n = (a["n"] + 190) if a["c"] == "A" else a["n"]
                f.write(write_line(idx, a["a"], a["res"], cmap[a["c"]], disp_n, t_xyz, a["bf"], a["el"]))
                idx = (idx % 99999) + 1
        f.write("END\n")

    print(f"\nSuperimposed 3D overlay saved to: {args.overlay}")

    # 6. Save results.csv
    with open(args.out_csv, "w", newline="") as cf:
        fieldnames = [
            "rank", "file", "DockQ", "CAPRI", "Fnat", "recov", "iRMSD", "lRMSD", "iptm", "ptm",
            "rmsd_mp", "rmsd_dis", "rmsd_crd", "rmsd_full", "rmsd_vh", "rmsd_vl", "rmsd_fv",
            "plddt_a", "plddt_h", "plddt_l"
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k not in ["R_crd", "t_crd", "m_atoms"]}
            writer.writerow(row)
    print(f"Benchmark metrics saved to: {args.out_csv}\n")


if __name__ == "__main__":
    evaluate()
