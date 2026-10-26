# ═══════════════════════════════════════════════════════════════════════════════
# PREPARE FINAL DATASET — PlantDoc (Kaggle, 2,315 train / 236 test)
#
# Order:
#   1. Duplicate / leakage check on the ORIGINAL dataset (exact MD5 + perceptual hash).
#      Near-duplicate candidates are written to a review sheet; the script stops until
#      every candidate has a decision. Produces an EXCLUSION LIST — nothing on disk is
#      deleted, neither in the original nor in your current dataset.
#   2. Compare the CURRENT (cleaned, CLAHE) dataset with the ORIGINAL by class/filename.
#   3. TEST: every original test image is kept. Images missing from the current set are
#      restored from the original with the exact same resize + CLAHE pipeline.
#   4. TRAIN: images removed during cleaning stay removed; images on the exclusion list
#      are removed; anything unexplained is reported.
#   5. Verify counts (fails loudly if test != expected).
#   6. Stratified 10% validation split from the final train set, keeping near-duplicate
#      pairs on the same side.
#
# Everything is written to a NEW folder (OUTPUT_DIR/dataset/{train,val,test}) plus
# reports in OUTPUT_DIR/reports. Re-running is safe.
#
# Duplicate rules (test is never modified):
#   - group contains a test image       -> remove all TRAIN members of the group
#   - train-only group, one class       -> keep one copy (largest resolution), remove rest
#   - train-only group, several classes -> remove all (label is ambiguous)
#   - several test images in one group  -> flagged only (reported in limitations)
# ═══════════════════════════════════════════════════════════════════════════════
import csv
import hashlib
import os
import random
import shutil
from collections import defaultdict

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageDraw

CONFIG = {
    # ── paths ──
    "original_train": "<< PATH TO ORIGINAL KAGGLE train/ >>",
    "original_test":  "<< PATH TO ORIGINAL KAGGLE test/ >>",
    "current_train":  "<< PATH TO CURRENT CLEANED CLAHE train/ >>",
    "current_test":   "<< PATH TO CURRENT CLEANED CLAHE test/ >>",
    "output_dir":     "<< PATH TO NEW OUTPUT FOLDER >>",

    # ── CLAHE pipeline (must match apply_clahe_offline exactly) ──
    "img_size": 384,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": (8, 8),

    # ── duplicates ──
    "phash_threshold": 8,          # Hamming distance (of 64 bits) to flag a near-duplicate candidate

    # ── split ──
    "val_fraction": 0.10,
    "seed": 42,
    "expected_test_count": 236,

    # ── safety ──
    "fidelity_samples": 10,         # existing test images re-processed to verify the CLAHE replica
    "fidelity_max_mean_diff": 0.5,  # mean abs pixel difference allowed (0-255 scale)
    "include_unexplained_train": False,  # current-train files with no original counterpart
    "overwrite_output": False,
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff", ".gif"}


# ── helpers ────────────────────────────────────────────────────────────────────
def list_images(root):
    """{'class/filename': absolute_path} for every image directly inside class folders."""
    out = {}
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if f.startswith(".") or os.path.splitext(f)[1].lower() not in IMG_EXT:
                continue
            out[f"{cls}/{f}"] = os.path.join(cdir, f)
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_bytes(path):
    with Image.open(path) as im:
        h = imagehash.phash(im.convert("RGB"))
    return np.packbits(h.hash.flatten())          # 8 bytes


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def image_size(path):
    try:
        with Image.open(path) as im:
            return im.size[0] * im.size[1]
    except Exception:
        return 0


def read_bgr(path):
    """cv2.imread like the original pipeline; PIL fallback only for files cv2 cannot decode."""
    img = cv2.imread(path)
    if img is not None:
        return img, False
    with Image.open(path) as im:
        return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR), True


def clahe_process(path, cfg):
    """Exact replica of apply_clahe_offline(): resize -> LAB -> CLAHE on L -> BGR."""
    img, used_fallback = read_bgr(path)
    img = cv2.resize(img, (cfg["img_size"], cfg["img_size"]))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=cfg["clahe_clip_limit"], tileGridSize=cfg["clahe_tile_grid"])
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), used_fallback


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def side_by_side(path_a, label_a, path_b, label_b, out_path, height=300):
    tiles = []
    for p, lab in [(path_a, label_a), (path_b, label_b)]:
        with Image.open(p) as im:
            im = im.convert("RGB")
            w = max(1, int(im.width * height / im.height))
            im = im.resize((w, height))
        canvas = Image.new("RGB", (w, height + 40), "white")
        canvas.paste(im, (0, 40))
        ImageDraw.Draw(canvas).text((5, 5), lab[:80], fill="black")
        tiles.append(canvas)
    total = Image.new("RGB", (tiles[0].width + tiles[1].width + 10, height + 40), "white")
    total.paste(tiles[0], (0, 0))
    total.paste(tiles[1], (tiles[0].width + 10, 0))
    total.save(out_path, quality=90)


# ── step 1: duplicates on the original ─────────────────────────────────────────
def step1_duplicates(cfg, report_dir):
    orig = {"train": list_images(cfg["original_train"]), "test": list_images(cfg["original_test"])}
    items = [{"split": s, "rel": r, "path": p, "cls": r.split("/")[0]}
             for s in ("train", "test") for r, p in orig[s].items()]
    print(f"[1] Original: {len(orig['train'])} train, {len(orig['test'])} test")

    for it in items:
        it["md5"] = md5(it["path"])
    hashes = np.stack([phash_bytes(it["path"]) for it in items])
    n = len(items)

    exact_edges = []
    by_md5 = defaultdict(list)
    for i, it in enumerate(items):
        by_md5[it["md5"]].append(i)
    for idxs in by_md5.values():
        exact_edges += [(idxs[0], j) for j in idxs[1:]]

    candidates = []
    for i in range(n - 1):
        dist = _POPCOUNT[np.bitwise_xor(hashes[i + 1:], hashes[i])].sum(axis=1)
        for k in np.nonzero(dist <= cfg["phash_threshold"])[0]:
            j = i + 1 + int(k)
            if items[i]["md5"] != items[j]["md5"]:
                candidates.append((i, j, int(dist[k])))
    print(f"[1] Exact duplicate links: {len(exact_edges)} | near-duplicate candidates: {len(candidates)}")

    # ── review sheet ──
    review_csv = os.path.join(report_dir, "near_duplicate_review.csv")
    img_dir = os.path.join(report_dir, "near_duplicate_review_images")
    os.makedirs(img_dir, exist_ok=True)
    decisions = {}
    if os.path.exists(review_csv):
        with open(review_csv, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                decisions[(r["a_split"], r["a_rel"], r["b_split"], r["b_rel"])] = r["is_duplicate"].strip().lower()

    rows, undecided = [], 0
    for pid, (i, j, d) in enumerate(candidates):
        a, b = items[i], items[j]
        key = (a["split"], a["rel"], b["split"], b["rel"])
        dec = decisions.get(key, "")
        img_name = f"pair_{pid:04d}_d{d}.jpg"
        if not os.path.exists(os.path.join(img_dir, img_name)):
            side_by_side(a["path"], f"{a['split']}: {a['rel']}", b["path"], f"{b['split']}: {b['rel']}",
                         os.path.join(img_dir, img_name))
        if dec not in ("y", "n"):
            undecided += 1
        rows.append({"pair_id": pid, "image": img_name, "hamming": d, "a_split": a["split"], "a_rel": a["rel"],
                     "b_split": b["split"], "b_rel": b["rel"], "is_duplicate": dec})
    write_csv(review_csv, rows, ["pair_id", "image", "hamming", "a_split", "a_rel", "b_split", "b_rel",
                                 "is_duplicate"])
    if undecided:
        print(f"\n[1] STOP: {undecided} near-duplicate pair(s) need a decision.")
        print(f"    Open {img_dir}, then fill the 'is_duplicate' column in\n    {review_csv}\n"
              f"    with y (same photo) or n (different photo), and run again.")
        return None

    # ── groups from confirmed duplicates ──
    uf = UnionFind(n)
    for i, j in exact_edges:
        uf.union(i, j)
    similar_pairs = []
    for i, j, d in candidates:
        key = (items[i]["split"], items[i]["rel"], items[j]["split"], items[j]["rel"])
        if decisions[key] == "y":
            uf.union(i, j)
        else:
            similar_pairs.append((i, j))

    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    groups = [g for g in groups.values() if len(g) > 1]

    exclusion, flags = [], []
    for gid, g in enumerate(groups):
        members = [items[i] for i in g]
        test_m = [m for m in members if m["split"] == "test"]
        train_m = [m for m in members if m["split"] == "train"]
        classes = sorted({m["cls"] for m in members})
        if len(test_m) > 1:
            flags.append({"group": gid, "issue": "duplicate inside test set",
                          "files": "; ".join(f"{m['split']}:{m['rel']}" for m in members)})
        if test_m:
            if len({m["cls"] for m in members}) > 1:
                flags.append({"group": gid, "issue": "label conflict involving a test image",
                              "files": "; ".join(f"{m['split']}:{m['rel']}" for m in members)})
            for m in train_m:
                exclusion.append({"rel": m["rel"], "reason": "duplicate of a test image", "group": gid,
                                  "classes": "|".join(classes)})
        elif len(classes) > 1:
            for m in train_m:
                exclusion.append({"rel": m["rel"], "reason": "cross-class duplicate within train (ambiguous label)",
                                  "group": gid, "classes": "|".join(classes)})
        else:
            keep = max(train_m, key=lambda m: (image_size(m["path"]), m["rel"]))
            for m in train_m:
                if m is not keep:
                    exclusion.append({"rel": m["rel"], "reason": f"same-class duplicate within train (kept {keep['rel']})",
                                      "group": gid, "classes": "|".join(classes)})

    write_csv(os.path.join(report_dir, "exclusion_list.csv"), exclusion, ["rel", "reason", "group", "classes"])
    write_csv(os.path.join(report_dir, "duplicate_flags.csv"), flags, ["group", "issue", "files"])
    print(f"[1] Duplicate groups: {len(groups)} | train images to exclude: {len(exclusion)} | flags: {len(flags)}")

    similar_train = [(items[i]["rel"], items[j]["rel"]) for i, j in similar_pairs
                     if items[i]["split"] == "train" and items[j]["split"] == "train"]
    return {"orig": orig, "exclusion": {e["rel"]: e["reason"] for e in exclusion},
            "flags": flags, "similar_train": similar_train}


# ── step 2–4: reconcile current dataset with original ──────────────────────────
def fidelity_check(cfg, orig_test, cur_test, rng):
    common = sorted(set(orig_test) & set(cur_test))
    sample = rng.sample(common, min(cfg["fidelity_samples"], len(common)))
    diffs, unreadable, wrong_size = [], [], []
    for rel in sample:
        arr, _ = clahe_process(orig_test[rel], cfg)
        ext = os.path.splitext(rel)[1].lower() or ".jpg"
        ok, buf = cv2.imencode(ext if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp") else ".jpg", arr)
        replica = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        existing = cv2.imread(cur_test[rel])
        if existing is None:
            unreadable.append(cur_test[rel])
            continue
        if existing.shape != replica.shape:
            wrong_size.append((rel, existing.shape[:2]))
            continue
        diffs.append(float(np.abs(replica.astype(np.int16) - existing.astype(np.int16)).mean()))

    if unreadable:
        raise RuntimeError(
            f"OpenCV could not read {len(unreadable)} of {len(sample)} current test images, e.g.\n  "
            + "\n  ".join(unreadable[:3])
            + "\nCheck that current_test points at the CLAHE test folder and that these are image files.")
    if wrong_size:
        sizes = sorted({s for _, s in wrong_size})
        raise RuntimeError(
            f"{len(wrong_size)} of {len(sample)} current test images are not {cfg['img_size']}x{cfg['img_size']}. "
            f"Sizes found (height, width): {sizes}, e.g. {wrong_size[0][0]}.\n"
            "Either current_test is not the CLAHE-processed folder, or the dataset was built with a "
            "different img_size — set img_size in CONFIG to match.")

    worst = max(diffs) if diffs else 0.0
    print(f"[3] CLAHE fidelity check on {len(sample)} existing test images: "
          f"mean abs diff = {np.mean(diffs):.4f}, worst = {worst:.4f}")
    if worst > cfg["fidelity_max_mean_diff"]:
        raise RuntimeError(
            "Restored images would NOT match your existing CLAHE images (worst diff "
            f"{worst:.3f}). Check img_size / clahe_clip_limit / clahe_tile_grid in CONFIG "
            "against the values used when the dataset was built.")


def run(cfg):
    out_root = cfg["output_dir"]
    ds_dir, report_dir = os.path.join(out_root, "dataset"), os.path.join(out_root, "reports")
    if os.path.exists(ds_dir):
        if not cfg["overwrite_output"]:
            raise RuntimeError(f"{ds_dir} already exists. Delete it or set overwrite_output=True.")
        shutil.rmtree(ds_dir)
    os.makedirs(report_dir, exist_ok=True)
    rng = random.Random(cfg["seed"])

    s1 = step1_duplicates(cfg, report_dir)
    if s1 is None:
        return None
    orig, exclusion = s1["orig"], s1["exclusion"]

    # ── step 2: compare ──
    cur = {"train": list_images(cfg["current_train"]), "test": list_images(cfg["current_test"])}
    print(f"[2] Current: {len(cur['train'])} train, {len(cur['test'])} test")
    moved = []
    for split in ("train", "test"):
        orig_by_name = defaultdict(list)
        for rel in orig[split]:
            orig_by_name[rel.split("/", 1)[1]].append(rel.split("/")[0])
        for rel in cur[split]:
            cls, name = rel.split("/", 1)
            if rel not in orig[split] and name in orig_by_name and cls not in orig_by_name[name]:
                moved.append({"split": split, "current": rel, "original_class": "|".join(orig_by_name[name])})
    if moved:
        print(f"[2] WARNING: {len(moved)} file(s) are in a different class folder than in the original "
              "(possible relabel) — see moved_files.csv")
    write_csv(os.path.join(report_dir, "moved_files.csv"), moved, ["split", "current", "original_class"])

    manifest = []

    def place(src, split, rel, write_array=None):
        dst = os.path.join(ds_dir, split, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if write_array is not None:
            if not cv2.imwrite(dst, write_array):
                raise RuntimeError(f"cv2.imwrite failed for {dst}")
        else:
            shutil.copy2(src, dst)

    # ── step 3: test ──
    fidelity_check(cfg, orig["test"], cur["test"], rng)
    restored, fallback_used = [], []
    for rel, opath in sorted(orig["test"].items()):
        if rel in cur["test"]:
            place(cur["test"][rel], "test", rel)
            manifest.append({"rel": rel, "original_split": "test", "final_split": "test", "status": "kept"})
        else:
            arr, fb = clahe_process(opath, cfg)
            place(None, "test", rel, write_array=arr)
            restored.append(rel)
            if fb:
                fallback_used.append(rel)
            manifest.append({"rel": rel, "original_split": "test", "final_split": "test",
                             "status": "restored from original (resize + CLAHE)"})
    extra_test = sorted(set(cur["test"]) - set(orig["test"]))
    for rel in extra_test:
        manifest.append({"rel": rel, "original_split": "", "final_split": "excluded",
                         "status": "in current test but not in original test — not used"})
    print(f"[3] Test: {len(orig['test'])} total | restored {len(restored)} | "
          f"not-in-original (dropped) {len(extra_test)} | PIL fallback {len(fallback_used)}")

    # ── step 4: train ──
    final_train, removed_by_leakage, removed_by_cleaning, unexplained = [], [], [], []
    for rel in sorted(orig["train"]):
        if rel in exclusion:
            removed_by_leakage.append(rel)
            manifest.append({"rel": rel, "original_split": "train", "final_split": "excluded",
                             "status": f"step 1: {exclusion[rel]}"})
        elif rel not in cur["train"]:
            removed_by_cleaning.append(rel)
            manifest.append({"rel": rel, "original_split": "train", "final_split": "excluded",
                             "status": "removed during earlier cleaning"})
        else:
            final_train.append(rel)
    for rel in sorted(set(cur["train"]) - set(orig["train"])):
        unexplained.append(rel)
        if cfg["include_unexplained_train"]:
            final_train.append(rel)
        manifest.append({"rel": rel, "original_split": "", "final_split": "train" if cfg["include_unexplained_train"] else "excluded",
                         "status": "in current train but not in original train — review"})
    newly_removed = [r for r in removed_by_leakage if r in cur["train"]]
    print(f"[4] Train: {len(final_train)} kept | removed earlier by cleaning {len(removed_by_cleaning)} | "
          f"excluded by step 1 {len(removed_by_leakage)} (of which still present in your current set: "
          f"{len(newly_removed)}) | unexplained {len(unexplained)}")

    # ── step 6: stratified, grouped validation split ──
    in_train = set(final_train)
    uf_idx = {rel: i for i, rel in enumerate(final_train)}
    uf = UnionFind(len(final_train))
    for a, b in s1["similar_train"]:
        if a in in_train and b in in_train:
            uf.union(uf_idx[a], uf_idx[b])
    groups = defaultdict(list)
    for rel in final_train:
        groups[uf.find(uf_idx[rel])].append(rel)
    by_class = defaultdict(list)
    for g in groups.values():
        g.sort()
        by_class[g[0].split("/")[0]].append(g)

    val = set()
    for cls in sorted(by_class):
        glist = sorted(by_class[cls])
        n_cls = sum(len(g) for g in glist)
        target = round(cfg["val_fraction"] * n_cls)
        if n_cls >= 2:
            target = max(1, target)
        rng.shuffle(glist)
        taken = 0
        for g in glist:
            if taken + len(g) <= target:
                val.update(g)
                taken += len(g)
            if taken >= target:
                break

    for rel in final_train:
        split = "val" if rel in val else "train"
        place(cur["train"].get(rel) or cur["train"][rel], split, rel)
        manifest.append({"rel": rel, "original_split": "train" if rel in orig["train"] else "",
                         "final_split": split, "status": "kept"})

    # ── step 5: verify + counts ──
    counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for r in manifest:
        if r["final_split"] in ("train", "val", "test"):
            counts[r["rel"].split("/")[0]][r["final_split"]] += 1
    rows = [{"class": c, **counts[c]} for c in sorted(counts)]
    totals = {k: sum(r[k] for r in rows) for k in ("train", "val", "test")}
    rows.append({"class": "TOTAL", **totals})
    write_csv(os.path.join(report_dir, "counts.csv"), rows, ["class", "train", "val", "test"])
    write_csv(os.path.join(report_dir, "manifest.csv"), manifest, ["rel", "original_split", "final_split", "status"])

    print(f"\n[5] FINAL: train {totals['train']} | val {totals['val']} | test {totals['test']}")
    if totals["test"] != cfg["expected_test_count"]:
        raise RuntimeError(f"Test has {totals['test']} images, expected {cfg['expected_test_count']}.")
    on_disk = len(list_images(os.path.join(ds_dir, "test")))
    if on_disk != totals["test"]:
        raise RuntimeError(f"Test folder on disk has {on_disk} images, manifest says {totals['test']}.")
    empty_val = [r["class"] for r in rows[:-1] if r["val"] == 0]
    if empty_val:
        print(f"[5] NOTE: classes with no validation image (too few train images): {empty_val}")
    print(f"[5] Dataset: {ds_dir}\n[5] Reports: {report_dir}")
    return {"totals": totals, "restored": restored, "excluded_by_step1": removed_by_leakage,
            "newly_removed": newly_removed, "unexplained": unexplained, "flags": s1["flags"], "moved": moved}


if __name__ == "__main__":
    run(CONFIG)
