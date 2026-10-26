# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY CLAHE ON THE FINAL DATASET — every image, every split
# For each image in dataset/{train,val,test}, re-runs your exact pipeline
# (resize -> LAB -> CLAHE on L -> BGR -> save) on the ORIGINAL file and compares.
#   match (diff ~ 0)   : exactly one CLAHE pass, correct size        -> OK
#   large diff         : e.g. CLAHE applied twice, or not at all      -> FLAGGED
#   wrong size         : raw / unprocessed image                      -> FLAGGED
#   no original found  : file not traceable to the original dataset   -> FLAGGED
# ═══════════════════════════════════════════════════════════════════════════════
import os
import cv2
import numpy as np

VERIFY = {
    "final_dataset":  "/home/auc/PlantDoc/Datasets/dataset",
    "original_train": "<< PATH TO ORIGINAL KAGGLE train/ >>",
    "original_test":  "<< PATH TO ORIGINAL KAGGLE test/ >>",
    "img_size": 224,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": (8, 8),
    "max_mean_diff": 0.5,      # 0-255 scale; one exact CLAHE pass gives ~0.0
}


def clahe_replica(path, cfg):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (cfg["img_size"], cfg["img_size"]))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=cfg["clahe_clip_limit"], tileGridSize=cfg["clahe_tile_grid"])
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    ext = os.path.splitext(path)[1].lower()
    ok, buf = cv2.imencode(ext if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp") else ".jpg", img)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)   # same encode/decode round-trip as the saved files


def verify(cfg):
    # Guard: raw originals come in many sizes; if the "original" folders are almost all
    # exactly img_size x img_size, they are already processed and the check is meaningless.
    for key in ("original_train", "original_test"):
        sizes = []
        for cls in sorted(os.listdir(cfg[key]))[:10]:
            for f in sorted(os.listdir(os.path.join(cfg[key], cls)))[:5]:
                im = cv2.imread(os.path.join(cfg[key], cls, f))
                if im is not None:
                    sizes.append(im.shape[:2])
        square = sum(s == (cfg["img_size"], cfg["img_size"]) for s in sizes)
        if sizes and square / len(sizes) > 0.9:
            raise RuntimeError(
                f"{key} = {cfg[key]}\n{square}/{len(sizes)} sampled images are already "
                f"{cfg['img_size']}x{cfg['img_size']} — this looks like a PROCESSED folder, not the raw "
                "Kaggle download. Use the same original paths you used in prepare_final_dataset.")
        print(f"{key}: sampled {len(sizes)} images, sizes vary as expected for raw originals "
              f"({len(set(sizes))} distinct sizes)")

    flagged, summary = [], {}
    for split, orig_root in [("train", cfg["original_train"]), ("val", cfg["original_train"]),
                             ("test", cfg["original_test"])]:
        root = os.path.join(cfg["final_dataset"], split)
        n, diffs = 0, []
        for cls in sorted(os.listdir(root)):
            for f in sorted(os.listdir(os.path.join(root, cls))):
                if f.startswith("."):
                    continue
                n += 1
                rel = f"{cls}/{f}"
                final = cv2.imread(os.path.join(root, cls, f))
                orig = os.path.join(orig_root, cls, f)
                if final is None:
                    flagged.append((split, rel, "unreadable"))
                    continue
                if final.shape[:2] != (cfg["img_size"], cfg["img_size"]):
                    flagged.append((split, rel, f"size {final.shape[1]}x{final.shape[0]} (raw/unprocessed?)"))
                    continue
                if not os.path.exists(orig):
                    flagged.append((split, rel, "no original with this class/filename"))
                    continue
                rep = clahe_replica(orig, cfg)
                d = float(np.abs(rep.astype(np.int16) - final.astype(np.int16)).mean())
                diffs.append(d)
                if d > cfg["max_mean_diff"]:
                    flagged.append((split, rel, f"mean diff {d:.2f} (CLAHE twice, or different processing)"))
        summary[split] = (n, max(diffs) if diffs else 0.0)
        print(f"{split:5s}: {n} images checked | worst diff among matches {summary[split][1]:.4f}")

    print(f"\nFlagged: {len(flagged)}")
    for s, rel, why in flagged[:30]:
        print(f"  {s}: {rel} -> {why}")
    if len(flagged) > 30:
        print(f"  ... and {len(flagged) - 30} more")
    print("\nRESULT: " + ("every image is exactly one CLAHE pass of its original — OK"
                          if not flagged else "FIX THE FLAGGED IMAGES BEFORE TRAINING"))
    return flagged


flagged = verify(VERIFY)
